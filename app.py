"""Meter-reading data-prep app.

Two tools in one local web server:

  * Sample filtering  - review all photos, keep/reject for AI training,
                        export a curated dataset + manifest.
  * Auto-crop         - detect and crop the reading display; adjust the box
                        by hand; export crops and YOLO-OBB labels. If a
                        trained model exists (runs/obb or runs/detect .../
                        best.pt) it is used for detection instead of the
                        heuristic (auto_crop.py).

Run:  .venv/bin/python app.py   (then open http://127.0.0.1:5000)
"""
import io
import json
import math
import os
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

from auto_crop import (
    box_to_quad,
    corners_to_label,
    detect_display_obb,
    quad_to_box,
)
from mid_roll_detector import detect_mid_roll

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR
FILTER_STATE = BASE_DIR / "filter_progress.json"
FILTERED_DIR = BASE_DIR / "filtered"
CROPS_DIR = BASE_DIR / "crops"
YOLO_DATASET_DIR = BASE_DIR / "yolo_dataset"
TEST_RESULTS_PATH = BASE_DIR / "test_results.json"
REVIEW_STATE_PATH = BASE_DIR / "review_state.json"
# Trained weights can live under runs/obb or runs/detect depending on the task
# used to train (OBB checkpoints write to runs/obb).
# Paths are checked in order; the first existing file wins.
MODEL_PATHS = [
    BASE_DIR / "runs" / "obb" / "train-2" / "weights" / "best.pt",
    BASE_DIR / "runs" / "obb" / "train" / "weights" / "best.pt",
    BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt",
]

# Meter type classes used as YOLO class IDs in OBB labels.
# Each meter type gets its own class so the model can distinguish them.
METER_CLASSES = {
    "co":     {"id": 0, "label": "Cơ"},
    "dientu": {"id": 1, "label": "Điện tử"},
}
DEFAULT_METER_CLASS = "co"
# Path that stores per-image meter-type decisions (separate from filter state)
METER_STATE_PATH = BASE_DIR / "meter_classes.json"

REASON_LABELS = {
    "keep": "Good sample",
    "reject-blur": "Blurry",
    "reject-dark": "Too dark / unreadable",
    "reject-partial": "Meter cut off / occluded",
    "reject-nometer": "Not a meter",
    "reject-dup": "Duplicate / near-identical",
    "reject-other": "Other problem",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class LetterboxPad:
    def __init__(self, target_size=(72, 128)):
        self.target_w, self.target_h = target_size

    def __call__(self, img):
        w, h = img.size
        scale = min(self.target_w / w, self.target_h / h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img_resized = img.resize((new_w, new_h))
        canvas = Image.new("RGB", (self.target_w, self.target_h), (0, 0, 0))
        paste_x = (self.target_w - new_w) // 2
        paste_y = (self.target_h - new_h) // 2
        canvas.paste(img_resized, (paste_x, paste_y))
        return canvas

class MultiTaskDigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.digit_head = nn.Linear(256, num_classes)
        self.half_head = nn.Linear(256, 1)
        self.decimal_head = nn.Linear(256, 1)

    def forward(self, x):
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        d_logits = self.digit_head(feat)
        h_logit = self.half_head(feat).squeeze(-1)
        dec_logit = self.decimal_head(feat).squeeze(-1)
        return d_logits, h_logit, dec_logit

TinyDigitCNN = MultiTaskDigitCNN  # Backward compatibility alias

_ocr_model = None
_ocr_transform = None


_parseq_model = None
_parseq_transform = None
_parseq_loaded = False

def get_parseq_model():
    global _parseq_model, _parseq_transform, _parseq_loaded
    if _parseq_loaded:
        return _parseq_model, _parseq_transform
    _parseq_loaded = True
    device = "cuda" if _device_ok() else "cpu"
    try:
        import torch
        from torchvision import transforms
        path = BASE_DIR / "best_parseq_elec.pt"
        if path.exists():
            _parseq_model = torch.hub.load('baudm/parseq', 'parseq', pretrained=False).to(device).eval()
            _parseq_model.load_state_dict(torch.load(path, map_location=device))
            print(f"Loaded tuned PARSeq from {path}")
            _parseq_transform = transforms.Compose([
                transforms.Resize((32, 128), transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])
    except Exception as e:
        print(f"Could not load tuned PARSeq model: {e}")
    return _parseq_model, _parseq_transform

def get_ocr_model():
    global _ocr_model, _ocr_transform
    if _ocr_model is not None:
        return _ocr_model, _ocr_transform
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskDigitCNN(num_classes=10)
    
    path = BASE_DIR / "output_digit_clf" / "best_multitask_digit_clf.pt"
    if not path.exists():
        path = BASE_DIR / "output_digit_clf" / "best_digit_clf.pt"

    if path.exists():
        try:
            state = torch.load(str(path), map_location=device)
            model.load_state_dict(state, strict=False)
            model.to(device)
            model.eval()
            _ocr_model = model
            _ocr_transform = transforms.Compose([
                LetterboxPad((72, 128)),
                transforms.ToTensor()
            ])
            print(f"Loaded Multi-Task OCR model from {path}")
        except Exception as e:
            print(f"Failed to load OCR model: {e}")
            _ocr_model = None
            _ocr_transform = None
    else:
        print(f"OCR model not found at {path}")
        
    return _ocr_model, _ocr_transform

TEST_IMAGES_DIR = Path("/home/deist/Downloads/OCR/testocr")
if not TEST_IMAGES_DIR.exists():
    TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

_model = None
_model_loaded = False


def _image_names():
    """Return image filenames to show in the labeling tool.

    If a filter_progress.json exists (from the Filter-samples tab) only images
    explicitly marked 'keep' are returned – duplicates, blurry shots, and other
    rejected images are hidden so the labeler only ever sees good samples.
    Falls back to all images when no filter state has been saved yet.
    """
    all_names = sorted(f.name for f in IMAGES_DIR.iterdir()
                       if f.suffix.lower() in IMAGE_EXTS)
    state = _load_state()
    if not state:
        return all_names
    keep = [n for n in all_names if state.get(n, {}).get("label") == "keep"]
    return keep if keep else all_names


def _load_state():
    if FILTER_STATE.exists():
        try:
            return json.loads(FILTER_STATE.read_text())
        except Exception:
            pass
    return {}


def _save_state(state):
    FILTER_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1))


def _load_meter_state():
    """Return dict {image_name: meter_class_key} from meter_classes.json."""
    if METER_STATE_PATH.exists():
        try:
            return json.loads(METER_STATE_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_meter_state(mstate):
    METER_STATE_PATH.write_text(json.dumps(mstate, ensure_ascii=False, indent=1))


def _best_model_path():
    for p in MODEL_PATHS:
        if p.exists():
            return p
    return None


_yolo_digits = None
_yolo_digits_loaded = False

def get_yolo_digits():
    """Lazy-load the digit bbox YOLO model."""
    global _yolo_digits, _yolo_digits_loaded
    if _yolo_digits_loaded:
        return _yolo_digits
    _yolo_digits_loaded = True
    path = BASE_DIR / "yolov8n-digits-bbox.pt"
    if path.exists():
        try:
            from ultralytics import YOLO
            _yolo_digits = YOLO(str(path))
            print(f"Loaded digit bbox model from {path}")
        except Exception as e:
            print(f"Could not load digit bbox model: {e}")
    return _yolo_digits

_yolo_electronic = None
_yolo_electronic_loaded = False
def get_yolo_electronic():
    global _yolo_electronic, _yolo_electronic_loaded
    if _yolo_electronic_loaded:
        return _yolo_electronic
    _yolo_electronic_loaded = True
    path = BASE_DIR / "yolov8n-electronic-bbox.pt"
    if path.exists():
        try:
            from ultralytics import YOLO
            _yolo_electronic = YOLO(str(path))
            print(f"Loaded electronic bbox model from {path}")
        except Exception as e:
            print(f"Could not load electronic bbox model: {e}")
    return _yolo_electronic


def _load_model():
    """Load trained YOLO model if present; else None."""
    global _model, _model_loaded
    if _model_loaded:
        return _model
    path = _best_model_path()
    if path is not None:
        try:
            from ultralytics import YOLO
            _model = YOLO(str(path))
            _model.to("cuda" if _device_ok() else "cpu")
            _model_loaded = True
        except Exception as e:  # pragma: no cover
            print("Could not load YOLO model:", e)
            _model = None
    return _model


def _device_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
@app.route("/api/images")
def api_images():
    return jsonify({"names": _image_names(), "total": len(_image_names()),
                    "reasons": REASON_LABELS,
                    "meterClasses": METER_CLASSES,
                    "defaultMeterClass": DEFAULT_METER_CLASS,
                    "meterState": _load_meter_state()})


@app.route("/api/meter_class", methods=["POST"])
def api_set_meter_class():
    """Save or update the meter type for a single image.
    Body: {name: str, meterClass: str}
    """
    data = request.get_json(force=True)
    name = data.get("name")
    cls = data.get("meterClass")
    if name not in _image_names():
        return jsonify({"error": "unknown image"}), 400
    if cls not in METER_CLASSES:
        return jsonify({"error": "unknown meterClass"}), 400
    mstate = _load_meter_state()
    mstate[name] = cls
    _save_meter_state(mstate)
    return jsonify({"ok": True, "classId": METER_CLASSES[cls]["id"]})


@app.route("/images/<name>")
def serve_image(name):
    path = IMAGES_DIR / name
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
        return "not found", 404
    return send_file(str(path), mimetype="image/jpeg")


@app.route("/images/<name>/thumb")
def serve_thumb(name):
    path = IMAGES_DIR / name
    if not path.is_file():
        return "not found", 404
    img = Image.open(path)
    img.thumbnail((420, 560))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=78)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


# ---------------------------------------------------------------------------
# Filter tool
# ---------------------------------------------------------------------------
@app.route("/api/progress")
def api_progress():
    return jsonify({"state": _load_state()})


@app.route("/api/classify", methods=["POST"])
def api_classify():
    data = request.get_json(force=True)
    name = data.get("name")
    if name not in _image_names():
        return jsonify({"error": "unknown image"}), 400
    label = data.get("label")
    if label not in REASON_LABELS:
        return jsonify({"error": "unknown label"}), 400
    state = _load_state()
    state[name] = {"label": label, "ts": time.time()}
    _save_state(state)
    return jsonify({"ok": True})


@app.route("/api/classify/undo", methods=["POST"])
def api_undo():
    data = request.get_json(force=True)
    name = data.get("name")
    state = _load_state()
    if name in state:
        del state[name]
        _save_state(state)
    return jsonify({"ok": True})


@app.route("/api/stats")
def api_stats():
    state = _load_state()
    total = len(_image_names())
    done = len(state)
    by_label = {}
    for v in state.values():
        by_label[v["label"]] = by_label.get(v["label"], 0) + 1
    return jsonify({"total": total, "done": done, "remaining": total - done,
                    "by_label": by_label})


@app.route("/api/image_delete", methods=["POST"])
def api_image_delete():
    data = request.get_json(force=True)
    name = data.get("name")
    if name not in _image_names():
        return jsonify({"error": "unknown image"}), 400
    
    # Delete the image file
    img_path = IMAGES_DIR / name
    if img_path.is_file():
        img_path.unlink()
        
    # Delete from filter state
    fstate = _load_state()
    if name in fstate:
        del fstate[name]
        _save_state(fstate)
        
    # Delete from meter state
    mstate = _load_meter_state()
    if name in mstate:
        del mstate[name]
        _save_meter_state(mstate)
        
    # Delete label if exists
    lbl_path = _label_path(name)
    if lbl_path.is_file():
        lbl_path.unlink()
        
    # Delete from test results
    tstate = _load_test_results()
    if name in tstate:
        del tstate[name]
        _save_test_results(tstate)

    # Delete from review state
    rstate = _load_review_state()
    if name in rstate:
        del rstate[name]
        _save_review_state(rstate)
        
    return jsonify({"ok": True})


@app.route("/api/export", methods=["POST"])
def api_export():
    data = request.get_json(force=True) or {}
    mode = data.get("mode", "copy")          # copy | symlink
    keep_only = bool(data.get("keep_only", False))
    state = _load_state()
    keep = [n for n, v in state.items() if v["label"] == "keep"]
    reject = [n for n, v in state.items() if v["label"].startswith("reject")]

    FILTERED_DIR.mkdir(exist_ok=True)
    keep_dir = FILTERED_DIR / "keep"
    reject_dir = FILTERED_DIR / "reject"
    keep_dir.mkdir(exist_ok=True)
    reject_dir.mkdir(exist_ok=True)

    def place(names, out_dir):
        for n in names:
            src = IMAGES_DIR / n
            if not src.is_file():
                continue
            dst = out_dir / n
            if mode == "symlink":
                if dst.is_symlink() or dst.exists():
                    dst.unlink(missing_ok=True)
                os.symlink(src, dst)
            else:
                dst.write_bytes(src.read_bytes())

    place(keep, keep_dir)
    if not keep_only:
        place(reject, reject_dir)

    manifest = []
    for n in _image_names():
        v = state.get(n)
        manifest.append({"name": n, "label": v["label"] if v else "",
                         "reason": REASON_LABELS.get(v["label"], "") if v else "",
                         "decided": bool(v)})
    (FILTERED_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1))
    with open(FILTERED_DIR / "manifest.csv", "w") as fh:
        fh.write("name,label,reason,decided\n")
        for m in manifest:
            fh.write(f'{m["name"]},{m["label"]},{m["reason"]},{m["decided"]}\n')

    return jsonify({"ok": True, "keep": len(keep), "reject": len(reject),
                    "out": str(FILTERED_DIR)})


# ---------------------------------------------------------------------------
# Manual labeling tool (train YOLOv8 on the reading-display boxes)
# ---------------------------------------------------------------------------
def _image_for(name):
    """Return (path, PIL Image) for an image name."""
    path = IMAGES_DIR / name
    if not path.is_file():
        return None, None
    try:
        return path, Image.open(path).convert("RGB")
    except Exception:
        return None, None


def _label_path(name):
    return CROPS_DIR / "labels" / f"{Path(name).stem}.txt"


@app.route("/api/labeled")
def api_labeled():
    """List image stems that already have a saved YOLO label."""
    if not (CROPS_DIR / "labels").is_dir():
        return jsonify({"labels": []})
    labels = sorted(p.stem for p in (CROPS_DIR / "labels").glob("*.txt"))
    return jsonify({"labels": labels, "count": len(labels)})


@app.route("/api/label", methods=["POST"])
def api_label():
    """Save a YOLO-OBB label for the reading display box on a full image.
    Body: {name, box:{cx,cy,w,h,angleDeg}, meterClass?}
      - meterClass: one of the METER_CLASSES keys (default: DEFAULT_METER_CLASS)
    The legacy {box:[x0,y0,x1,y1]} shape is still accepted for backward
    compatibility (treated as axis-aligned with the default class)."""
    data = request.get_json(force=True)
    name = data.get("name")
    if name not in _image_names():
        return jsonify({"error": "unknown image"}), 400

    # Resolve meter class → YOLO class id
    cls_key = data.get("meterClass", DEFAULT_METER_CLASS)
    if cls_key not in METER_CLASSES:
        cls_key = DEFAULT_METER_CLASS
    class_id = METER_CLASSES[cls_key]["id"]

    b = data.get("box")
    if isinstance(b, (list, tuple)):  # legacy axis-aligned
        if not b or len(b) != 4:
            return jsonify({"error": "box must be {cx,cy,w,h,angleDeg}"}), 400
        x0, y0, x1, y1 = (float(v) for v in b)
        b = {"cx": (x0 + x1) / 2.0, "cy": (y0 + y1) / 2.0,
             "w": x1 - x0, "h": y1 - y0, "angleDeg": 0.0}
    if not isinstance(b, dict):
        return jsonify({"error": "box must be {cx,cy,w,h,angleDeg}"}), 400
    try:
        cx, cy, bw, bh = float(b["cx"]), float(b["cy"]), float(b["w"]), float(b["h"])
        ang = float(b.get("angleDeg", 0.0))
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "box needs cx,cy,w,h (angleDeg optional)"}), 400

    path, img = _image_for(name)
    if img is None:
        return jsonify({"error": "cannot open image"}), 400
    W, H = img.size
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(bw)
            and math.isfinite(bh) and math.isfinite(ang)) or bw < 0 or bh < 0:
        return jsonify({"error": "invalid box"}), 400
    cx, cy = max(0.0, min(W, cx)), max(0.0, min(H, cy))
    if bw < 3 or bh < 3:
        return jsonify({"error": "box too small"}), 400

    (CROPS_DIR / "labels").mkdir(parents=True, exist_ok=True)
    corners = box_to_quad(cx, cy, bw, bh, ang)
    nums = corners_to_label(corners, W, H)
    _label_path(name).write_text(
        f"{class_id} " + " ".join(f"{v:.6f}" for v in nums) + "\n")

    # Persist meter class choice for this image
    mstate = _load_meter_state()
    mstate[name] = cls_key
    _save_meter_state(mstate)

    return jsonify({"ok": True, "stem": Path(name).stem, "classId": class_id})


def _migrate_axis_labels(labels_dir=None):
    """Rewrite 5-col (axis-aligned YOLO detect) lines to 9-col OBB in place.

    Ultralytics refuses datasets that mix 5-col and 9-col rows, so any labels
    saved before OBB support must be upgraded before export/training.
    Returns the number of lines converted."""
    labels_dir = labels_dir or CROPS_DIR / "labels"
    if not labels_dir.is_dir():
        return 0
    n = 0
    for p in sorted(labels_dir.glob("*.txt")):
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        out, changed = [], False
        for ln in lines:
            tok = ln.split()
            if len(tok) == 5:
                cls, cx, cy, bw, bh = (float(t) for t in tok)
                x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                vals = [max(0.0, min(1.0, v)) for v in (x0, y0, x1, y0, x1, y1, x0, y1)]
                out.append(f"{int(cls)} " + " ".join(f"{v:.6f}" for v in vals))
                changed, n = True, n + 1
            else:
                out.append(ln)   # 9-col OBB (or unknown): leave as-is
        if changed:
            p.write_text("\n".join(out) + "\n")
    return n


@app.route("/api/label/delete", methods=["POST"])
def api_label_delete():
    data = request.get_json(force=True)
    name = data.get("name")
    p = _label_path(name)
    if p.is_file():
        p.unlink()
    return jsonify({"ok": True})


@app.route("/api/detect/<name>")
def api_detect(name):
    """Auto-suggest a starting box (OBB-aware). The user still confirms and
    adjusts it before saving.

    Response is always an oriented box description {kind, cx, cy, w, h,
    angleDeg, strategy, W, H} in image pixels (angleDeg in degrees). A trained
    OBB checkpoint yields a pre-rotated box; otherwise the heuristic runs and
    adds a tilt when it is confident the reading row is not axis-aligned."""
    if name not in _image_names():
        return jsonify({"error": "unknown image"}), 400
    img = Image.open(IMAGES_DIR / name).convert("RGB")
    W, H = img.size

    model = _load_model()
    if model is not None:
        results = model.predict(str(IMAGES_DIR / name), conf=0.25, verbose=False)
        r = results[0]
        obb = getattr(r, "obb", None)
        if obb is not None and len(obb) > 0:
            # xyxyxyxy is (N,4,2): four corner points [x,y]; take the top box.
            quad = [tuple(p) for p in obb.xyxyxyxy[0].tolist()]
            cx, cy, bw, bh, ang = quad_to_box(quad)
            class_id = int(obb.cls[0].item()) if getattr(obb, "cls", None) is not None else -1
            predicted_class_key = next((k for k, v in METER_CLASSES.items() if v["id"] == class_id), None)
            return jsonify({"kind": "obb", "cx": cx, "cy": cy, "w": bw,
                            "h": bh, "angleDeg": 0.0 if abs(ang) < 1 else ang,
                            "strategy": "yolo", "W": W, "H": H,
                            "meterClass": predicted_class_key})
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            x0, y0, x1, y1 = boxes.xyxy[0].tolist()
            return jsonify({"kind": "box", "cx": (x0 + x1) / 2.0,
                            "cy": (y0 + y1) / 2.0, "w": x1 - x0, "h": y1 - y0,
                            "angleDeg": 0.0, "strategy": "yolo",
                            "W": W, "H": H})

    import numpy as np
    det, strategy = detect_display_obb(np.asarray(img))
    if det is None:
        return jsonify({"kind": None, "cx": None, "cy": None, "w": None,
                        "h": None, "angleDeg": None, "strategy": strategy,
                        "W": W, "H": H})
    return jsonify({"kind": "obb", "cx": det["cx"], "cy": det["cy"],
                    "w": det["w"], "h": det["h"],
                    "angleDeg": det["angleDeg"], "strategy": strategy,
                    "W": W, "H": H})


@app.route("/api/export_yolo_dataset", methods=["POST"])
def api_export_yolo_dataset():
    """Assemble yolo_dataset/ (full images + labels + data.yaml) with a
    train/val split, ready for ultralytics (YOLOv8)."""
    data = request.get_json(force=True) or {}
    val_frac = float(data.get("val_frac", 0.15))
    labels_dir = CROPS_DIR / "labels"
    if not labels_dir.is_dir():
        return jsonify({"error": "no labels yet - label some images first"}), 400

    # upgrade any legacy 5-col rows first; OBB training rejects mixed formats
    migrated = _migrate_axis_labels(labels_dir)

    import random
    random.seed(42)
    stems = []
    for lab in labels_dir.glob("*.txt"):
        if (IMAGES_DIR / f"{lab.stem}.jpg").is_file():
            stems.append(lab.stem)
    if not stems:
        return jsonify({"error": "labels found but no matching images"}), 400
    random.shuffle(stems)
    n_val = max(1, int(len(stems) * val_frac))
    train_set, val_set = stems[n_val:], stems[:n_val]

    YOLO_DATASET_DIR.mkdir(exist_ok=True)
    for c in YOLO_DATASET_DIR.rglob("*.cache"):   # drop stale scans from old formats
        c.unlink(missing_ok=True)
    for split, files in (("train", train_set), ("val", val_set)):
        (YOLO_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        for s in files:
            (YOLO_DATASET_DIR / "images" / split / f"{s}.jpg").write_bytes(
                (IMAGES_DIR / f"{s}.jpg").read_bytes())
            (YOLO_DATASET_DIR / "labels" / split / f"{s}.txt").write_bytes(
                (labels_dir / f"{s}.txt").read_bytes())

    class_names = [METER_CLASSES[k]["label"] for k in sorted(METER_CLASSES.keys(), key=lambda x: METER_CLASSES[x]["id"])]
    names_yaml = ", ".join(f"'{name}'" for name in class_names)
    data_yaml = f"""path: {YOLO_DATASET_DIR}
train: images/train
val: images/val
nc: {len(class_names)}
names: [{names_yaml}]
"""
    (YOLO_DATASET_DIR / "data.yaml").write_text(data_yaml)
    return jsonify({"ok": True, "train": len(train_set), "val": len(val_set),
                    "out": str(YOLO_DATASET_DIR)})


@app.route("/api/yolo_status")
def api_yolo_status():
    return jsonify({"model": _best_model_path() is not None,
                    "labels": sum(1 for _ in (CROPS_DIR / "labels").glob("*.txt"))
                    if (CROPS_DIR / "labels").is_dir() else 0})


# ---------------------------------------------------------------------------
# Model test screen
# ---------------------------------------------------------------------------

def _test_images():
    """Images suitable for model testing: rejected by filter but NOT duplicates.
    Duplicates are excluded because they are effectively the same shot and
    would inflate recall metrics artificially."""
    state = _load_state()
    all_names = sorted(f.name for f in IMAGES_DIR.iterdir()
                       if f.suffix.lower() in IMAGE_EXTS)
    if not state:
        return []
    return [n for n in all_names
            if state.get(n, {}).get("label", "").startswith("reject-")
            and state.get(n, {}).get("label") != "reject-dup"]


def _load_test_results():
    if TEST_RESULTS_PATH.exists():
        try:
            return json.loads(TEST_RESULTS_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_test_results(results):
    TEST_RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=1))


@app.route("/api/test_images")
def api_test_images():
    names = _test_images()
    state = _load_state()
    results = _load_test_results()
    # Attach reject reason to each image
    images = []
    for n in names:
        images.append({
            "name": n,
            "rejectLabel": state.get(n, {}).get("label", ""),
            "result": results.get(n),
        })
    return jsonify({
        "images": images,
        "meterClasses": METER_CLASSES,
        "hasModel": _best_model_path() is not None,
    })


@app.route("/api/test_detect/<name>")
def api_test_detect(name):
    """Run YOLO OBB inference on a single test image. Returns box + class."""
    if name not in _test_images():
        return jsonify({"error": "not a valid test image"}), 400
    model = _load_model()
    if model is None:
        return jsonify({"error": "no trained model found"}), 503

    img = Image.open(IMAGES_DIR / name).convert("RGB")
    W, H = img.size

    results = model.predict(str(IMAGES_DIR / name), conf=0.20, verbose=False)
    r = results[0]
    obb = getattr(r, "obb", None)
    if obb is None or len(obb) == 0:
        return jsonify({"detected": False, "W": W, "H": H})

    # Return all detected boxes (sorted by confidence desc) for display
    boxes = []
    for i in range(len(obb)):
        quad = [tuple(p) for p in obb.xyxyxyxy[i].tolist()]
        cx, cy, bw, bh, ang = quad_to_box(quad)
        cls_id = int(obb.cls[i].item())
        conf = float(obb.conf[i].item())
        cls_key = next((k for k, v in METER_CLASSES.items() if v["id"] == cls_id), None)
        boxes.append({
            "cx": cx, "cy": cy, "w": bw, "h": bh, "angleDeg": ang,
            "classId": cls_id, "classKey": cls_key,
            "classLabel": METER_CLASSES.get(cls_key, {}).get("label", str(cls_id)),
            "conf": round(conf, 3),
        })
    boxes.sort(key=lambda b: -b["conf"])
    return jsonify({"detected": True, "boxes": boxes, "W": W, "H": H})


OCR_STATE_PATH = BASE_DIR / "ocr_labels.json"

def _load_ocr_state():
    if OCR_STATE_PATH.exists():
        try:
            return json.loads(OCR_STATE_PATH.read_text())
        except Exception:
            pass
    return {}

def _save_ocr_state(state):
    OCR_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=1))

@app.route("/api/ocr_images")
def api_ocr_images():
    fstate = _load_state()
    ostate = _load_ocr_state()
    
    if METER_STATE_PATH.exists():
        import json
        with open(METER_STATE_PATH) as f:
            meter_state = json.load(f)
    else:
        meter_state = {}
        
    images = []
    
    labeled = set(p.stem for p in (CROPS_DIR / "labels").glob("*.txt")) if (CROPS_DIR / "labels").is_dir() else set()
    
    for n in _image_names():
        stem = Path(n).stem
        if fstate.get(n, {}).get("label") == "keep" and stem in labeled:
            text = ostate.get(n, "")
            
            # Check minority conditions
            minority_reasons = []
            
            # 1. Abnormal text length
            if text and (len(text) < 6 or len(text) > 8):
                minority_reasons.append(f"Len={len(text)}")
                
            # 2. Check JSON data for box counts and unreadable
            jf = CROPS_DIR / "digits_data" / f"{stem}.json"
            if jf.exists():
                try:
                    import json
                    with open(jf) as f:
                        jdata = json.load(f)
                    digits = jdata.get("digits", [])
                    
                    if len(digits) not in (0, 1, 6):
                        minority_reasons.append(f"Boxes={len(digits)}")
                        
                    has_unreadable = any(d.get("label") == "unreadable" for d in digits)
                    if has_unreadable:
                        minority_reasons.append("Unreadable")
                except:
                    pass

            m_class = meter_state.get(n, "unknown")
            
            images.append({
                "name": n, 
                "text": text,
                "meter_class": m_class,
                "minority": ", ".join(minority_reasons) if minority_reasons else None
            })
    return jsonify({"images": images})

@app.route("/api/ocr_data/<name>")
def api_ocr_data(name):
    if name not in _image_names():
        return jsonify({"error": "unknown image"}), 400
    
    stem = Path(name).stem
    lbl_path = CROPS_DIR / "labels" / f"{stem}.txt"
    if not lbl_path.is_file():
        return jsonify({"error": "no label"}), 404
        
    # Read the first OBB from the label file
    lines = lbl_path.read_text().splitlines()
    if not lines:
        return jsonify({"error": "empty label"}), 404
        
    parts = lines[0].strip().split()
    if len(parts) >= 9:
        nums = [float(p) for p in parts[1:9]]
        # The coordinates are normalized. We return them as they are.
        return jsonify({"ok": True, "quad": nums, "text": _load_ocr_state().get(name, "")})
    return jsonify({"error": "invalid label format"}), 400

@app.route("/api/rectified_crop/<name>")
def api_rectified_crop(name):
    if name not in _image_names():
        return jsonify({"error": "unknown image"}), 400
    
    stem = Path(name).stem
    lbl_path = CROPS_DIR / "labels" / f"{stem}.txt"
    if not lbl_path.is_file():
        return jsonify({"error": "no label"}), 404
        
    lines = lbl_path.read_text().splitlines()
    if not lines:
        return jsonify({"error": "empty label"}), 404
        
    parts = lines[0].strip().split()
    if len(parts) >= 9:
        nums = [float(p) for p in parts[1:9]]
        img_path = IMAGES_DIR / name
        import cv2
        import numpy as np
        img = cv2.imread(str(img_path))
        if img is None:
            return jsonify({"error": "could not read image"}), 500
            
        H, W = img.shape[:2]
        pts = np.array(nums).reshape(4, 2)
        pts[:, 0] *= W
        pts[:, 1] *= H
        pts = pts.astype(np.float32)
        
        w1 = np.linalg.norm(pts[0] - pts[1])
        w2 = np.linalg.norm(pts[2] - pts[3])
        h1 = np.linalg.norm(pts[1] - pts[2])
        h2 = np.linalg.norm(pts[3] - pts[0])
        
        max_w = max(int(w1), int(w2))
        max_h = max(int(h1), int(h2))
        
        # Add some padding to crop to avoid cutting off digits
        pad = int(max_h * 0.1)
        
        dst_pts = np.array([
            [pad, pad],
            [max_w - 1 + pad, pad],
            [max_w - 1 + pad, max_h - 1 + pad],
            [pad, max_h - 1 + pad]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (max_w + 2*pad, max_h + 2*pad))
        
        # Base64 encode
        ret, buf = cv2.imencode('.jpg', warped)
        import base64
        b64 = base64.b64encode(buf).decode('utf-8')
        
        # Check if we have existing digit labels
        digits_path = CROPS_DIR / "digits_data" / f"{stem}.json"
        import torch
    digits = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    M_inv = np.linalg.inv(M)
    
    if cls_id == 1:
        # Electronic uses PARSeq
        parseq_model, parseq_transform = get_parseq_model()
        from PIL import Image
        import torch
        for i, bx in enumerate(boxes):
            x1_f, y1_f, x2_f, y2_f = bx["x1"], bx["y1"], bx["x2"], bx["y2"]
            cw_pts = np.array([[x1_f, y1_f], [x2_f, y1_f], [x2_f, y2_f], [x1_f, y2_f]], dtype=np.float32).reshape(-1, 1, 2)
            c_orig = cv2.perspectiveTransform(cw_pts, M_inv).reshape(4, 2).tolist()
            
            x1, y1, x2, y2 = int(bx["x1"]), int(bx["y1"]), int(bx["x2"]), int(bx["y2"])
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(warped.shape[1], x2); y2 = min(warped.shape[0], y2)
            
            pred_str = "?"
            pred_conf = 1.0
            
            if x2 > x1 and y2 > y1 and parseq_model:
                crop = warped[y1:y2, x1:x2]
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                inp = parseq_transform(crop_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = parseq_model(inp)
                    pred = logits.softmax(-1)
                    label_pred, prob = parseq_model.tokenizer.decode(pred)
                    pred_str = label_pred[0]
                    if isinstance(prob, list) and len(prob) > 0 and len(prob[0]) > 0:
                        pred_conf = prob[0].mean().item()
                        
            digits.append({
                "index": i,
                "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
                "box_orig": c_orig,
                "char": pred_str,
                "conf": bx["conf"],
                "ocr_conf": pred_conf
            })
    else:
        # Mechanical uses TinyDigitCNN
        ocr_model, ocr_transform = get_ocr_model()
        from PIL import Image
        import torch
        for i, bx in enumerate(boxes):
            x1_f, y1_f, x2_f, y2_f = bx["x1"], bx["y1"], bx["x2"], bx["y2"]
            cw_pts = np.array([[x1_f, y1_f], [x2_f, y1_f], [x2_f, y2_f], [x1_f, y2_f]], dtype=np.float32).reshape(-1, 1, 2)
            c_orig = cv2.perspectiveTransform(cw_pts, M_inv).reshape(4, 2).tolist()
            
            if not ocr_model:
                digits.append({
                    "index": i,
                    "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
                    "box_orig": c_orig,
                    "char": "?"
                })
                continue
                
            x1, y1, x2, y2 = int(bx["x1"]), int(bx["y1"]), int(bx["x2"]), int(bx["y2"])
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(warped.shape[1], x2); y2 = min(warped.shape[0], y2)
            if x2 <= x1 or y2 <= y1: continue
            
            crop = warped[y1:y2, x1:x2]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            inp = ocr_transform(crop_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                preds = ocr_model(inp)
                probs = torch.nn.functional.softmax(preds, dim=1)
                pred_idx = probs.argmax(dim=1).item()
                pred_conf = probs[0, pred_idx].item()
                pred_str = str(pred_idx)
                
            digits.append({
                "index": i,
                "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
                "box_orig": c_orig,
                "char": pred_str,
                "conf": bx["conf"],
                "ocr_conf": pred_conf
            })

    return jsonify({
            "ok": True,
            "image_b64": b64,
            "w": max_w + 2*pad,
            "h": max_h + 2*pad,
            "corners": pts.tolist(),
            "matrix": M.tolist(),
            "class_id": cls_id,
            "digits": digits,
            "full_text": full_text
        })
    return jsonify({"error": "invalid label format"}), 400

@app.route("/api/yolo_detect_digits/<name>")
def api_yolo_detect_digits(name):
    """Run YOLO digit bbox detection on the rectified crop, ignoring saved labels."""
    if name not in _image_names():
        return jsonify({"error": "unknown image"}), 400

    stem = Path(name).stem
    lbl_path = CROPS_DIR / "labels" / f"{stem}.txt"
    if not lbl_path.is_file():
        return jsonify({"error": "no OBB label"}), 404

    parts = lbl_path.read_text().splitlines()[0].strip().split()
    if len(parts) < 9:
        return jsonify({"error": "invalid label format"}), 400

    import cv2, numpy as np
    cls_id = int(parts[0])
    nums = [float(p) for p in parts[1:9]]
    img_path = IMAGES_DIR / name
    img = cv2.imread(str(img_path))
    if img is None:
        return jsonify({"error": "could not read image"}), 500

    H, W = img.shape[:2]
    pts = np.array(nums).reshape(4, 2)
    pts[:, 0] *= W; pts[:, 1] *= H
    pts = pts.astype(np.float32)

    pad = 20
    w1 = int(np.linalg.norm(pts[0] - pts[1])); w2 = int(np.linalg.norm(pts[2] - pts[3]))
    h1 = int(np.linalg.norm(pts[1] - pts[2])); h2 = int(np.linalg.norm(pts[0] - pts[3]))
    max_w = max(w1, w2); max_h = max(h1, h2)

    dst = np.array([[pad, pad], [pad + max_w, pad], [pad + max_w, pad + max_h], [pad, pad + max_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (max_w + 2*pad, max_h + 2*pad))

    meter_class = "co"
    if METER_STATE_PATH.exists():
        import json
        with open(METER_STATE_PATH) as f:
            meter_state = json.load(f)
            meter_class = meter_state.get(name, "co")

    if meter_class == "dientu" or cls_id == 1:
        model = get_yolo_electronic()
        conf_thresh = 0.25
        iou_thresh = 0.45
    else:
        model = get_yolo_digits()
        conf_thresh = 0.35
        iou_thresh = 0.45

    if model is None:
        return jsonify({"error": "YOLO model not loaded. Train first."}), 503

    results = model(warped, conf=conf_thresh, iou=iou_thresh, verbose=False)
    import torch
    digits = []
    is_electronic = (meter_class == "dientu" or cls_id == 1)
    
    for r in results:
        for box in r.boxes:
            c = int(box.cls[0].item())
            
            # For mechanical meters (cls_id == 0), digit class is 0.
            # For electronic meters (cls_id == 1), the electronic model only has class 0.
            if not is_electronic and c != 0:
                continue
            if is_electronic and c != 0:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cw = warped.shape[1]; ch = warped.shape[0]
            digits.append({
                "cx": (x1 + x2) / 2 / cw,
                "cy": (y1 + y2) / 2 / ch,
                "w": (x2 - x1) / cw,
                "h": (y2 - y1) / ch,
                "label": "",
                "conf": round(float(box.conf[0].item()), 3)
            })
    digits.sort(key=lambda d: d["cx"])
    return jsonify({"ok": True, "digits": digits})


@app.route("/api/digit_labels", methods=["POST"])

def api_digit_labels():
    data = request.get_json(force=True)
    name = data.get("name")
    if name not in _image_names():
        return jsonify({"error": "unknown image"}), 400
        
    stem = Path(name).stem
    digits_dir = CROPS_DIR / "digits_data"
    digits_dir.mkdir(parents=True, exist_ok=True)
    
    cls_id = data.get("class_id", 0)
    
    out_data = {
        "image": name,
        "corners": data.get("corners", []),
        "matrix": data.get("matrix", []),
        "crop_pipeline_version": "v2",
        "class_id": cls_id,
        "digits": data.get("digits", []),
        "full_text": data.get("full_text", "")
    }
    
    (digits_dir / f"{stem}.json").write_text(json.dumps(out_data, ensure_ascii=False, indent=2))
    
    if cls_id == 1:
        text = out_data.get("full_text", "")
    else:
        text = "".join(("." if d.get("is_decimal") else "") + ("_" if d.get("label") == "unreadable" else d.get("label", "")) + ("↕" if d.get("is_between") else "") for d in out_data["digits"] if d.get("label"))
        
    ostate = _load_ocr_state()
    ostate[name] = text
    _save_ocr_state(ostate)
    
    return jsonify({"ok": True})


@app.route("/api/test_feedback", methods=["POST"])
def api_test_feedback():
    """Record whether the model's top detection was correct for an image.
    Body: {name, ok: bool, boxes: [...]}"""
    data = request.get_json(force=True)
    name = data.get("name")
    if name not in _test_images():
        return jsonify({"error": "not a valid test image"}), 400
    from datetime import datetime, timezone
    results = _load_test_results()
    results[name] = {
        "ok": bool(data.get("ok")),
        "boxes": data.get("boxes", []),
        "testedAt": datetime.now(timezone.utc).isoformat(),
    }
    _save_test_results(results)
    return jsonify({"ok": True, "total": len(results)})


@app.route("/api/test_stats")
def api_test_stats():
    """Aggregate stats over all evaluated test images."""
    results = _load_test_results()
    test_names = set(_test_images())
    state = _load_state()

    total = len(test_names)
    evaluated = {n: v for n, v in results.items() if n in test_names}
    n_ok = sum(1 for v in evaluated.values() if v.get("ok"))
    n_fail = sum(1 for v in evaluated.values() if not v.get("ok"))

    # Breakdown by reject type
    by_type: dict = {}
    for n, v in evaluated.items():
        label = state.get(n, {}).get("label", "unknown")
        if label not in by_type:
            by_type[label] = {"ok": 0, "fail": 0}
        if v.get("ok"):
            by_type[label]["ok"] += 1
        else:
            by_type[label]["fail"] += 1

    # Breakdown by predicted class
    by_class: dict = {}
    for v in evaluated.values():
        boxes = v.get("boxes", [])
        cls = boxes[0]["classKey"] if boxes else "no_detection"
        if cls not in by_class:
            by_class[cls] = {"ok": 0, "fail": 0}
        if v.get("ok"):
            by_class[cls]["ok"] += 1
        else:
            by_class[cls]["fail"] += 1

    return jsonify({
        "total": total,
        "evaluated": len(evaluated),
        "remaining": total - len(evaluated),
        "ok": n_ok,
        "fail": n_fail,
        "accuracy": round(n_ok / len(evaluated), 3) if evaluated else None,
        "byType": by_type,
        "byClass": by_class,
    })


# ---------------------------------------------------------------------------
# Label review screen
# ---------------------------------------------------------------------------

def _parse_label_file(label_path, W, H):
    """Parse a YOLO-OBB label file and return list of box dicts in pixel coords.
    Each box: {classId, classKey, classLabel, cx, cy, w, h, angleDeg, corners}.
    Returns [] if file is empty or unreadable."""
    try:
        lines = [l for l in label_path.read_text().splitlines() if l.strip()]
    except Exception:
        return []
    boxes = []
    for line in lines:
        toks = line.split()
        if len(toks) != 9:
            continue
        cls_id = int(toks[0])
        coords = [float(t) for t in toks[1:]]
        # Denormalize: [x1,y1, x2,y2, x3,y3, x4,y4] → pixel pairs
        corners = [(coords[i] * W, coords[i + 1] * H) for i in range(0, 8, 2)]
        cx, cy, bw, bh, ang = quad_to_box(corners)
        cls_key = next((k for k, v in METER_CLASSES.items() if v["id"] == cls_id), None)
        boxes.append({
            "classId": cls_id,
            "classKey": cls_key,
            "classLabel": METER_CLASSES.get(cls_key, {}).get("label", str(cls_id)),
            "cx": round(cx, 2), "cy": round(cy, 2),
            "w": round(bw, 2), "h": round(bh, 2),
            "angleDeg": round(ang, 2),
            "corners": [[round(x, 2), round(y, 2)] for x, y in corners],
        })
    return boxes


def _load_review_state():
    if REVIEW_STATE_PATH.exists():
        try:
            return json.loads(REVIEW_STATE_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_review_state(rstate):
    REVIEW_STATE_PATH.write_text(json.dumps(rstate, ensure_ascii=False, indent=1))


@app.route("/api/review_images")
def api_review_images():
    """Return all labeled images with their parsed OBB data and review status."""
    labels_dir = CROPS_DIR / "labels"
    if not labels_dir.is_dir():
        return jsonify({"images": [], "meterClasses": METER_CLASSES})

    rstate = _load_review_state()
    images = []
    for lp in sorted(labels_dir.glob("*.txt")):
        stem = lp.stem
        # Support jpg; try common extensions
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            p = IMAGES_DIR / (stem + ext)
            if p.is_file():
                img_path = p
                break
        if img_path is None:
            continue
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception:
            continue
        boxes = _parse_label_file(lp, W, H)
        if not boxes:
            continue
        images.append({
            "name": img_path.name,
            "stem": stem,
            "W": W, "H": H,
            "boxes": boxes,
            "review": rstate.get(img_path.name),   # None | "ok" | "flagged"
        })
    return jsonify({"images": images, "meterClasses": METER_CLASSES})


@app.route("/api/review_mark", methods=["POST"])
def api_review_mark():
    """Mark a label as ok or flagged.
    Body: {name: str, status: "ok" | "flagged"}"""
    data = request.get_json(force=True)
    name = data.get("name")
    status = data.get("status")
    if status not in ("ok", "flagged"):
        return jsonify({"error": "status must be ok or flagged"}), 400
    rstate = _load_review_state()
    rstate[name] = status
    _save_review_state(rstate)
    return jsonify({"ok": True})


@app.route("/api/review_delete", methods=["POST"])
def api_review_delete():
    """Delete the label file for an image (so it can be re-labelled).
    Body: {name: str}"""
    data = request.get_json(force=True)
    name = data.get("name")
    stem = Path(name).stem
    lp = CROPS_DIR / "labels" / (stem + ".txt")
    if lp.is_file():
        lp.unlink()
    # Also remove from review state
    rstate = _load_review_state()
    rstate.pop(name, None)
    _save_review_state(rstate)
    return jsonify({"ok": True, "deleted": lp.name})


@app.route("/api/review_stats")
def api_review_stats():
    """Aggregate review statistics."""
    labels_dir = CROPS_DIR / "labels"
    total = sum(1 for _ in labels_dir.glob("*.txt")) if labels_dir.is_dir() else 0
    rstate = _load_review_state()
    n_ok = sum(1 for v in rstate.values() if v == "ok")
    n_flagged = sum(1 for v in rstate.values() if v == "flagged")
    return jsonify({
        "total": total,
        "reviewed": len(rstate),
        "remaining": total - len(rstate),
        "ok": n_ok,
        "flagged": n_flagged,
    })



# -------------------------------------------------------------
# MULTI-BATCH VALIDATION & EDITING PIPELINE SUPPORT
# -------------------------------------------------------------
VALIDATE1_DIR = Path("/home/deist/Downloads/OCR/testocr")
VALIDATE2_DIR = Path("/home/deist/Downloads/OCR/Validate 2")
VALIDATE1_EVAL = BASE_DIR / "validate1_eval.json"
VALIDATE2_EVAL = BASE_DIR / "validate2_eval.json"

def get_validate_info(batch="2"):
    b = str(batch)
    if b == "1":
        return VALIDATE1_DIR, VALIDATE1_EVAL
    return VALIDATE2_DIR, VALIDATE2_EVAL

def find_validate_image_path(name, batch=None):
    if str(batch) == "1":
        p = VALIDATE1_DIR / name
        if p.exists(): return p, "1"
    elif str(batch) == "2":
        p = VALIDATE2_DIR / name
        if p.exists(): return p, "2"
    # Fallback search
    p2 = VALIDATE2_DIR / name
    if p2.exists(): return p2, "2"
    p1 = VALIDATE1_DIR / name
    if p1.exists(): return p1, "1"
    return None, None

@app.route("/validate1")
def view_validate1():
    """Màn hình chỉnh sửa riêng cho tập Validate 1 (48 ảnh) để chuẩn bị train"""
    return render_template("validate1_edit.html")

@app.route("/api/validate1/export_to_train", methods=["POST"])
def api_validate1_export_to_train():
    """Xuất các ảnh đã chỉnh sửa từ validate1_eval.json trực tiếp vào ocr_dataset"""
    import cv2, json, numpy as np
    eval_file = VALIDATE1_EVAL
    if not eval_file.exists():
        return jsonify({"error": "validate1_eval.json không tồn tại"}), 404
        
    with open(eval_file, encoding="utf-8") as f:
        eval_data = json.load(f)
        
    crops_target = BASE_DIR / "ocr_dataset" / "crops"
    labels_target = BASE_DIR / "ocr_dataset" / "labels"
    crops_target.mkdir(parents=True, exist_ok=True)
    labels_target.mkdir(parents=True, exist_ok=True)
    
    exported_images = 0
    clean_digits_count = 0
    half_digits_count = 0
    
    obb_model = _load_model()
    model_bbox = get_yolo_digits()
    
    for fname, d in eval_data.items():
        if d.get("type") != "mechanical" or not d.get("digits_bbox_correct"):
            continue
            
        img_path = VALIDATE1_DIR / fname
        if not img_path.exists(): continue
        
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        # OBB
        res_obb = obb_model(img, conf=0.25, verbose=False)
        if len(res_obb[0].obb) == 0: continue
        best_obb = max(res_obb[0].obb, key=lambda x: x.conf[0].item())
        quad = [tuple(p) for p in best_obb.xyxyxyxy[0].tolist()]
        pts_sum = [p[0] + p[1] for p in quad]
        pts_diff = [p[0] - p[1] for p in quad]
        tl = quad[np.argmin(pts_sum)]
        br = quad[np.argmax(pts_sum)]
        tr = quad[np.argmax(pts_diff)]
        bl = quad[np.argmin(pts_diff)]
        
        pad = 20
        max_w = max(int(np.linalg.norm(np.array(tl) - np.array(tr))), int(np.linalg.norm(np.array(bl) - np.array(br))))
        max_h = max(int(np.linalg.norm(np.array(tl) - np.array(bl))), int(np.linalg.norm(np.array(tr) - np.array(br))))
        pts = np.array([tl, tr, br, bl], dtype=np.float32)
        dst = np.array([[pad, pad], [pad + max_w, pad], [pad + max_w, pad + max_h], [pad, pad + max_h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(img, M, (max_w + 2*pad, max_h + 2*pad))
        
        W_warp, H_warp = warped.shape[1], warped.shape[0]
        
        boxes = []
        if "boxes" in d and d["boxes"]:
            boxes = d["boxes"]
        else:
            res_bbox = model_bbox(warped, conf=0.35, iou=0.45, verbose=False)
            for bx in res_bbox[0].boxes:
                boxes.append({
                    "x1": float(bx.xyxy[0][0]), "y1": float(bx.xyxy[0][1]),
                    "x2": float(bx.xyxy[0][2]), "y2": float(bx.xyxy[0][3]),
                })
            boxes.sort(key=lambda b: b["x1"])
            
        details = d.get("digit_details", [])
        stem = Path(fname).stem
        
        digit_entries = []
        for idx, bx in enumerate(boxes):
            if idx < len(details):
                det = details[idx]
                if det.get("is_spurious"):
                    continue
                lbl = str(det.get("actual", "")).strip()
                if not lbl or not lbl.isdigit():
                    continue
                is_half = bool(det.get("is_half", False))
                is_dec = bool(det.get("is_decimal", False))
                
                x_norm = bx["x1"] / W_warp
                y_norm = bx["y1"] / H_warp
                w_norm = (bx["x2"] - bx["x1"]) / W_warp
                h_norm = (bx["y2"] - bx["y1"]) / H_warp
                
                digit_entries.append({
                    "digit_index": len(digit_entries),
                    "x": x_norm, "y": y_norm, "w": w_norm, "h": h_norm,
                    "label": lbl,
                    "is_decimal": is_dec,
                    "mid_transition": is_half
                })
                if is_half:
                    half_digits_count += 1
                else:
                    clean_digits_count += 1
                    
        if digit_entries:
            cv2.imwrite(str(crops_target / f"{stem}.jpg"), warped)
            lbl_data = {
                "crop_id": stem,
                "source_image": fname,
                "digits": digit_entries
            }
            with open(labels_target / f"{stem}.json", "w", encoding="utf-8") as f:
                json.dump(lbl_data, f, ensure_ascii=False, indent=2)
            exported_images += 1
            
    return jsonify({
        "ok": True,
        "exported_images": exported_images,
        "clean_digits_count": clean_digits_count,
        "half_digits_count": half_digits_count
    })

@app.route("/testocr")
def view_testocr():
    return render_template("testocr.html")

@app.route("/api/testocr/image/<name>")
def api_testocr_serve(name):
    from flask import request
    batch = request.args.get("batch")
    p, _ = find_validate_image_path(name, batch)
    if p and p.exists():
        return send_file(str(p))
    return jsonify({"error": "File not found"}), 404


@app.route("/api/testocr/evaluate/<name>", methods=["POST"])
def api_testocr_evaluate(name):
    from flask import request
    data = request.json
    batch = request.args.get("batch")
    
    # Determine which file to save to
    if batch:
        _, eval_file = get_validate_info(batch)
    else:
        _, detected_b = find_validate_image_path(name)
        _, eval_file = get_validate_info(detected_b if detected_b else "2")
        
    records = {}
    if eval_file.exists():
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                records = json.load(f)
        except:
            pass
            
    records[name] = data
    
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        
    # Also sync with testocr_eval.json if batch 1
    if str(batch) == "1":
        try:
            with open(BASE_DIR / "testocr_eval.json", "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        except: pass

    return jsonify({"ok": True})

@app.route("/api/testocr/evaluate/<name>", methods=["GET"])
def api_testocr_evaluate_get(name):
    from flask import request
    batch = request.args.get("batch")
    _, eval_file = get_validate_info(batch if batch else "2")
    if not eval_file.exists():
        return jsonify(None)
    import json
    with open(eval_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    return jsonify(records.get(name, None))

@app.route("/api/testocr/all_evals")
def api_testocr_all_evals():
    from flask import request
    batch = request.args.get("batch", "2")
    _, eval_file = get_validate_info(batch)
    if not eval_file.exists():
        return jsonify({})
    import json
    with open(eval_file, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))

@app.route("/api/testocr/images")
def api_testocr_images():
    from flask import request
    batch = request.args.get("batch", "2")
    v_dir, _ = get_validate_info(batch)
    valid = []
    if v_dir.exists():
        for p in v_dir.glob("*.*"):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                valid.append(p.name)
    return jsonify(sorted(valid))

@app.route("/api/testocr/custom_obb/<name>", methods=["POST"])
def api_testocr_custom_obb(name):
    from flask import request
    data = request.json
    batch = request.args.get("batch") or data.get("batch")
    img_path, _ = find_validate_image_path(name, batch)
    if not img_path:
        img_path = TEST_IMAGES_DIR / name
    if not img_path.is_file():
        return jsonify({"error": "File not found"}), 404
        
    import cv2
    import numpy as np
    import base64
    from PIL import Image
    import torch
    
    img = cv2.imread(str(img_path))
    if img is None:
        return jsonify({"error": "Bad image"}), 400
        
    H, W = img.shape[:2]
    quad = data.get("quad", [])
    if len(quad) != 4:
        return jsonify({"error": "Invalid quad points (need 4 corners)"}), 400
        
    pts_sum = [p[0] + p[1] for p in quad]
    pts_diff = [p[0] - p[1] for p in quad]
    tl = quad[np.argmin(pts_sum)]
    br = quad[np.argmax(pts_sum)]
    tr = quad[np.argmax(pts_diff)]
    bl = quad[np.argmin(pts_diff)]
    ordered_quad = [tl, tr, br, bl]
    
    pts = np.array(ordered_quad, dtype=np.float32)
    pad = 20
    w1 = int(np.linalg.norm(pts[0] - pts[1]))
    w2 = int(np.linalg.norm(pts[2] - pts[3]))
    h1 = int(np.linalg.norm(pts[1] - pts[2]))
    h2 = int(np.linalg.norm(pts[0] - pts[3]))
    max_w = max(20, max(w1, w2))
    max_h = max(20, max(h1, h2))
    
    dst = np.array([
        [pad, pad],
        [pad + max_w, pad],
        [pad + max_w, pad + max_h],
        [pad, pad + max_h]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (max_w + 2*pad, max_h + 2*pad))
    
    cls_id = int(data.get("class_id", 0))
    model_bbox = get_yolo_electronic() if cls_id == 1 else get_yolo_digits()
    conf_thresh = 0.25
        
    res_bbox = model_bbox(warped, conf=conf_thresh, iou=0.45, verbose=False)
    boxes = []
    for r in res_bbox:
        for bx in r.boxes:
            boxes.append({
                "x1": float(bx.xyxy[0][0]),
                "y1": float(bx.xyxy[0][1]),
                "x2": float(bx.xyxy[0][2]),
                "y2": float(bx.xyxy[0][3]),
                "conf": float(bx.conf[0].item()),
                "cls": int(bx.cls[0].item())
            })
    boxes.sort(key=lambda b: b["x1"])
    
    digits = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cls_id == 0:
        ocr_model, ocr_transform = get_ocr_model()
        for i, bx in enumerate(boxes):
            x1, y1, x2, y2 = int(bx["x1"]), int(bx["y1"]), int(bx["x2"]), int(bx["y2"])
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(warped.shape[1], x2); y2 = min(warped.shape[0], y2)
            if x2 <= x1 or y2 <= y1: continue
            
            crop = warped[y1:y2, x1:x2]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            inp = ocr_transform(crop_pil).unsqueeze(0).to(device)
            pred_str = "?"
            pred_conf = 1.0
            with torch.no_grad():
                preds = ocr_model(inp)
                probs = torch.nn.functional.softmax(preds, dim=1)
                pred_idx = probs.argmax(dim=1).item()
                pred_conf = probs[0, pred_idx].item()
                pred_str = str(pred_idx)
            digits.append({
                "index": i,
                "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
                "char": pred_str,
                "conf": bx["conf"],
                "ocr_conf": pred_conf
            })
    else:
        for i, bx in enumerate(boxes):
            digits.append({
                "index": i,
                "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
                "char": "?",
                "conf": bx["conf"],
                "ocr_conf": 1.0
            })
            
    _, buf = cv2.imencode(".jpg", warped)
    warped_b64 = base64.b64encode(buf).decode("utf-8")
    
    return jsonify({
        "ok": True,
        "class_id": cls_id,
        "quad": ordered_quad,
        "digits": digits,
        "warped_w": max_w + 2*pad,
        "warped_h": max_h + 2*pad,
        "warped_b64": warped_b64,
        "image_b64": warped_b64,
        "M": M.tolist()
    })

@app.route("/api/testocr/infer/<name>")
def api_testocr_infer(name):
    from flask import request
    batch = request.args.get("batch")
    img_path, _ = find_validate_image_path(name, batch)
    if not img_path:
        img_path = TEST_IMAGES_DIR / name
    if not img_path.is_file():
        return jsonify({"error": "File not found"}), 404
        
    import cv2
    import numpy as np
    img = cv2.imread(str(img_path))
    if img is None:
        return jsonify({"error": "Bad image"}), 400
        
    H, W = img.shape[:2]
    
    # 1. OBB (Screen/Meter Detection)
    obb_model = _load_model()
    if not obb_model:
        return jsonify({"error": "OBB model not loaded"}), 503
        
    results = obb_model(img, verbose=False, conf=0.15)
    r = results[0]
    obb = getattr(r, "obb", None)
    if obb is None or len(obb) == 0:
        return jsonify({"error": "No OBB detected"}), 400
        
    best_idx = 0
    best_conf = 0
    for i in range(len(obb)):
        c = float(obb.conf[i].item())
        if c > best_conf:
            best_conf = c
            best_idx = i
            
    quad = [tuple(p) for p in obb.xyxyxyxy[best_idx].tolist()]
    cls_id = int(obb.cls[best_idx].item())
    
    pts_sum = [p[0] + p[1] for p in quad]
    pts_diff = [p[0] - p[1] for p in quad]
    tl = quad[np.argmin(pts_sum)]
    br = quad[np.argmax(pts_sum)]
    tr = quad[np.argmax(pts_diff)]
    bl = quad[np.argmin(pts_diff)]
    quad = [tl, tr, br, bl]
    
    pts = np.array(quad, dtype=np.float32)
    pad = 20
    w1 = int(np.linalg.norm(pts[0] - pts[1]))
    w2 = int(np.linalg.norm(pts[2] - pts[3]))
    h1 = int(np.linalg.norm(pts[1] - pts[2]))
    h2 = int(np.linalg.norm(pts[0] - pts[3]))
    max_w = max(w1, w2)
    max_h = max(h1, h2)
    
    dst = np.array([
        [pad, pad],
        [pad + max_w, pad],
        [pad + max_w, pad + max_h],
        [pad, pad + max_h]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (max_w + 2*pad, max_h + 2*pad))
    
    if cls_id == 1:
        model_bbox = get_yolo_electronic()
        conf_thresh = 0.25
        iou_thresh = 0.45
    else:
        model_bbox = get_yolo_digits()
        conf_thresh = 0.35
        iou_thresh = 0.45
        
    if model_bbox is None:
        return jsonify({"error": "YOLO bbox model not loaded"}), 503
        
    results_bbox = model_bbox(warped, conf=conf_thresh, iou=iou_thresh, verbose=False)
    boxes = []
    for bx in results_bbox[0].boxes:
        c = int(bx.cls[0].item())
        conf = float(bx.conf[0].item())
        x1, y1, x2, y2 = map(float, bx.xyxy[0].tolist())
        boxes.append({"c": c, "conf": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        
    boxes.sort(key=lambda b: b["x1"])
    
    filtered_boxes = []
    margin = pad
    for bx in boxes:
        cx = (bx["x1"] + bx["x2"]) / 2
        cy = (bx["y1"] + bx["y2"]) / 2
        if cx >= pad - margin and cx <= pad + max_w + margin and cy >= pad - margin and cy <= pad + max_h + margin:
            filtered_boxes.append(bx)
    boxes = filtered_boxes
    
    import torch
    digits = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    M_inv = np.linalg.inv(M)
    
    if cls_id == 1:
        # Electronic uses PARSeq
        parseq_model, parseq_transform = get_parseq_model()
        from PIL import Image
        import torch
        for i, bx in enumerate(boxes):
            x1_f, y1_f, x2_f, y2_f = bx["x1"], bx["y1"], bx["x2"], bx["y2"]
            cw_pts = np.array([[x1_f, y1_f], [x2_f, y1_f], [x2_f, y2_f], [x1_f, y2_f]], dtype=np.float32).reshape(-1, 1, 2)
            c_orig = cv2.perspectiveTransform(cw_pts, M_inv).reshape(4, 2).tolist()
            
            x1, y1, x2, y2 = int(bx["x1"]), int(bx["y1"]), int(bx["x2"]), int(bx["y2"])
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(warped.shape[1], x2); y2 = min(warped.shape[0], y2)
            
            pred_str = "?"
            pred_conf = 1.0
            
            if x2 > x1 and y2 > y1 and parseq_model:
                crop = warped[y1:y2, x1:x2]
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                inp = parseq_transform(crop_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = parseq_model(inp)
                    pred = logits.softmax(-1)
                    label_pred, prob = parseq_model.tokenizer.decode(pred)
                    pred_str = label_pred[0]
                    if isinstance(prob, list) and len(prob) > 0 and len(prob[0]) > 0:
                        pred_conf = prob[0].mean().item()
                        
            digits.append({
                "index": i,
                "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
                "box_orig": c_orig,
                "char": pred_str,
                "conf": bx["conf"],
                "ocr_conf": pred_conf
            })
    else:
        # Mechanical uses TinyDigitCNN
        ocr_model, ocr_transform = get_ocr_model()
        from PIL import Image
        import torch
        for i, bx in enumerate(boxes):
            x1_f, y1_f, x2_f, y2_f = bx["x1"], bx["y1"], bx["x2"], bx["y2"]
            cw_pts = np.array([[x1_f, y1_f], [x2_f, y1_f], [x2_f, y2_f], [x1_f, y2_f]], dtype=np.float32).reshape(-1, 1, 2)
            c_orig = cv2.perspectiveTransform(cw_pts, M_inv).reshape(4, 2).tolist()
            
            if not ocr_model:
                digits.append({
                    "index": i,
                    "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
                    "box_orig": c_orig,
                    "char": "?"
                })
                continue
                
            x1, y1, x2, y2 = int(bx["x1"]), int(bx["y1"]), int(bx["x2"]), int(bx["y2"])
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(warped.shape[1], x2); y2 = min(warped.shape[0], y2)
            if x2 <= x1 or y2 <= y1: continue
            
            crop = warped[y1:y2, x1:x2]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            inp = ocr_transform(crop_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = ocr_model(inp)
                if isinstance(out, tuple) and len(out) == 3:
                    preds, h_logit, dec_logit = out
                    p_half = torch.sigmoid(h_logit).item()
                    p_dec = torch.sigmoid(dec_logit).item()
                    model_is_half = (p_half >= 0.5)
                    model_is_dec = (p_dec >= 0.5)
                else:
                    preds = out
                    p_half = 0.0
                    p_dec = 0.0
                    model_is_half = False
                    model_is_dec = False

                probs = torch.nn.functional.softmax(preds, dim=1)[0]
                p_dict = {str(k): probs[k].item() for k in range(10)}
                res_mid = detect_mid_roll(p_dict)
                is_mid_roll_final = bool(res_mid.is_mid_roll or model_is_half)
                pred_str = res_mid.resolved_value if res_mid.is_mid_roll else res_mid.top1_label
                pred_conf = res_mid.top1_prob
                
            digits.append({
                "index": i,
                "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
                "box_orig": c_orig,
                "char": pred_str,
                "conf": bx["conf"],
                "ocr_conf": pred_conf,
                "is_mid_roll": is_mid_roll_final,
                "is_decimal": model_is_dec,
                "half_prob": round(p_half, 3),
                "dec_prob": round(p_dec, 3),
                "mid_roll_info": {
                    "top1": res_mid.top1_label,
                    "prob1": round(res_mid.top1_prob, 3),
                    "top2": res_mid.top2_label,
                    "prob2": round(res_mid.top2_prob, 3),
                    "margin": round(res_mid.margin, 3),
                    "reason": res_mid.reason
                } if res_mid.is_mid_roll else ({
                    "top1": res_mid.top1_label,
                    "prob1": round(res_mid.top1_prob, 3),
                    "top2": res_mid.top2_label,
                    "prob2": round(res_mid.top2_prob, 3),
                    "margin": round(res_mid.margin, 3),
                    "reason": f"Mô hình phát hiện Half-digit ({p_half:.1%})"
                } if model_is_half else None)
            })

    import base64
    _, buf = cv2.imencode(".jpg", warped)
    warped_b64 = base64.b64encode(buf).decode("utf-8")

    return jsonify({
        "ok": True,
        "class_id": cls_id,
        "quad": quad,
        "digits": digits,
        "warped_w": max_w + 2*pad,
        "warped_h": max_h + 2*pad,
        "warped_b64": warped_b64,
        "image_b64": warped_b64,
        "M": M.tolist()
    })

if __name__ == "__main__":
    print("Meter-reading data-prep app")
    print(f"  images: {len(_image_names())}")
    migrated = _migrate_axis_labels()
    if migrated:
        print(f"  migrated {migrated} legacy label(s) to OBB format")
    
    # Eagerly load models to prevent multithreading issues
    try:
        _load_model()
        get_yolo_digits()
        get_yolo_electronic()
        get_ocr_model()
        get_parseq_model()
    except Exception as e:
        print("Error eagerly loading models:", e)

    print("  open http://127.0.0.1:5000")
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
