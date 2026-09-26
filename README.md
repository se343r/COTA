# Meter-reading data prep

Two local tools for building a YOLO training set from the meter photos in this
folder (1089 JPGs, 960×1280):

1. **Filter samples** — review every photo and mark it keep/reject (with a
   reason). Export a curated dataset + manifest for AI training.
2. **Label boxes** — hand-draw the (rotatable) reading-display box on each
   photo, save YOLO-**OBB** labels, and export a dataset to train **YOLO-OBB**.

```
meter photos (*.jpg) ──► ① Filter  ──►  filtered/keep  (curated set)
                    └──► ② Label  ──►  crops/labels/*.txt  (YOLO-OBB labels)
                                        └─► yolo_dataset/  (images+labels+data.yaml)
                                            └─► python train_yolo.py ─► best.pt ─► app uses it
```

## Quick start

```bash
# one-time setup
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# run the app
.venv/bin/python app.py
# open http://127.0.0.1:5000
```

## Tab 1 — Filter samples

- **Next/Prev** or `←`/`→` keys to browse.
- One click per photo: **Keep** (`1`), or a rejection reason (`2`–`3`, or the
  buttons: Blurry / Too dark / Meter cut off / Not a meter / Duplicate / Other).
- Progress bar + live stats; green/warn/red border shows status on the thumbnails.
- **Export filtered dataset** copies the accepted set into `filtered/keep` and
  rejects into `filtered/reject`, with `manifest.json` + `manifest.csv`.
  Progress is saved automatically to `filter_progress.json` — safe to stop and
  resume later.

Keyboard: `1` keep, `2` blurry, `3` partial/cut-off, `←`/`→` navigate.

## Tab 2 — Label boxes (for training YOLO-OBB)

- Pick an image from the left grid (green border = already labeled).
- **Drag on the photo to draw the reading-display box.** Drag inside it to
  move, drag a corner handle to resize (the box keeps its tilt).
- Photos taken at an angle? **Rotate the box**: drag the round **● handle**
  above the top edge, or press `[` / `]` (±1°, `Shift` = ±5°). `R` resets the
  tilt to 0. The box always stays a proper rectangle.
- A drawn-but-unsaved box is **orange**; a saved box is **green**.
- Boxes **auto-save** when you navigate away — or press **`S`** to save & go next.
- **`D`** suggests a starting box from the heuristic detector — it also returns
  the display's tilt when it is confident, so slightly rotated photos come in
  pre-rotated (adjust it, then save).
- **`X`** clears the box and deletes the saved label for that image.
- **`N`** jumps to the next unlabeled image.

Keyboard: `S` save+next, `X` clear/delete, `D` auto-suggest, `N` next unlabeled,
`[`/`]` rotate, `Shift`+`[`/`]` rotate 5°, `R` reset tilt, `←`/`→` navigate.

Labels are YOLO-**OBB** format (`class x1 y1 x2 y2 x3 y3 x4 y4`, class `0` =
reading, the 8 corner coordinates normalized to the **full** image, clockwise
from the top-left), in `crops/labels/<name>.txt`. Labels saved before OBB
support are upgraded to this format automatically on app start / export.

## Exporting + training YOLO-OBB

Label a few hundred diverse photos (clear + dim + small meters), then:

1. Press **📦 Export YOLO dataset** → builds `yolo_dataset/` with a train/val
   split and `data.yaml`.
2. Train:
   ```bash
   .venv/bin/pip install ultralytics torch   # if not already installed
   .venv/bin/python train_yolo.py --epochs 120
   ```
   The default model is `yolov8n-obb.pt` (or try `--model yolo11n-obb.pt`);
   best weights → `runs/obb/train/weights/best.pt`.
3. Restart the app: **`D`** then uses the trained OBB model instead of the
   heuristic, so new (rotated) photos are suggested pre-tilted and you label
   faster.

How many to label? ~200–400 gives a usable first model; 500–800 is better. The
heuristic auto-suggest (`D`) makes this much faster.

## Files

| file | purpose |
|---|---|
| `app.py` | Flask server: filter tool + manual box-labeling tool, YOLO dataset export |
| `templates/index.html` | single-page UI (no dependencies, works offline) |
| `auto_crop.py` | heuristic display detector (the `D` auto-suggest) + YOLO label helpers |
| `train_yolo.py` | YOLOv8 training script |
| `filtered/` | exported curated dataset + manifest |
| `crops/labels/` | YOLO-OBB label txt files (class 0 = reading, 8 corners) |
| `yolo_dataset/` | train/val images + labels + data.yaml |
| `filter_progress.json` | filter decisions (auto-saved) |
