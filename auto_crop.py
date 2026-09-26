"""Bootstrap auto-crop detector for meter-reading photos.

This is the *heuristic* first pass. It is deliberately pragmatic: it reliably
finds the reading display on clear photos and gives a starting box everywhere
else. The crop app lets the user adjust, and the exported labels can be used to
fine-tune a YOLO model (see train_yolo.py) that replaces this heuristic.

Strategies tried in order of confidence:
  A. Bright horizontal panel  - the backlit LCD housing is the largest
       horizontally-elongated bright region in the frame. Uses minAreaRect
       to capture the panel even when it is tilted.
  B. Digit-glyph band         - adaptive threshold reveals the faint digits as
       a row of many small, regularly-spaced glyphs; brand/watermark text rows
       score lower. Handles dim meters and light-on-dark displays.
  C. Center band fallback     - a mid-frame horizontal band.

Tilted-image support
--------------------
A global skew angle is estimated from the image before running the above
strategies.  When the tilt is significant (≥ 1°) the image is virtually
rotated so that strategies A/B/C see a near-upright frame, and the resulting
box is then un-rotated back into the original image coordinate system.  This
makes the detector robust to meter photos taken at up to ±45° from vertical.

Each strategy returns a box + a confidence score; the best is returned.
"""
import math

import numpy as np
import cv2


def _clahe(gray, clip=2.5):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(gray)


# --------------------------------------------------------------------------
# Global skew detection
# --------------------------------------------------------------------------
def _global_skew_angle(gray, max_angle=45.0):
    """Estimate the dominant tilt of the whole image (degrees, screen coords).

    Uses Probabilistic Hough Lines on an edge map.  Returns 0.0 when the
    tilt cannot be estimated reliably or is below 1°.  Positive angles mean
    the image leans clockwise (top edge tilted right).
    """
    h, w = gray.shape
    eq = _clahe(gray, 2.0)
    edges = cv2.Canny(eq, 50, 150)
    # Dilate slightly so broken edges connect
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))

    min_len = int(min(h, w) * 0.15)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180,
                            threshold=60, minLineLength=min_len, maxLineGap=12)
    if lines is None or len(lines) == 0:
        return 0.0

    # HoughLinesP returns (N,1,4) in OpenCV 3.x and (N,4) in some builds
    if lines.ndim == 3:
        lines = lines[:, 0, :]

    angles = []
    weights = []
    for x1, y1, x2, y2 in lines:
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < min_len:
            continue
        ang = math.degrees(math.atan2(dy, dx))
        # Fold into [-90, 90] to compare all near-horizontal lines
        if ang > 90:
            ang -= 180
        elif ang < -90:
            ang += 180
        # Keep only lines that look roughly horizontal (dominant text direction)
        if abs(ang) <= max_angle:
            angles.append(ang)
            weights.append(length)

    if not angles:
        return 0.0

    angles = np.array(angles)
    weights = np.array(weights)

    # Weighted median — robust against noise lines
    order = np.argsort(angles)
    angles_s, weights_s = angles[order], weights[order]
    cum = np.cumsum(weights_s)
    med_angle = float(angles_s[np.searchsorted(cum, cum[-1] / 2.0)])

    if abs(med_angle) < 1.0:
        return 0.0
    return med_angle


def _rotate_image(img, angle_deg):
    """Rotate `img` clockwise by `angle_deg` degrees about its centre.

    Returns (rotated_img, rotation_matrix M) where M is the 2×3 affine
    matrix so that original point p maps to M @ [x, y, 1]^T.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    # Compute new bounding dimensions so no corner is clipped
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += new_w / 2.0 - cx
    M[1, 2] += new_h / 2.0 - cy
    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, M


def _unrotate_box(box_rotated, M):
    """Map an axis-aligned box from a rotated image back to original coords.

    `box_rotated` is (x0, y0, x1, y1) in the rotated image.
    Returns the 4 corners [(x,y), …] in original image coords (not axis-aligned).
    """
    x0, y0, x1, y1 = box_rotated
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    # Invert the affine mapping: orig = M_inv @ [rx, ry, 1]
    M_inv = cv2.invertAffineTransform(M)
    ones = np.ones((4, 1), dtype=np.float64)
    orig = (M_inv @ np.hstack([corners, ones]).T).T  # shape (4, 2)
    return [tuple(p) for p in orig]


# --------------------------------------------------------------------------
# Strategy A: bright horizontal panel (OBB-aware via minAreaRect)
# --------------------------------------------------------------------------
def _bright_panel(gray):
    """Detect the largest bright panel; return its OBB angle as well."""
    h, w = gray.shape
    eq = _clahe(gray, 2.0)
    thr = int(np.percentile(eq, 90))
    mask = (eq >= thr).astype(np.uint8)
    k = max(3, int(min(h, w) * 0.02) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, labels_img, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        x, y, ww, hh, aa = stats[i]
        if aa < 0.002 * h * w or ww < 0.1 * w or hh < 0.015 * h:
            continue
        ar = ww / max(1, hh)
        if not (1.5 <= ar <= 8):
            continue
        if best is None or aa > best[0]:
            best = (aa, i)
    if best is None:
        return None, 0.0, 0.0

    comp_mask = (labels_img == best[1]).astype(np.uint8)
    pts = np.column_stack(np.where(comp_mask > 0))[:, ::-1].astype(np.float32)
    rect = cv2.minAreaRect(pts)   # ((cx,cy), (rw,rh), angle_cv)
    (rcx, rcy), (rw, rh), angle_cv = rect
    # cv2.minAreaRect angle: in [-90,0); convert to our convention (clockwise+)
    if rw < rh:
        panel_angle = angle_cv + 90.0
    else:
        panel_angle = angle_cv
    # Axis-aligned bounding box for compatibility with _digit_row_in
    x, y, ww, hh, aa = stats[best[1]]
    confidence = aa / (h * w)
    return (x, y, x + ww, y + hh), confidence, panel_angle


# --------------------------------------------------------------------------
# Strategy B: digit-glyph band
# --------------------------------------------------------------------------
def _glyph_band(gray, invert, y_lo=0.02, y_hi=0.88):
    h, w = gray.shape
    eq = _clahe(gray, 3.0)
    bs = 2 * (max(25, int(min(h, w) * 0.1)) | 1) + 1
    thr = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV if invert
                                else cv2.THRESH_BINARY, bs, 8)
    content = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(content, 8)
    if n <= 1:
        return None, 0.0, 0
    tops = stats[1:, cv2.CC_STAT_TOP].astype(float)
    lefts = stats[1:, cv2.CC_STAT_LEFT].astype(float)
    hs = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    ws = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    img_area = h * w

    big = areas > max(30.0, img_area * 0.00002)
    if big.sum() < 3:
        return None, 0.0, 0
    scale = np.median(hs[big])
    keep = big & (hs >= 0.4 * scale) & (hs <= 3.0 * scale) & \
           (ws < w * 0.4) & (areas < img_area * 0.02) & \
           (tops > y_lo * h) & (tops < y_hi * h)
    if keep.sum() < 3:
        return None, 0.0, 0

    tops, lefts, hs, ws = tops[keep], lefts[keep], hs[keep], ws[keep]
    yc = tops + hs / 2.0
    win = 2.5 * scale
    order = np.argsort(yc)
    yc_s, tops_s, lefts_s, hs_s, ws_s = (yc[order], tops[order], lefts[order],
                                         hs[order], ws[order])

    best = None
    for i in range(len(yc_s)):
        inb = (yc_s >= yc_s[i] - win / 2) & (yc_s <= yc_s[i] + win / 2)
        cnt = int(inb.sum())
        if cnt < 3:
            continue
        bx0 = lefts_s[inb].min()
        bx1 = (lefts_s[inb] + ws_s[inb]).max()
        by0 = tops_s[inb].min()
        by1 = (tops_s[inb] + hs_s[inb]).max()
        bw, bh = bx1 - bx0, by1 - by0
        if bw <= 0 or bh <= 0:
            continue
        cx = lefts_s[inb] + ws_s[inb] / 2.0
        gaps = np.diff(np.sort(cx))
        gaps = gaps[gaps > 0]
        regularity = 1.0 / (1.0 + gaps.std() / (gaps.mean() + 1e-6)) \
            if len(gaps) >= 2 else 0.2
        tightness = 1.0 / (1.0 + bh / (scale + 1e-6))
        spread = min(1.0, bw / (0.9 * w))
        score = cnt * (0.7 + regularity) * (0.6 + tightness) * (0.5 + spread)
        if best is None or score > best[0]:
            best = (score, (bx0, by0, bx1, by1), cnt)
    if best is None:
        return None, 0.0, 0
    return best[1], best[0], best[2]


def _digit_band(gray):
    h, w = gray.shape
    best = None
    for invert in (True, False):
        box, score, cnt = _glyph_band(gray, invert)
        if box:
            x0, y0, x1, y1 = box
            if (x1 - x0) > 0.12 * w:
                if best is None or score > best[0]:
                    best = (score, box)
    return (best[1], best[0]) if best else (None, 0.0)


# --------------------------------------------------------------------------
# Strategy C: center-band fallback
# --------------------------------------------------------------------------
def _center_band(gray):
    h, w = gray.shape
    return (int(0.15 * w), int(0.25 * h), int(0.85 * w), int(0.55 * h)), 0.1


def _digit_row_in(gray, box):
    """Tighten a panel box to its densest internal digit band."""
    x0, y0, x1, y1 = box
    sub = gray[y0:y1, x0:x1]
    sh, sw = sub.shape
    if sh < 8 or sw < 16:
        return box
    eq = _clahe(sub, 3.0)
    bs = 2 * (max(21, int(sh * 0.5)) | 1) + 1
    thr = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, bs, 8)
    rows = thr.sum(axis=1) / 255.0
    ks = max(3, int(sh * 0.12) | 1)
    kernel = np.ones(ks, dtype=np.float32) / ks
    smooth = np.convolve(rows, kernel, mode='same')
    peak = int(np.argmax(smooth))
    if smooth[peak] < 1.0:
        return box
    # keep rows >= 45% of peak
    above = smooth >= 0.45 * smooth[peak]
    ys = np.where(above)[0]
    if len(ys) == 0:
        return box
    yy0, yy1 = ys.min(), ys.max()
    # vertical margin inside the panel
    pad = max(2, int(sh * 0.08))
    ny0 = max(0, yy0 - pad)
    ny1 = min(sh, yy1 + pad)
    return (x0, y0 + ny0, x1, y0 + ny1)


def detect_display(img, margin=0.05):
    """Return ((x0, y0, x1, y1), strategy) or (None, strategy).

    Box is in original image pixel coordinates. The bright panel is preferred
    (it captures the light display background); glyph band and center band are
    fallbacks.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    box, _, _ = _bright_panel(gray)
    if box:
        refined = _digit_row_in(gray, box)
        return _pad(refined, h, w, margin), 'panel'

    box, _ = _digit_band(gray)
    if box:
        return _pad(box, h, w, margin), 'digits'

    box, _ = _center_band(gray)
    return _pad(box, h, w, margin), 'fallback'


def _pad(box, h, w, margin):
    x0, y0, x1, y1 = [int(v) for v in box]
    mx, my = int(w * margin), int(h * margin)
    return (max(0, x0 - mx), max(0, y0 - my),
            min(w, x1 + mx), min(h, y1 + my))


# --------------------------------------------------------------------------
# YOLO label helpers
# --------------------------------------------------------------------------
def box_to_yolo(box, w, h):
    """Convert (x0,y0,x1,y1) to YOLO normalized (cx, cy, bw, bh)."""
    x0, y0, x1, y1 = [float(v) for v in box]
    cx = (x0 + x1) / 2.0 / w
    cy = (y0 + y1) / 2.0 / h
    bw = (x1 - x0) / w
    bh = (y1 - y0) / h
    return cx, cy, bw, bh


# --------------------------------------------------------------------------
# Oriented-box (OBB) geometry.
#
# A box is a rectangle: centre (cx, cy), size (w, h), rotation `angle_deg`.
# The angle is in degrees, measured in screen coordinates (y grows downward);
# the box's width axis points along (cos a, sin a), so a positive angle tilts
# the top edge down to the right (visually clockwise). angle 0 == axis-aligned.
# `box_to_quad` returns the 4 corners in clockwise order TL,TR,BR,BL, which is
# the order written to YOLO-OBB label files (the trainer is order-agnostic and
# canonicalizes via its min-area rectangle, but clockwise-from-TL matches docs).
# --------------------------------------------------------------------------
def box_to_quad(cx, cy, w, h, angle_deg):
    """Corners (image px) of a rectangle: TL, TR, BR, BL (clockwise)."""
    a = math.radians(angle_deg)
    cos, sin = math.cos(a), math.sin(a)
    ux, uy, vx, vy = cos, sin, -sin, cos   # u = width axis, v = u + 90 deg
    hw, hh = w / 2.0, h / 2.0
    return [
        (cx - hw * ux - hh * vx, cy - hw * uy - hh * vy),   # top-left
        (cx + hw * ux - hh * vx, cy + hw * uy - hh * vy),   # top-right
        (cx + hw * ux + hh * vx, cy + hw * uy + hh * vy),   # bottom-right
        (cx - hw * ux + hh * vx, cy - hw * uy + hh * vy),   # bottom-left
    ]


def quad_to_box(quad):
    """Fit (cx, cy, w, h, angle_deg) to the corners of a rectangle.

    `quad` must list 4 corners where consecutive points are adjacent (i.e. the
    polygon loops the rectangle); which corner it starts at is irrelevant. Width
    is taken as the longer side so an ultralytics prediction (whose canonical
    corners start at a different corner) round-trips cleanly.
    """
    xs = sum(p[0] for p in quad) / 4.0
    ys = sum(p[1] for p in quad) / 4.0
    ax, ay = quad[1][0] - quad[0][0], quad[1][1] - quad[0][1]
    bx, by = quad[2][0] - quad[1][0], quad[2][1] - quad[1][1]
    la, lb = math.hypot(ax, ay), math.hypot(bx, by)
    if la >= lb and la > 0:
        ux, uy, w, h = ax / la, ay / la, la, lb
    elif lb > 0:
        ux, uy, w, h = bx / lb, by / lb, lb, la
    else:
        return xs, ys, 0.0, 0.0, 0.0
    a = math.degrees(math.atan2(uy, ux))
    if a > 90.0:
        a -= 180.0
    elif a < -90.0:
        a += 180.0
    return xs, ys, w, h, a


def corners_to_label(corners, W, H):
    """8 normalized coords (x1n,y1n,...,x4n,y4n) for a YOLO-OBB label line."""
    out = []
    for (x, y) in corners:
        out.append(max(0.0, min(1.0, x / W)))
        out.append(max(0.0, min(1.0, y / H)))
    return out


# --------------------------------------------------------------------------
# Digit-row angle estimation (local, inside the detected box)
# --------------------------------------------------------------------------
def _hough_row_angle(gray, box, max_angle=45.0):
    """Estimate digit-row tilt via Hough Lines inside `box`.

    More robust than linear regression when glyphs are fragmented or sparse.
    Returns angle (degrees, screen convention) or 0.0 if unreliable.
    """
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    if x1 - x0 < 30 or y1 - y0 < 10:
        return 0.0
    sub = gray[y0:y1, x0:x1]
    sh, sw = sub.shape
    eq = _clahe(sub, 3.0)
    edges = cv2.Canny(eq, 40, 120)
    min_len = max(10, int(sw * 0.15))
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180,
                            threshold=20, minLineLength=min_len, maxLineGap=6)
    if lines is None or len(lines) == 0:
        return 0.0

    # HoughLinesP returns (N,1,4) in OpenCV 3.x and (N,4) in some builds
    if lines.ndim == 3:
        lines = lines[:, 0, :]

    angles, weights = [], []
    for x1l, y1l, x2l, y2l in lines:
        dx, dy = x2l - x1l, y2l - y1l
        length = math.hypot(dx, dy)
        if length < min_len:
            continue
        ang = math.degrees(math.atan2(dy, dx))
        if ang > 90:
            ang -= 180
        elif ang < -90:
            ang += 180
        if abs(ang) <= max_angle:
            angles.append(ang)
            weights.append(length)

    if not angles:
        return 0.0

    angles = np.array(angles)
    weights = np.array(weights)
    order = np.argsort(angles)
    cum = np.cumsum(weights[order])
    med = float(angles[order][np.searchsorted(cum, cum[-1] / 2.0)])
    if abs(med) < 0.5:
        return 0.0
    return med


def _digit_row_angle(gray, box):
    """Tilt (deg, screen convention) of the digit row inside `box`, else 0.0.

    Now uses a two-pass approach:
      1. Linear regression on glyph centres (precise when glyphs are clear)
      2. Hough Lines fallback (robust when regression fails)

    Angles in [0.5°, 45°] are reported; smaller angles are treated as zero.
    """
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    if x1 - x0 < 40 or y1 - y0 < 12:
        return 0.0
    sub = gray[y0:y1, x0:x1]
    sh, sw = sub.shape
    eq = _clahe(sub, 3.0)
    bs = 2 * (max(21, int(sh * 0.5)) | 1) + 1
    candidates = []
    for invert in (True, False):  # dark-on-light and light-on-dark displays
        thr = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV if invert
                                    else cv2.THRESH_BINARY, bs, 8)
        content = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(content, 8)
        if n <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
        hs = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
        ws = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
        tops = stats[1:, cv2.CC_STAT_TOP].astype(float)
        lefts = stats[1:, cv2.CC_STAT_LEFT].astype(float)
        big = (areas > 15) & (ws < sw * 0.4) & (hs < sh * 0.9) & \
              (areas < sh * sw * 0.2)
        if big.sum() < 4:
            continue
        scale = float(np.median(hs[big]))
        if scale <= 0:
            continue
        keep = big & (hs >= 0.45 * scale) & (hs <= 2.5 * scale)
        nk = int(keep.sum())
        if nk < 4:
            continue
        xs = lefts[keep] + ws[keep] / 2.0 + x0
        ys = tops[keep] + hs[keep] / 2.0 + y0
        candidates.append((nk, xs, ys, scale))
    if not candidates:
        # Fallback: Hough Lines
        return _hough_row_angle(gray, box)

    _, xs, ys, scale = max(candidates, key=lambda c: c[0])

    # Relaxed span requirement: 30% (was 45%)
    if xs.max() - xs.min() < 0.30 * (x1 - x0):
        return _hough_row_angle(gray, box)

    # Two-pass linear fit, dropping glyphs far off the fitted row
    slope, icpt = np.polyfit(xs, ys, 1)
    for _ in range(2):
        resid = ys - (slope * xs + icpt)
        good = np.abs(resid) <= 0.9 * scale
        if good.sum() < 4:
            return _hough_row_angle(gray, box)
        slope, icpt = np.polyfit(xs[good], ys[good], 1)
    resid = ys - (slope * xs + icpt)
    if np.sqrt(np.mean(resid ** 2)) > 0.45 * scale:
        return _hough_row_angle(gray, box)

    ang = math.degrees(math.atan(slope))
    # Relaxed angle range: [0.5°, 45°] (was [2°, 35°])
    if abs(ang) < 0.5 or abs(ang) > 45.0:
        return 0.0
    return ang


# --------------------------------------------------------------------------
# OBB-aware detection  (main public API)
# --------------------------------------------------------------------------
def detect_display_obb(img):
    """Auto-suggest box with tilt, like detect_display but OBB-aware.

    Pipeline:
      1. Estimate global image skew via Hough Lines.
      2a. If skew >= 1°: virtually rotate image → run detect_display →
          un-rotate the resulting box corners back to original coordinates →
          fit an OBB.  The panel tilt from Strategy A (minAreaRect) is also
          considered and the more confident estimate wins.
      2b. If skew < 1°: run detect_display on the original image, then
          estimate local digit-row angle from the box region (regression +
          Hough Lines fallback).

    Returns (dict(cx,cy,w,h,angleDeg) in image px, strategy) or
            (None, strategy) when no box was found.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Step 1: global skew ───────────────────────────────────────────────
    global_skew = _global_skew_angle(gray)

    # ── Step 2a: deskew-then-detect ───────────────────────────────────────
    if abs(global_skew) >= 1.0:
        rot_img, M = _rotate_image(img, global_skew)
        box_rot, strategy = detect_display(rot_img)
        if box_rot is not None:
            # Un-rotate the 4 corners back to original image coordinates
            orig_corners = _unrotate_box(box_rot, M)
            cx, cy, bw, bh, fit_angle = quad_to_box(orig_corners)
            # quad_to_box returns an OBB angle; the global skew is already
            # baked into the corner positions, so we just keep it.
            # Clamp centre to image bounds
            cx = max(0.0, min(w, cx))
            cy = max(0.0, min(h, cy))
            return {"cx": cx, "cy": cy, "w": bw, "h": bh,
                    "angleDeg": fit_angle}, strategy

    # ── Step 2b: standard detect + local tilt ────────────────────────────
    box, strategy = detect_display(img)
    if box is None:
        return None, strategy

    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bw, bh = x1 - x0, y1 - y0

    angle = 0.0
    if strategy in ("panel", "digits"):
        # Try the panel's own minAreaRect angle first (Strategy A)
        if strategy == "panel":
            _, _, panel_angle = _bright_panel(gray)
            if abs(panel_angle) >= 0.5:
                angle = panel_angle

        # Refine / override with digit-row angle (more accurate)
        if strategy in ("panel", "digits"):
            row_angle = _digit_row_angle(gray, box)
            if abs(row_angle) >= 0.5:
                angle = row_angle

    return {"cx": cx, "cy": cy, "w": bw, "h": bh, "angleDeg": angle}, strategy
