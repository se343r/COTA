"""Prototype v3: digit-row detector.

Reads the reading display off meter photos. Strategy:
  * adaptive threshold (dark glyphs on light panel, or light on dark)
  * connected components -> glyph blobs
  * exclude watermark zone (bottom ~9%) and extreme top
  * group blobs into horizontal bands by y-overlap
  * score bands by (glyph count) x (spacing regularity): the LCD digit row
    has many roughly-equal-spaced glyphs; brand/watermark text is irregular
  * pick best band across both polarities, return bbox with margin
"""
import sys, os, glob
import numpy as np
import cv2


def _glyph_band(img, invert, y_lo=0.01, y_hi=0.91):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    bs = 2 * (int(h * 0.08) | 1) + 1
    thr = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV if invert
                                else cv2.THRESH_BINARY, bs, 8)
    k_open = max(1, int(min(h, w) * 0.003) | 1)
    content = cv2.morphologyEx(thr, cv2.MORPH_OPEN,
                               np.ones((k_open, k_open), np.uint8))

    n, _, stats, _ = cv2.connectedComponentsWithStats(content, 8)
    if n <= 1:
        return None, 0.0
    tops = stats[1:, cv2.CC_STAT_TOP].astype(float)
    lefts = stats[1:, cv2.CC_STAT_LEFT].astype(float)
    hs = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    ws = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    img_area = h * w

    med_h = np.median(hs)
    keep = (areas > img_area * 0.00001) & (areas < img_area * 0.02) & \
           (hs >= 0.3 * med_h) & (hs <= 2.5 * med_h) & (ws < w * 0.5) & \
           (tops > y_lo * h) & (tops < y_hi * h)
    if keep.sum() < 3:
        return None, 0.0

    tops, lefts, hs, ws = tops[keep], lefts[keep], hs[keep], ws[keep]
    yc = tops + hs / 2.0
    win = max(np.median(hs) * 1.4, h * 0.01)

    order = np.argsort(yc)
    yc_s = yc[order]
    tops_s, lefts_s, hs_s, ws_s = tops[order], lefts[order], hs[order], ws[order]

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
        # spacing regularity: coefficient of variation of consecutive glyph centers
        cx = lefts_s[inb] + ws_s[inb] / 2.0
        gaps = np.diff(np.sort(cx))
        gaps = gaps[gaps > 0]
        if len(gaps) < 2:
            regularity = 0.2
        else:
            mean_gap = gaps.mean()
            regularity = 1.0 / (1.0 + gaps.std() / (mean_gap + 1e-6))
        # vertical band tightness
        tightness = 1.0 / (1.0 + bh / (np.median(hs_s[inb]) + 1e-6))
        # horizontal span
        spread = min(1.0, bw / (0.9 * w))
        score = cnt * (0.7 + regularity) * (0.6 + tightness) * (0.5 + spread)
        if best is None or score > best[0]:
            best = (score, (bx0, by0, bx1, by1), cnt)
    return (best[1], best[2], best[0]) if best else (None, 0, 0.0)


def detect_display(img, margin=0.06):
    h, w = img.shape[:2]
    cands = []
    for invert in (True, False):
        box, cnt, score = _glyph_band(img, invert)
        if box:
            x0, y0, x1, y1 = box
            if (x1 - x0) > 0.12 * w:
                cands.append((score, box, cnt))
    if not cands:
        return None
    cands.sort(reverse=True)
    _, (x0, y0, x1, y1), cnt = cands[0]
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    mx, my = int(w * margin), int(h * margin)
    x0, x1 = max(0, x0 - mx), min(w, x1 + mx)
    y0, y1 = max(0, y0 - my), min(h, y1 + my)
    return (x0, y0, x1, y1), cnt


def main():
    files = sys.argv[1:] or sorted(glob.glob('*.jpg'))
    if len(files) > 15:
        files = files[:15]
    for f in files:
        img = cv2.imread(f)
        res = detect_display(img)
        if res is None:
            print(f'{f:42s} NO DISPLAY')
            continue
        (x0, y0, x1, y1), cnt = res
        crop = img[y0:y1, x0:x1]
        cv2.imwrite('/tmp/_crop.jpg', crop)
        ocr = os.popen(f'tesseract /tmp/_crop.jpg - --psm 7 2>/dev/null').read().strip()
        print(f'{f:42s} box=({x0},{y0},{x1},{y1}) glyphs={cnt:3d} ocr="{ocr[:40]}"')


if __name__ == '__main__':
    main()
