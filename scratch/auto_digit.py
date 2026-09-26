import cv2
import numpy as np
import base64

def find_digit_boxes(img_bgr):
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold to isolate digits
    # First apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Use Canny edges or adaptive threshold
    # Canny is often better for numbers on dials since backgrounds vary
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate slightly to connect digit strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    H, W = img_bgr.shape[:2]
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter by aspect ratio and size
        # Digits are usually taller than they are wide, and take up 30-80% of the height
        if h > H * 0.3 and h < H * 0.95 and w > W * 0.02 and w < W * 0.3:
            # Maybe it's a digit
            boxes.append((x, y, w, h))
            
    # Sort boxes left to right
    boxes.sort(key=lambda b: b[0])
    
    # Merge boxes that heavily overlap horizontally (sometimes a digit is broken)
    merged = []
    for b in boxes:
        if not merged:
            merged.append(b)
        else:
            last = merged[-1]
            # Check horizontal overlap
            if b[0] < last[0] + last[2] * 0.5:
                # Merge
                x = min(last[0], b[0])
                y = min(last[1], b[1])
                xmax = max(last[0] + last[2], b[0] + b[2])
                ymax = max(last[1] + last[3], b[1] + b[3])
                merged[-1] = (x, y, xmax - x, ymax - y)
            else:
                merged.append(b)
                
    # Normalize
    norm_boxes = []
    for (x, y, w, h) in merged:
        cx = (x + w / 2) / W
        cy = (y + h / 2) / H
        nw = w / W
        nh = h / H
        norm_boxes.append({"cx": cx, "cy": cy, "w": nw, "h": nh})
        
    return norm_boxes

