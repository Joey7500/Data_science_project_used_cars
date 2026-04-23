import cv2
import numpy as np
import os

# ── TUNEABLE CONSTANTS ──────────────────────────────────────────────────────
SCALE_PERCENT     = 25    # resize factor  (2592 → 648 px wide)

MIN_DIE_AREA      = 1000  # px²  – blobs smaller than this are dust
EXPECTED_DIE_AREA = 2500  # px²  – one die ≈ 50×50 px after resize

ADAPTIVE_BLOCK    = 27    # adaptive-threshold neighbourhood size
ADAPTIVE_C        = 15    # adaptive-threshold constant

MIN_PIP_BLOB_AREA = 40    # px²  – ignore tiny speckle noise
MAX_PIP_BLOB_AREA = 800   # px²  – ignore large shadows

EXPECTED_PIP_AREA = 154   # px²  – 11×14 single pip
PIP_RADIUS_EST    = 6     # px   – estimated pip radius after resize

# ── KEY CONSTANT FOR THE PEANUT FIX ────────────────────────────────────────
# INCREASED from 0.40 to 0.60 to aggressively slice touching pips apart!
DIST_THRESH_ABS   = PIP_RADIUS_EST * 0.60  
# ────────────────────────────────────────────────────────────────────────────

def count_pips_via_distance_transform(pip_mask):
    dist = cv2.distanceTransform(pip_mask, cv2.DIST_L2, 5)

    _, sure_fg = cv2.threshold(dist, DIST_THRESH_ABS, 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(sure_fg, connectivity=8)

    centers = []
    for i in range(1, num_labels):           
        if stats[i, cv2.CC_STAT_AREA] >= 2:  
            centers.append((int(centroids[i][0]), int(centroids[i][1])))

    return len(centers), centers

def analyze_and_save_dice(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Skipping: could not load {image_path}")
        return None

    # ── 1. Resize ──────────────────────────────────────────────────────────
    w = int(img.shape[1] * SCALE_PERCENT / 100)
    h = int(img.shape[0] * SCALE_PERCENT / 100)
    img  = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── 2. Dice body detection (Otsu) ──────────────────────────────────────
    blurred_dice = cv2.GaussianBlur(gray, (7, 7), 0)
    _, dice_thresh = cv2.threshold(
        blurred_dice, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    dice_contours, _ = cv2.findContours(
        dice_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []
    num_dice = 0 
    dice_mask = np.zeros_like(gray)

    for cnt in dice_contours:
        area = cv2.contourArea(cnt)
        if area > MIN_DIE_AREA:
            valid_contours.append(cnt)
            cv2.drawContours(dice_mask, [cnt], -1, 255, -1)
            
            dice_in_this_blob = max(1, round(area / EXPECTED_DIE_AREA))
            num_dice += dice_in_this_blob

    # ── 3. Pip mask ────────────────────────────────────────────────────────
    blurred_pips = cv2.GaussianBlur(gray, (5, 5), 0)
    dark_regions = cv2.adaptiveThreshold(
        blurred_pips, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK, ADAPTIVE_C
    )

    # Restrict to dice area only
    pip_mask = cv2.bitwise_and(dark_regions, dark_regions, mask=dice_mask)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pip_mask = cv2.morphologyEx(pip_mask, cv2.MORPH_OPEN, open_kernel)

    # DECREASED kernel from 5x5 to 3x3 so we don't accidentally weld pips together!
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pip_mask = cv2.morphologyEx(pip_mask, cv2.MORPH_CLOSE, close_kernel)

    # ── 4. Count pips with distance transform ─────────────────────────────
    total_pips, pip_centers = count_pips_via_distance_transform(pip_mask)

    # ── 5. Per-die values ─────────────────────────────────────────────────
    die_label_map = np.zeros((h, w), dtype=np.int32)
    for idx, cnt in enumerate(valid_contours, start=1):
        cv2.drawContours(die_label_map, [cnt], -1, idx, -1)

    die_pips: dict[int, int] = {}
    for (cx, cy) in pip_centers:
        if 0 <= cy < h and 0 <= cx < w:
            die_id = die_label_map[cy, cx]
            if die_id > 0:
                die_pips[die_id] = die_pips.get(die_id, 0) + 1

    # ── 6. Draw results ───────────────────────────────────────────────────
    for (cx, cy) in pip_centers:
        cv2.circle(img, (cx, cy), int(PIP_RADIUS_EST * 1.3), (0, 0, 255), 2)

    for idx, cnt in enumerate(valid_contours, start=1):
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
        x, y, bw, bh = cv2.boundingRect(cnt)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"Total Points: {total_pips}", (15, h - 20),
                font, 0.8, (0, 255, 255), 2)
    cv2.putText(img, f"Number of Dice: {num_dice}", (15, h - 50),
                font, 0.8, (0, 255, 255), 2)

    cv2.imwrite(output_path, img)
    print(
        f"{os.path.basename(image_path):12s} -> "
        f"Dice: {num_dice}, Total pips: {total_pips}, "
        f"Per-die: {die_pips}"
    )
    return die_pips