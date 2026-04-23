import cv2
import numpy as np
import os

def analyze_and_save_dice(image_path, output_path):
    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Skipping: Could not load image at {image_path}")
        return
    
    # --- Resize ---
    scale_percent = 25 
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Detect the Dice
    _, dice_thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    dice_contours, _ = cv2.findContours(dice_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_dice_contours = []
    for cnt in dice_contours:
        area = cv2.contourArea(cnt)
        if area > 400: 
            valid_dice_contours.append(cnt)

    num_dice = len(valid_dice_contours)

    # 3. Detect the Pips (Spots)
    _, dark_regions = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

    dice_mask = np.zeros_like(gray)
    cv2.drawContours(dice_mask, valid_dice_contours, -1, 255, -1)

    isolated_pips = cv2.bitwise_and(dark_regions, dark_regions, mask=dice_mask)
    pip_contours, _ = cv2.findContours(isolated_pips, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    valid_pips = 0
    for cnt in pip_contours:
        area = cv2.contourArea(cnt)
        
        if 10 < area < 200: 
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
                
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            if circularity > 0.5: 
                valid_pips += 1
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(img, (int(x), int(y)), int(radius), (0, 0, 255), 2)

    cv2.drawContours(img, valid_dice_contours, -1, (0, 255, 0), 2)

    # --- NEW: Write Text in Bottom Left Corner ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (0, 255, 255) # Yellow text (BGR format) so it contrasts well against black
    thickness = 2
    
    # Calculate Y coordinates for the bottom left (leaving a small margin)
    text_y_pips = height - 20
    text_y_dice = height - 50
    
    cv2.putText(img, f"Total Points: {valid_pips}", (15, text_y_pips), font, font_scale, color, thickness)
    cv2.putText(img, f"Number of Dice: {num_dice}", (15, text_y_dice), font, font_scale, color, thickness)

    # 4. Save the Final Image
    cv2.imwrite(output_path, img)
    print(f"Processed: {image_path} -> Saved to {output_path} (Dice: {num_dice}, Pips: {valid_pips})")


# ==========================================
# BATCH PROCESSING LOGIC
# ==========================================

input_folder = "photos"
output_folder = "classified_bad"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created directory: {output_folder}")

print("-" * 30)
print(f"Starting batch processing...")
print("-" * 30)

# Iterate from 1 to 100
for i in range(1, 101):
    filename = f"{i}.jpg"
    in_path = os.path.join(input_folder, filename)
    out_path = os.path.join(output_folder, filename)
    
    # Only try to process the file if it actually exists in the folder
    if os.path.exists(in_path):
        analyze_and_save_dice(in_path, out_path)
    else:
        # Silently skip missing numbers (e.g., if you only have 1.jpg to 50.jpg)
        pass

print("-" * 30)
print("Batch processing complete! Check the 'classified' folder.")