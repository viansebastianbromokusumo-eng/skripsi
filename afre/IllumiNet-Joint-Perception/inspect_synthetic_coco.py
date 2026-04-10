import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

# Path to your new dataset CSV
CSV_PATH = r'datasets/coco_mini_dark/train_metadata.csv' 
BASE_DIR = r'datasets/coco_mini_dark' # Root of the dataset

def inspect_dataset():
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows.")

    # Pick a random sample
    row = df.sample(1).iloc[0]
    
    # Construct paths (handling ./ prefix)
    p_low = row['image_low'].replace('./', '')
    p_high = row['image_high'].replace('./', '')
    p_lbl = row['label'].replace('./', '')
    
    full_low = os.path.join(BASE_DIR, p_low)
    full_high = os.path.join(BASE_DIR, p_high)
    full_lbl = os.path.join(BASE_DIR, p_lbl)

    # Load Images
    img_low = cv2.imread(full_low)
    img_high = cv2.imread(full_high)
    
    if img_low is None:
        print(f"❌ Could not load low image: {full_low}")
        return

    # Draw Labels on High Image
    h, w, _ = img_high.shape
    if os.path.exists(full_lbl):
        with open(full_lbl, 'r') as f:
            boxes = f.readlines()
            for b in boxes:
                cls, cx, cy, bw, bh = map(float, b.split())
                # Convert YOLO (norm) to xyxy (pixels)
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                cv2.rectangle(img_high, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(cv2.cvtColor(img_low, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Input (Low Light)")
    ax[1].imshow(cv2.cvtColor(img_high, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Ground Truth (High Light)")
    plt.show()

if __name__ == "__main__":
    inspect_dataset()