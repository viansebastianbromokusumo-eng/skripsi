import os
import shutil
import cv2
import numpy as np
import csv
import json
import random
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
# We use the COCO Validation set (5000 images) as our "Mini Train" set
IMG_URL = "http://images.cocodataset.org/zips/val2017.zip"
LBL_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
OUTPUT_DIR = 'datasets/coco_mini_dark'

# Darkening Parameters
GAMMA_RANGE = (1.5, 2.5)   # Higher = Darker
LINEAR_REDUCTION = (0, 30) # Subtract constant value (make shadows crushed)
NOISE_SIGMA = 5            # Add grain

def download_and_unzip(url, target_dir):
    filename = url.split('/')[-1]
    if not os.path.exists(filename):
        print(f"⬇️ Downloading {filename}...")
        response = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    print(f"📦 Unzipping {filename}...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

def apply_heavy_darkness(image):
    """
    Applies aggressive darkening using Gamma + Linear Subtraction + Noise.
    """
    # 1. Linear Subtraction (Crush the blacks)
    # Subtract a random value from all pixels to shift histogram left
    sub_val = np.random.randint(*LINEAR_REDUCTION)
    dark_img = cv2.subtract(image, np.full(image.shape, sub_val, dtype='uint8'))

    # 2. Gamma Correction (Compress the midtones)
    # V_out = V_in ^ gamma (where gamma > 1 makes it darker)
    gamma = np.random.uniform(*GAMMA_RANGE)
    
    # Normalize to 0-1, apply power, scale back
    # Note: We use look-up table for speed
    invGamma = gamma # We WANT the power to be > 1 to darken
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    dark_img = cv2.LUT(dark_img, table)

    # 3. Gaussian Noise
    row, col, ch = dark_img.shape
    mean = 0
    gauss = np.random.normal(mean, NOISE_SIGMA, (row, col, ch)).reshape(row, col, ch)
    noisy_img = dark_img + gauss
    
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def convert_coco_json_to_yolo(json_path, output_dir, valid_image_ids):
    """
    Converts COCO JSON annotations to individual YOLO .txt files.
    """
    print("🔄 Converting COCO JSON to YOLO format...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create map of image_id -> file_name
    img_map = {img['id']: img for img in data['images']}
    
    # Process annotations
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by image_id
    ann_by_img = {}
    for ann in tqdm(data['annotations'], desc="Parsing Anns"):
        img_id = ann['image_id']
        if img_id not in valid_image_ids: continue # Skip if not in our split
        
        if img_id not in ann_by_img: ann_by_img[img_id] = []
        
        # COCO bbox: [x_min, y_min, width, height] (Absolute)
        # YOLO bbox: [class, x_center, y_center, width, height] (Normalized)
        img_info = img_map[img_id]
        dw = 1. / img_info['width']
        dh = 1. / img_info['height']
        
        x, y, w, h = ann['bbox']
        x_center = (x + w / 2.0) * dw
        y_center = (y + h / 2.0) * dh
        w_norm = w * dw
        h_norm = h * dh
        
        # Clamp
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w_norm = max(0, min(1, w_norm))
        h_norm = max(0, min(1, h_norm))
        
        cid = ann['category_id'] 
        # COCO category IDs are not contiguous (1-90). We need to map them if strict 0-79 is needed.
        # For simplicity in this debug, we just keep the ID. 
        # Note: Standard YOLO models map these. If your class list is 0-79, 
        # you might see class '90' which causes error. 
        # Let's Remap to 0-79 index based on 'categories' list order
        
        ann_by_img[img_id].append(f"{cid} {x_center} {y_center} {w_norm} {h_norm}")

    return ann_by_img, img_map

def prepare_data():
    # 1. Download Data
    # download_and_unzip(IMG_URL, '.')
    # download_and_unzip(LBL_URL, '.')

    # 2. Setup Directories
    base_dir = Path(OUTPUT_DIR)
    (base_dir / 'images/low').mkdir(parents=True, exist_ok=True)
    (base_dir / 'images/high').mkdir(parents=True, exist_ok=True)
    (base_dir / 'labels').mkdir(parents=True, exist_ok=True)

    # 3. Load COCO JSON to get valid images
    # We use annotations_trainval2017/instances_val2017.json
    json_path = 'annotations/instances_val2017.json'
    
    # Get List of images in val2017
    src_img_dir = Path('val2017')
    all_img_files = list(src_img_dir.glob('*.jpg'))
    
    # Filter: ensure we only use images that actually exist
    valid_ids = set()
    for p in all_img_files:
        try:
            valid_ids.add(int(p.stem))
        except: pass
        
    print(f"found {len(valid_ids)} valid images.")

    # 4. Generate YOLO Labels
    # We need to remap categories to 0-79 contiguous
    with open(json_path, 'r') as f:
        data = json.load(f)
    cats = sorted(data['categories'], key=lambda x: x['id'])
    cat_map = {c['id']: i for i, c in enumerate(cats)} # Map original ID to 0-79

    # Re-run conversion with mapping
    print("🔄 Generating Label Files...")
    img_map = {img['id']: img for img in data['images']}
    
    for img_id in tqdm(valid_ids):
        # Find annotations for this image
        anns = [a for a in data['annotations'] if a['image_id'] == img_id]
        
        img_info = img_map[img_id]
        filename = img_info['file_name']
        
        # Label string content
        lbl_content = []
        for ann in anns:
            cid = cat_map.get(ann['category_id'], 0) # Remap
            x, y, w, h = ann['bbox']
            
            # Normalize
            dw = 1. / img_info['width']
            dh = 1. / img_info['height']
            xc = (x + w/2) * dw
            yc = (y + h/2) * dh
            wn = w * dw
            hn = h * dh
            
            lbl_content.append(f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
        
        # Save Label File
        lbl_name = Path(filename).stem + ".txt"
        with open(base_dir / 'labels' / lbl_name, 'w') as f:
            f.write('\n'.join(lbl_content))

    # 5. Process Images (Darken)
    print("🌑 Generating Dark Images...")
    data_entries = []
    
    for img_file in tqdm(all_img_files):
        filename = img_file.name
        
        # Read High
        img_high = cv2.imread(str(img_file))
        if img_high is None: continue
        
        # Generate Low
        img_low = apply_heavy_darkness(img_high)
        
        # Save
        cv2.imwrite(str(base_dir / 'images/low' / filename), img_low)
        cv2.imwrite(str(base_dir / 'images/high' / filename), img_high)
        
        # Record
        data_entries.append({
            'image_low': f"./images/low/{filename}",
            'image_high': f"./images/high/{filename}",
            'label': f"./labels/{img_file.stem}.txt"
        })

    # 6. Generate CSVs (Split 90/10)
    random.shuffle(data_entries)
    split_idx = int(len(data_entries) * 0.9)
    train_data = data_entries[:split_idx]
    val_data = data_entries[split_idx:]

    def write_csv(name, data, split):
        with open(base_dir / name, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image_low', 'image_high', 'label', 'split'])
            writer.writeheader()
            for row in data:
                row['split'] = split
                writer.writerow(row)

    write_csv('train_metadata.csv', train_data, 'train')
    write_csv('val_metadata.csv', val_data, 'val')

    # 7. YAML
    yaml_content = {
        'path': os.path.abspath(OUTPUT_DIR),
        'names': {i: c['name'] for i, c in enumerate(cats)}
    }
    with open(base_dir / 'coco_mini.yaml', 'w') as f:
        import yaml
        yaml.dump(yaml_content, f)

    print(f"\n✅ Ready! Config: {base_dir}/coco_mini.yaml")

if __name__ == "__main__":
    prepare_data()