import torch
import cv2
import numpy as np
import requests
import os
from model.models.detection_model import DetectionModel_MTL
from model.utils.ops import nms

# --- CONFIGURATION ---
MODEL_CONFIG = 'model/config/models/yolov8n.yaml'
# Download official weights if you don't have them
WEIGHTS_URL = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt'
WEIGHTS_PATH = 'yolov8n.pt'
IMG_URL = 'http://images.cocodataset.org/val2017/000000000139.jpg' # Standard living room image
IMG_PATH = 'test_coco.jpg'

def download_file(url, path):
    if not os.path.exists(path):
        print(f"⬇️ Downloading {path}...")
        r = requests.get(url)
        with open(path, 'wb') as f:
            f.write(r.content)

def test_pretrained_baseline():
    # 1. Setup Data & Weights
    download_file(WEIGHTS_URL, WEIGHTS_PATH)
    download_file(IMG_URL, IMG_PATH)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Running Baseline Test on {device}")

    # 2. Load Your Model Structure
    # mode='train' to bypass your internal postprocess for a moment so we can inspect raw
    model = DetectionModel_MTL(MODEL_CONFIG, verbose=False, mode='train') 
    model.to(device)

    # 3. Load Official YOLOv8 Weights
    # strict=False allows loading YOLO weights into your MTL structure 
    # (It will just skip the LLIE head layers, which is fine)
    ckpt = torch.load(WEIGHTS_PATH, map_location=device)
    # Handle both Ultralytics format and standard state_dict
    state_dict = ckpt['model'].float().state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt
    
    # We might need to adjust keys if your model.model matches their model.model
    # Your class wraps Ultralytics parse_model, so keys usually match.
    try:
        model.load_state_dict(state_dict, strict=False)
        print("✅ Official YOLOv8n weights loaded (LLIE layers skipped).")
    except Exception as e:
        print(f"⚠️ Weight loading warning (expected for LLIE): {e}")

    model.eval()

    # 4. Prepare Image
    img = cv2.imread(IMG_PATH)
    h0, w0 = img.shape[:2]
    # Resize to 640x640 standard
    img_resized = cv2.resize(img, (640, 640))
    input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().to(device)
    input_tensor /= 255.0
    input_tensor = input_tensor.unsqueeze(0)

    # 5. Run Inference (Get RAW Output)
    with torch.no_grad():
        # Call _predict directly to get raw output (bypass postprocess)
        preds = model._predict(input_tensor)
        
        # Unpack tuple (Detection, LLIE)
        det_out = preds[0]
        
        # Transpose if needed [B, 84, 8400] -> [B, 8400, 84]
        if det_out.shape[1] != 8400:
            det_out = det_out.transpose(1, 2)
            
        print("\n🔍 RAW OUTPUT ANALYSIS (Official Weights):")
        print(f"  Shape: {det_out.shape}")
        
        # Analyze Box Coordinates (Indices 0-3)
        box_data = det_out[0, :, :4]
        print(f"  Max Value (likely x,y): {box_data.max().item():.2f}")
        print(f"  Min Value: {box_data.min().item():.2f}")
        print(f"  Mean Value: {box_data.mean().item():.2f}")
        
        # Check a high-confidence prediction
        cls_data = det_out[0, :, 4:]
        scores, cls_ids = cls_data.max(dim=1)
        best_idx = scores.argmax()
        best_box = box_data[best_idx]
        
        print(f"\n🏆 Best Detection (Conf: {scores[best_idx]:.4f}):")
        print(f"  Raw Box Values: {best_box.tolist()}")
        
        # 6. DETERMINE SPACE
        x, y, w, h = best_box
        if x > 1.0 and w > 1.0:
            print("  ✅ Conclusion: Output is ABSOLUTE PIXELS (No scaling needed?)")
        elif x <= 1.0 and w <= 1.0:
            print("  ⚠️ Conclusion: Output is NORMALIZED (Needs * 640)")
        elif w < 20 and x > 50:
            print("  ⚠️ Conclusion: Output is HYBRID (x,y=Pixels, w,h=Grid?)")
        else:
            print("  ❓ Conclusion: Unknown/DFL Distribution")

    # 7. Apply Manual Decoding based on discovery (Try Standard)
    # Standard YOLOv8 usually outputs [xc, yc, w, h] in absolute pixels if the head includes DFL
    # Let's try drawing it AS IS first
    
    cx, cy, w, h = best_box.cpu().numpy()
    
    # If Grid Space logic applies (Manual Decode):
    # This assumes output is pixels. If it draws a tiny box, we know it's unscaled.
    x1 = int(cx - w/2)
    y1 = int(cy - h/2)
    x2 = int(cx + w/2)
    y2 = int(cy + h/2)
    
    print(f"\nDrawing Box: {x1, y1, x2, y2}")
    cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite('test_baseline_result.jpg', img_resized)
    print("Snapshot saved to test_baseline_result.jpg")

if __name__ == "__main__":
    test_pretrained_baseline()