import sys
import os
import torch
import cv2
import numpy as np
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.append(os.getcwd())

from model.models.detection_model import DetectionModel_MTL
from model.data.dataset import MTLDataset
from model.data.detections import Detections
from model.utils.ops import nms 

# --- CONFIGURATION ---
model_config_path = r'model/config/models/yolov8n.yaml'
train_config_path = r'model/config/training/fine_tune.yaml'
weights_path = r'runs/debug/debug_mtl_model.pt' 
output_dir = r'runs/debug/inference_results'

# FIX 1: Raise threshold so we don't draw 8000 garbage boxes
conf_threshold = 0.25  
iou_threshold = 0.5 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 10 * torch.log10(1.0 / mse)

def calculate_ssim(img1, img2):
    try:
        from skimage.metrics import structural_similarity as ssim
        i1 = img1.permute(1, 2, 0).cpu().numpy()
        i2 = img2.permute(1, 2, 0).cpu().numpy()
        return ssim(i1, i2, data_range=1.0, channel_axis=2)
    except ImportError:
        return 0.0

def box_iou_batch(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)

def compute_batch_metrics(pred_boxes, gt_boxes, iou_thres=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0, 0.0
    ious = box_iou_batch(pred_boxes, gt_boxes)
    iou_max, _ = ious.max(dim=0)
    tp = (iou_max > iou_thres).sum().item()
    fn = len(gt_boxes) - tp
    fp = len(pred_boxes) - tp
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return precision, recall

def debug_inference():
    print(f"Running Inference on: {device}")
    os.makedirs(output_dir, exist_ok=True)

    model = DetectionModel_MTL(model_config_path, verbose=False)
    model.to(device)

    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        print("✅ Weights loaded.")
    else:
        print("❌ Weights not found.")
        return

    model.eval()

    full_dataset = MTLDataset(train_config_path, mode='train')
    subset = Subset(full_dataset, range(10)) 
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=MTLDataset.collate_fn)
    class_names = full_dataset.config.get('names', {i: str(i) for i in range(80)})

    print("\n--- STARTING INFERENCE ---")
    
    for i, batch in enumerate(dataloader):
        input_img = batch['img'].to(device)
        gt_img = batch['gt_img'].to(device)
        img_h, img_w = input_img.shape[2:]
        
        with torch.no_grad():
            det_preds, llie_res = model(input_img)
            
            # 1. Transpose if needed
            if det_preds.shape[1] != 8400: 
                 det_preds = det_preds.transpose(1, 2)
            
            # 2. Convert xywh -> xyxy
            det_preds[..., :4] = xywh2xyxy(det_preds[..., :4])
            det_preds[..., 0].clamp_(0, img_w)
            det_preds[..., 1].clamp_(0, img_h)
            det_preds[..., 2].clamp_(0, img_w)
            det_preds[..., 3].clamp_(0, img_h)

            # 3. NMS (Filter garbage)
            det_preds = nms(det_preds, conf_threshold, iou_threshold)

        # Process LLIE
        enhanced_img = torch.clamp(input_img + llie_res, 0.0, 1.0)
        
        # Process GT
        current_gt_boxes = batch['bboxes'][batch['batch_idx'] == 0].to(device)
        current_gt_cls = batch['cls'][batch['batch_idx'] == 0].to(device)

        if len(current_gt_boxes) > 0:
            if current_gt_boxes.max() <= 1.0:
                current_gt_boxes[:, [0, 2]] *= img_w
                current_gt_boxes[:, [1, 3]] *= img_h
            current_gt_boxes = xywh2xyxy(current_gt_boxes)

        # --- METRICS ---
        if len(det_preds[0]) > 0:
            pred_boxes = det_preds[0][:, :4]
            precision, recall = compute_batch_metrics(pred_boxes, current_gt_boxes)
        else:
            precision, recall = 0.0, 0.0
            
        # FIX 2: Added PSNR/SSIM back
        psnr_val = calculate_psnr(enhanced_img[0], gt_img[0]).item()
        ssim_val = calculate_ssim(enhanced_img[0], gt_img[0])

        print(f"Img {i} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f} | Prec: {precision:.2f} | Recall: {recall:.2f}")

        # --- VISUALIZATION ---
        def to_cv2(t):
            img = t[0].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()

        vis_input = to_cv2(input_img)
        vis_enhanced = to_cv2(enhanced_img)
        vis_gt = to_cv2(gt_img)

        # Draw PREDS
        if len(det_preds[0]) > 0:
            for box in det_preds[0]:
                x1, y1, x2, y2 = map(int, box[:4].tolist())
                conf = box[4].item()
                cls = int(box[5].item())
                # Only draw high confidence for visualization to keep it clean
                label = f"{class_names.get(cls, str(cls))} {conf:.2f}"
                cv2.rectangle(vis_enhanced, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_enhanced, label, (x1, y1-5), 0, 0.5, (0, 255, 0), 1)

        # Draw GT
        if len(current_gt_boxes) > 0:
            for box, cls in zip(current_gt_boxes, current_gt_cls):
                x1, y1, x2, y2 = map(int, box.tolist())
                label = class_names.get(int(cls.item()), str(int(cls.item())))
                cv2.rectangle(vis_gt, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(vis_gt, label, (x1, y1-5), 0, 0.5, (255, 0, 0), 1)

        combined = np.hstack((vis_input, vis_enhanced, vis_gt))
        cv2.imwrite(os.path.join(output_dir, f"val_{i}.jpg"), combined)

    print(f"\n✅ Results saved to {output_dir}")

if __name__ == "__main__":
    debug_inference()
    
    
# 
# 
# import sys
# import os
# import torch
# import cv2
# import numpy as np
# import yaml
# import math
# from torch.utils.data import DataLoader, Subset

# # Ensure root path is in sys.path
# sys.path.append(os.getcwd())

# from model.models.detection_model import DetectionModel_MTL
# from model.data.dataset import MTLDataset
# from model.data.detections import Detections

# # --- CONFIGURATION ---
# model_config_path = r'model/config/models/yolov8n.yaml'
# train_config_path = r'model/config/training/fine_tune.yaml'
# # Point this to the weight file you just saved in debug_train
# weights_path = r'runs/debug/debug_mtl_model.pt' 
# output_dir = r'runs/debug/inference_results'

# # Setup Device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def calculate_psnr(img1, img2):
#     """
#     img1, img2: torch tensors (C, H, W) normalized 0-1
#     """
#     mse = torch.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return 100
#     return 10 * torch.log10(1.0 / mse)

# def calculate_ssim(img1, img2):
#     """
#     Simple wrapper for SSIM. 
#     Requires: pip install scikit-image
#     If not found, returns 0.0
#     """
#     try:
#         from skimage.metrics import structural_similarity as ssim
#         # Convert to numpy (H, W, C)
#         i1 = img1.permute(1, 2, 0).cpu().numpy()
#         i2 = img2.permute(1, 2, 0).cpu().numpy()
        
#         # Use simple averaging for multichannel
#         # data_range=1.0 since images are 0-1
#         score = ssim(i1, i2, data_range=1.0, channel_axis=2)
#         return score
#     except ImportError:
#         return 0.0

# def debug_inference():
#     print(f"Running Inference on: {device}")
#     os.makedirs(output_dir, exist_ok=True)

#     # 1. Load Model
#     print("Initializing Model...")
#     model = DetectionModel_MTL(model_config_path, verbose=False)
#     model.to(device)

#     # 2. Load Weights
#     if os.path.exists(weights_path):
#         print(f"Loading weights from {weights_path}...")
#         ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        
#         # Extract state_dict
#         if 'model' in ckpt:
#             state_dict = ckpt['model']
#         else:
#             state_dict = ckpt
            
#         model.load_state_dict(state_dict, strict=True) # Strict should pass now if configs match
#         print("✅ Weights loaded.")
#     else:
#         print(f"❌ Weights not found at {weights_path}. Please run debug_train.py first.")
#         return

#     # 3. Set Eval Mode
#     model.eval()
#     model.mode = 'eval' # Important for BaseModel to trigger NMS postprocessing

#     # 4. Load Data (Val Subset)
#     print("Initializing Val Data...")
#     # Using 'val' mode usually implies 'val_metadata.csv' exists
#     # If not, switch mode='train' just to test
#     full_dataset = MTLDataset(train_config_path, mode='train') 
    
#     # Take first 10 images
#     subset = Subset(full_dataset, range(10))
    
#     dataloader = DataLoader(
#         subset, 
#         batch_size=1, # Batch size 1 is easier for visualization logic
#         shuffle=False, 
#         collate_fn=MTLDataset.collate_fn
#     )

#     print("\n--- STARTING INFERENCE ---")
    
#     for i, batch in enumerate(dataloader):
#         input_img = batch['img'].to(device) # (B, 3, H, W)
#         gt_img = batch['gt_img'].to(device)
        
#         with torch.no_grad():
#             # Model returns tuple in eval mode: (det_preds, llie_residual)
#             det_preds, llie_res = model(input_img)

#         # --- PROCESS LLIE OUTPUT ---
#         # Enhanced = Input + Residual
#         enhanced_img = input_img + llie_res
#         enhanced_img = torch.clamp(enhanced_img, 0.0, 1.0)

#         # Calculate Metrics (on first image of batch)
#         psnr_val = calculate_psnr(enhanced_img[0], gt_img[0])
#         ssim_val = calculate_ssim(enhanced_img[0], gt_img[0])
        
#         print(f"Img {i} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f}")

#         # --- VISUALIZATION ---
#         # Convert tensors to BGR numpy for OpenCV
#         def to_cv2(t):
#             img = t[0].permute(1, 2, 0).cpu().numpy()
#             img = (img * 255).astype(np.uint8)
#             return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#         vis_input = to_cv2(input_img)
#         vis_enhanced = to_cv2(enhanced_img)
#         vis_gt = to_cv2(gt_img)

#         # --- DRAW DETECTIONS ---
#         # det_preds is a list of tensors (one per image)
#         # Using your Detections class wrapper
#         if len(det_preds) > 0 and det_preds[0] is not None:
#             # Check if detections exist
#             try:
#                 detections = Detections.from_yolo(det_preds[0])
#                 # We draw on the Enhanced Image
#                 # Note: 'view' usually modifies image in-place or returns it
#                 # We need class names config. Assuming COCO 80 classes default if not loaded
#                 class_names = full_dataset.config.get('names', {i: str(i) for i in range(80)})
                
#                 # Check if your Detections.view supports returning the image
#                 # Or if it modifies in place. Assuming it modifies 'vis_enhanced'
#                 detections.view(vis_enhanced, classes_dict=class_names)
#             except Exception as e:
#                 print(f"  -> Warning: Could not draw detections: {e}")

#         # Stack images horizontally: Input | Enhanced | GT
#         # Add labels
#         cv2.putText(vis_input, "Low Light", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#         cv2.putText(vis_enhanced, f"Enhanced (PSNR:{psnr_val:.1f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         cv2.putText(vis_gt, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#         combined = np.hstack((vis_input, vis_enhanced, vis_gt))
        
#         # Save
#         save_file = os.path.join(output_dir, f"val_{i}.jpg")
#         cv2.imwrite(save_file, combined)
        
#     print(f"\n✅ Results saved to {output_dir}")

# if __name__ == "__main__":
#     debug_inference()