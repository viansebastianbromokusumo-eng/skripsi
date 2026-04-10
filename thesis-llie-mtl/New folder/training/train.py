import torch
import time
# import pyiqa
# import lpips
import torchvision
from tqdm import tqdm
from torch.optim import AdamW
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import nms, box_iou
from training.callbacks import set_task_specific_params, CosineAnnealingWarmupLR, CompositeScoreCallback, BestMAPCallback
from losses.total_loss import compute_total_loss_unsup, compute_total_loss
from losses.llie_loss import psnr, ssim_metric
import numpy as np 


# try:
#     import pyiqa
#     import lpips
#     METRICS_AVAILABLE = True
# except ImportError:
#     print("⚠️ Warning: 'pyiqa' or 'lpips' not found. validation metrics will be skipped.")
#     print("👉 Please run: pip install pyiqa lpips")
#     METRICS_AVAILABLE = False


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# map_metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=False, max_detection_thresholds=[100]).to(DEVICE)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """
    Runs NMS on the raw YOLO output.
    Input: prediction [batch, anchors, 5 + classes] (e.g. [B, 8400, 85])
    Output: List of tensors [x1, y1, x2, y2, conf, cls]
    """
    # Checks if your model output is [B, 85, 8400] (v8 style) or [B, 8400, 85] (v5 style)
    # YOLOv8 usually outputs [B, 4+cls, N]. We need to transpose it to [B, N, 4+cls]
    if prediction.shape[1] == 4 + 80: # Assuming 80 classes, adjust if needed
         prediction = prediction.transpose(1, 2)
         
    bs = prediction.shape[0]
    nc = prediction.shape[2] - 4  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    
    for xi, x in enumerate(prediction):  # Iterate over batch
        x = x[xc[xi]]  # Apply confidence constraint

        if not x.shape[0]:
            continue

        # Split boxes and scores
        box = x[:, :4] # cx, cy, w, h
        cls_scores = x[:, 4:] # classes
        
        # Calculate box (cx,cy,w,h) -> (x1,y1,x2,y2)
        # Note: This depends on how your model outputs boxes. 
        # Assuming standard xywh center format:
        box = xywh2xyxy(box) 

        # Get best class and score
        conf, j = cls_scores.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Batched NMS
        boxes, scores, idxs = x[:, :4], x[:, 4], x[:, 5]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
            
        output[xi] = x[i]

    return output

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def xywh_norm_to_xyxy_abs(boxes, img_size=640):
    """
    boxes: Tensor [N, 4] in normalized cx,cy,w,h
    returns: Tensor [N, 4] in absolute xyxy
    """
    if boxes.numel() == 0:
        return boxes

    cx, cy, w, h = boxes.unbind(dim=1)

    x1 = (cx - w / 2) * img_size
    y1 = (cy - h / 2) * img_size
    x2 = (cx + w / 2) * img_size
    y2 = (cy + h / 2) * img_size

    return torch.stack([x1, y1, x2, y2], dim=1)

def validate_unsup(model, val_loader, det_loss_fn, enhance_loss_fn, iou_v=torch.tensor([0.5], dtype=torch.float32)):
    model.eval()

    total_val_loss = 0
    total_val_det_loss = 0
    total_val_enhance_loss = 0

    # total_psnr = 0
    # total_ssim_count = 0
    # total_ssim = 0
    total_niqe = 0 
    total_brisque = 0 
    count = 0 

    metrics = {}
    
    # Device setup
    device = next(model.parameters()).device

    val_pbar = tqdm(val_loader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_pbar): # Changed 'batch_datain' back to 'batch_data'

            # 💡 CRITICAL: Ensure data is moved to the correct device (fix from previous step)
            low_imgs = batch_data['images_low'].to(device)
            # high_imgs = batch_data['images_high'].to(device)

            # Move det_targets contents to device
            det_targets = {
                'cls': batch_data['cls'].to(device),
                'bboxes': batch_data['bboxes'].to(device),
                'batch_idx': batch_data['batch_idx'].to(device)
            }

            # 1. Compute Loss and Get Outputs
            # NOTE: compute_total_loss_unsup must be defined elsewhere
            with torch.cuda.amp.autocast():
                total_loss, det_loss, enhance_loss, det_out, llie_out, llie_residuals = compute_total_loss_unsup(
                    model, low_imgs, det_targets, det_loss_fn, enhance_loss_fn)

            total_val_loss += total_loss.item()
            total_val_det_loss += det_loss.item()
            total_val_enhance_loss += enhance_loss.item()

            # 2. Enhancement Metrics (PSNR & SSIM)
            # NOTE: psnr and ssim_metric must be defined elsewhere
            niqe_score = metrics['niqe'](llie_out).mean().item()
            brisque_score = metrics['brisque'](llie_out).mean().item()

            total_niqe += niqe_score
            total_brisque += brisque_score
            count += 1

            val_pbar.set_postfix({
                'NIQE': f'{niqe_score:.2f}', 
                'BRISQUE': f'{brisque_score:.2f}'
            })

    # 4. Final Averaging
    num_batches = len(val_loader)

    avg_val_loss = total_val_loss / num_batches
    avg_val_det_loss = total_val_det_loss / num_batches
    avg_val_enhance_loss = total_val_enhance_loss / num_batches

    avg_niqe = total_niqe / count
    avg_brisque = total_brisque / count
        
     # 5. Finalize Detection Metrics (The new core step)
    map_results = map_metric.compute()
    
    # Extract specific values (mAP50 corresponds to iou_threshold=0.5)
    # The output structure is a dict, results['map_50'] gives mAP@0.5
    map50 = map_results['map_50'].item()
    
    # Precision and Recall (often averaged over all classes/confidence thresholds)
    # Note: TorchMetrics does not easily give single P/R values like your dummy output.
    # It provides Average Precision per class. We can approximate P/R from the mAP results.
    # For simplicity, we can use a dummy value or a calculated average for P/R if a single number is required.
    # A more rigorous approach requires implementing a specific P/R calculation after NMS.
    # Using the overall mean/weighted average from TorchMetrics:
    avg_precision = map_results['map'].item() # This is typically mAP@[.5:.95]
    avg_recall = map_results['mar_100'].item() # Max Average Recall at 100 detections

    model.train() # Set model back to train mode

    metrics = {
        'avg_loss': avg_val_loss,
        'avg_det_loss': avg_val_det_loss,
        'avg_enh_loss': avg_val_enhance_loss,
        'val_niqe': avg_niqe,
        'val_brisque': avg_brisque,
        'precision': float(avg_precision),
        'recall': float(avg_recall),
        'mAP50': float(map50)
    }
    return metrics

def compute_pr_loop(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, iou_thresh=0.5):
    """
    Computes Precision and Recall natively on tensors for the validation loop.
    Assumes boxes are in absolute [x1, y1, x2, y2] format.
    """
    if len(gt_boxes) == 0:
        return (0.0, 0.0) if len(pred_boxes) > 0 else (1.0, 1.0)
    
    if len(pred_boxes) == 0:
        return 0.0, 0.0

    # Calculate IoU [N_preds, M_gts]
    ious = box_iou(pred_boxes, gt_boxes) 

    # Precision: Did prediction hit ANY GT?
    max_iou_per_pred, _ = ious.max(dim=1)
    tp = (max_iou_per_pred >= iou_thresh).sum().item()
    fp = len(pred_boxes) - tp
    precision = tp / (tp + fp + 1e-6)

    # Recall: Was GT hit by ANY prediction?
    max_iou_per_gt, _ = ious.max(dim=0)
    tp_gt = (max_iou_per_gt >= iou_thresh).sum().item()
    fn = len(gt_boxes) - tp_gt
    recall = tp_gt / (tp_gt + fn + 1e-6)

    return precision, recall

# def train_unsup(
#     model, 
#     name, 
#     train_loader, 
#     val_loader,
#     det_loss_fn, 
#     enh_loss_fn, 
#     epochs
# ):
#     warmup_epochs = 2 
#     initial_lr = 1e-2
#     param_groups = set_task_specific_params(model, decay=1e-4)
#     optimizer = AdamW(param_groups, lr=initial_lr)
    
#     total_steps = epochs * len(train_loader)
#     warmup_steps = int(warmup_epochs * len(train_loader))
#     scheduler = CosineAnnealingWarmupLR(optimizer, total_steps, warmup_steps)

#     callback = CompositeScoreCallback(model_name=name)
    
#     scaler = torch.cuda.amp.GradScaler()
    
#     history = []
    
#     model.train() 
    
#     for epoch in range(epochs):
#         start_time = time.time()
#         total_epoch_loss, total_det_loss, total_enhance_loss = 0, 0, 0

#         train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)

#         for batch_idx, batch_data in enumerate(train_pbar):

#             # 💡 CRITICAL: Ensure model's device is correctly retrieved
#             device = next(model.parameters()).device
#             low_imgs = batch_data['images_low'].to(device)
#             high_imgs = batch_data['images_high'].to(device)

#             # Move det_targets contents to device
#             det_targets = {
#                 'cls': batch_data['cls'].to(device),
#                 'bboxes': batch_data['bboxes'].to(device),
#                 'batch_idx': batch_data['batch_idx'].to(device)
#             }
            
#             optimizer.zero_grad() 
#             with torch.cuda.amp.autocast(): 
#                 total_loss, det_loss, enhance_loss, _, _, _ = compute_total_loss_unsup(
#                     model, low_imgs, high_imgs, det_targets, det_loss_fn, enh_loss_fn)
                
#             scaler.scale(total_loss).backward() 
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
#             scheduler.step() 
            
#             total_epoch_loss += total_loss.item()
#             total_det_loss += det_loss.item()
#             total_enhance_loss += enhance_loss.item()

#             train_pbar.set_postfix({
#                 'TLoss': f'{total_loss.item():.4f}',
#                 'DetLoss': f'{det_loss.item():.4f}',
#                 'EnhLoss': f'{enhance_loss.item():.4f}',
#                 'LR_BackBone': f'{optimizer.param_groups[0]["lr"]:.2e}',
#                 'LR_Det': f'{optimizer.param_groups[1]["lr"]:.2e}',
#                 'LR_Dec': f'{optimizer.param_groups[2]["lr"]:.2e}'
#             })
            
#             avg_train_loss = total_epoch_loss / len(train_loader)
#             avg_train_det_loss = total_det_loss / len(train_loader)
#             avg_train_enhance_loss = total_enhance_loss / len(train_loader)

#             # 2. Run Validation
#             val_metrics = validate_unsup(model, val_loader, det_loss_fn, enh_loss_fn)
    
#             # 3. Compile Metrics
#             epoch_history = {
#                 'epoch': epoch,
#                 'train_loss_total': avg_train_loss,
#                 'train_loss_det': avg_train_det_loss,
#                 'train_loss_enh': avg_train_enhance_loss,
#                 'val_loss_total': val_metrics['avg_loss'],
#                 'val_loss_det': val_metrics['avg_det_loss'],
#                 'val_loss_enh': val_metrics['avg_enh_loss'],
#                 'val_niqe': val_metrics['avg_niqe'],
#                 'val_brisque': val_metrics['avg_brisque'],
#                 'val_precision': val_metrics['precision'],
#                 'val_recall': val_metrics['recall'],
#                 'val_mAP50': val_metrics['mAP50'],
#                 'lr_backbone': optimizer.param_groups[0]['lr'],
#                 'lr_det_head': optimizer.param_groups[1]['lr'],
#                 'lr_decoder': optimizer.param_groups[2]['lr']
#             }
#             history.append(epoch_history)

#             # 4. Run the Multi-Metric Callback (NEW)
#             callback.check_and_save(model, epoch_history, epoch)

#             print(f"\n--- Epoch {epoch} Summary ({time.time() - start_time:.2f}s) ---")
#             print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['avg_loss']:.4f}")
#             print(f"Enhancement: PSNR={val_metrics['psnr']:.2f}, SSIM={val_metrics['ssim']:.3f}")
#             print(f"Detection: P={val_metrics['precision']:.3f}, R={val_metrics['recall']:.3f}, mAP@0.5={val_metrics['mAP50']:.3f}")
#             print("-------------------------------------------\n")

#         # 5. Final Save (Optional, but good for last epoch)
#         final_save_path = f'final_{name}_epoch{epochs}.pt'
#         torch.save(model.state_dict(), final_save_path)
#         print(f"Final model weights saved to {final_save_path}")

#         # 6. Return full training history
#         return history
                


# class UnsupervisedTrainer:
#     def __init__(self, 
#                  model, 
#                  train_loader, 
#                  val_loader, 
#                  loss_fn,
#                  experiment_name='unsupervised_llie',
#                  device=None,
#                  initial_lr=1e-4,
#                  weight_decay=1e-4):
        
#         self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = model.to(self.device)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.loss_fn = loss_fn
#         self.name = experiment_name
        
#         # Optimizer (Standard AdamW usually works best for Zero-DCE types)
#         self.optimizer = AdamW(self.model.parameters(), lr=initial_lr, weight_decay=weight_decay)
#         self.scaler = torch.cuda.amp.GradScaler()
        
#         # Initialize Metrics (Lazy loading to save memory if not needed)
#         self.metrics = {}
#         if METRICS_AVAILABLE:
#             print("Initializing IQA Metrics (this may take a moment)...")
#             # NIQE: Natural Image Quality Evaluator (Lower is better)
#             self.metrics['niqe'] = pyiqa.create_metric('niqe', device=self.device)
#             # BRISQUE: Blind/Referenceless Image Spatial Quality Evaluator (Lower is better)
#             self.metrics['brisque'] = pyiqa.create_metric('brisque', device=self.device)
#             # LPIPS: Learned Perceptual Image Patch Similarity (Lower is better)
#             # Note: LPIPS usually requires a reference. If you strictly have NO reference, 
#             # you might compare Input vs Output (identity loss) or skip this.
#             # I will include it assuming you might have unpaired high-quality shots or 
#             # want to measure deviation from input (content preservation).
#             self.metrics['lpips'] = lpips.LPIPS(net='alex').to(self.device)

#         # Track "Best" model (Lower Score = Better for these metrics)
#         self.best_score = float('inf') 
#         self.history = []

#     def _train_epoch(self, epoch, total_epochs):
#         self.model.train()
#         start_time = time.time()
        
#         total_loss = 0
#         loss_components = {'spa': 0, 'col': 0, 'exp': 0, 'tv': 0}

#         pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}/{total_epochs}", leave=True)

#         for batch_data in pbar:
#             # 1. Prepare Data (Unsupervised = Only Low Light Images needed)
#             # Adjust key name if your loader uses 'images_low' or just 'image'
#             low_imgs = batch_data['images_low'].to(self.device)

#             self.optimizer.zero_grad()

#             # 2. Forward & Loss
#             with torch.cuda.amp.autocast():
#                 # Expecting model to return (enhanced, curve_map)
#                 enhanced_img, illu_map = self.model(low_imgs)
                
#                 # Calculate the Unsupervised Loss
#                 # Note: We calculate individual losses just for logging purposes here
#                 loss = self.loss_fn(low_imgs, enhanced_img, illu_map)
                
#             # 3. Backward
#             self.scaler.scale(loss).backward()
#             self.scaler.step(self.optimizer)
#             self.scaler.update()

#             # 4. Logging
#             total_loss += loss.item()
#             pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

#         avg_loss = total_loss / len(self.train_loader)
#         return {'train_loss': avg_loss, 'duration': time.time() - start_time}

#     def validate(self):
#         """
#         Runs inference and calculates NIQE/BRISQUE.
#         Note: These metrics are computationally heavy compared to MSE.
#         """
#         if not METRICS_AVAILABLE:
#             return {}

#         self.model.eval()
#         total_niqe = 0
#         total_brisque = 0
#         count = 0
        
#         pbar = tqdm(self.val_loader, desc="Validating (NIQE/BRISQUE)", leave=False)

#         with torch.no_grad():
#             for batch_data in pbar:
#                 low_imgs = batch_data['images_low'].to(self.device)
                
#                 # Inference
#                 enhanced_img, _ = self.model(low_imgs)
                
#                 # Clamp to 0-1 for metrics
#                 enhanced_img = torch.clamp(enhanced_img, 0, 1)

#                 # Compute Metrics (Batch-wise if supported, or iterate)
#                 # pyiqa usually supports batch inputs
#                 niqe_score = self.metrics['niqe'](enhanced_img).mean().item()
#                 brisque_score = self.metrics['brisque'](enhanced_img).mean().item()

#                 total_niqe += niqe_score
#                 total_brisque += brisque_score
#                 count += 1
                
#                 pbar.set_postfix({
#                     'NIQE': f'{niqe_score:.2f}', 
#                     'BRISQUE': f'{brisque_score:.2f}'
#                 })

#         avg_niqe = total_niqe / count
#         avg_brisque = total_brisque / count
        
#         # Composite score (You can adjust weights)
#         # Both NIQE and BRISQUE: Lower is Better.
#         composite_score = (avg_niqe + avg_brisque) / 2.0

#         return {
#             'val_niqe': avg_niqe,
#             'val_brisque': avg_brisque,
#             'val_composite': composite_score
#         }

#     def fit(self, epochs):
#         print(f"🚀 Starting Unsupervised Training on {self.device} for {epochs} epochs")
#         print("   Goal: Minimize Loss + Minimize NIQE/BRISQUE")
        
#         for epoch in range(epochs):
#             # 1. Train
#             train_metrics = self._train_epoch(epoch, epochs)
            
#             # 2. Validate
#             val_metrics = self.validate()
            
#             # 3. Log
#             epoch_log = {'epoch': epoch, **train_metrics, **val_metrics}
#             self.history.append(epoch_log)
            
#             # 4. Save Logic (Lower Score is Better for NIQE/BRISQUE)
#             score = val_metrics.get('val_composite', float('inf'))
            
#             print(f"\n--- Epoch {epoch} Summary ---")
#             print(f"Train Loss: {train_metrics['train_loss']:.4f}")
#             if METRICS_AVAILABLE:
#                 print(f"Metrics: NIQE={val_metrics['val_niqe']:.3f} | BRISQUE={val_metrics['val_brisque']:.3f}")
            
#             if score < self.best_score:
#                 print(f"🏆 Best Score Improved ({self.best_score:.4f} -> {score:.4f}). Saving model.")
#                 self.best_score = score
#                 torch.save(self.model.state_dict(), f"best_{self.name}_unsupervised.pt")
            
#             print("-----------------------------\n")

#         # Final Save
#         torch.save(self.model.state_dict(), f"final_{self.name}_unsupervised.pt")
#         return self.history
    
    
class ModelTrainer:
    def __init__(self, 
                 model, 
                 train_loader, 
                 val_loader, 
                 epochs,
                 det_loss_fn, 
                 enhance_loss_fn,
                 experiment_name='mtl_yolo_llie',
                 device=None,
                 initial_lr=1e-2,
                 warmup_epochs=2):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.det_loss_fn = det_loss_fn
        self.enhance_loss_fn = enhance_loss_fn
        self.name = experiment_name
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.map_metric = MeanAveragePrecision(
            iou_thresholds=[0.5],
            class_metrics=False
        ).to(self.device)
        self.model.to(self.device)
        
        # --- Optimization Setup ---
        self.optimizer = self._setup_optimizer(initial_lr)
      
        self.device_type = self.device.type # Will be 'cuda' or 'cpu'
        if self.device_type == 'cuda':
            self.scaler = torch.amp.GradScaler(device=self.device_type)
        else:
            self.scaler = None
        
        # Scheduler setup
        # Note: We calculate total steps later in .fit() or assume a fixed epoch count now.
        # To be safe, we'll initialize the scheduler in .fit() where we know total_epochs.
        self.scheduler = None 
        self.warmup_epochs = warmup_epochs
        
        # --- Metrics & Saving ---
        self.saver = BestMAPCallback(model_name=self.name)
        self.history = []

    def _setup_optimizer(self, lr, decay=1e-10):
        """Internal helper to set task-specific learning rates."""
        backbone_params = []
        neck_head_params = []
        decoder_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(param)
            elif 'neck' in name or 'head' in name:
                neck_head_params.append(param)
            elif 'decoder' in name:
                decoder_params.append(param)

        param_groups = [
            {'params': backbone_params, 'lr': 1e-4, 'weight_decay': decay},
            {'params': neck_head_params, 'lr': 1e-4, 'weight_decay': decay},
            {'params': decoder_params, 'lr': 1e-3, 'weight_decay': decay}
        ]
        return AdamW(param_groups, lr=lr)

    def _train_epoch(self, epoch, total_epochs):
        self.model.train()
        start_time = time.time()
        
        total_epoch_loss = 0
        total_det_loss = 0
        total_enhance_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{total_epochs}", leave=True)

        for batch_idx, batch_data in enumerate(pbar):
            # if batch_idx == 2: 
            #     break
            # 1. Move data to device
            low_imgs = batch_data['images_low'].to(self.device)
            high_imgs = batch_data['images_high'].to(self.device)
            
            det_targets = {
                'cls': batch_data['cls'].to(self.device),
                'bboxes': batch_data['bboxes'].to(self.device),
                'batch_idx': batch_data['batch_idx'].to(self.device)
            }

            self.optimizer.zero_grad()

            # 2. Forward pass with AMP
            with torch.autocast(device_type=self.device.type, enabled=(self.device_type == 'cuda')):
                total_loss, det_loss, enhance_loss, _, _, _ = compute_total_loss(
                    self.model, low_imgs, high_imgs, det_targets, 
                    self.det_loss_fn, self.enhance_loss_fn
                )

            # 3. Backward & Step with Scaler
            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
            
            # 4. Step Scheduler
            if self.scheduler:
                self.scheduler.step()

            # 5. Logging
            total_epoch_loss += total_loss.item()
            total_det_loss += det_loss.item()
            total_enhance_loss += enhance_loss.item()

            pbar.set_postfix({
                'TLoss': f'{total_loss.item():.4f}',
                'DLoss': f'{det_loss.item():.4f}',
                'ELoss': f'{enhance_loss.item():.4f}',
                'LR_BB': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'LR_Det': f'{self.optimizer.param_groups[1]["lr"]:.2e}',
                'LR_Dec': f'{self.optimizer.param_groups[2]["lr"]:.2e}'
            })

        # Calculate averages
        n_batches = len(self.train_loader)
        return {
            'train_loss_total': total_epoch_loss / n_batches,
            'train_loss_det': total_det_loss / n_batches,
            'train_loss_enh': total_enhance_loss / n_batches,
            'epoch_duration': time.time() - start_time
        }
    def validate(self):
        self.model.eval()
        self.map_metric.reset()
        total_val_loss, total_det_loss, total_enh_loss = 0, 0, 0
        total_psnr, total_ssim = 0, 0
        total_ssim_count = 0

        pbar = tqdm(self.val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for idx, batch_data in enumerate(pbar):
                low_imgs = batch_data['images_low'].to(self.device)
                high_imgs = batch_data['images_high'].to(self.device)
                
                # Dynamically get the height and width of the padded batch
                B, _, H, W = low_imgs.shape 

                det_targets = {
                    'cls': batch_data['cls'].to(self.device),
                    'bboxes': batch_data['bboxes'].to(self.device),
                    'batch_idx': batch_data['batch_idx'].to(self.device)
                }

                with torch.autocast(device_type=self.device.type, enabled=(self.device_type == 'cuda')):
                    total_loss, det_loss, enhance_loss, _, llie_out, _ = compute_total_loss(
                        self.model, low_imgs, high_imgs, det_targets, 
                        self.det_loss_fn, self.enhance_loss_fn, 
                    )
                    
                    det_out, _, _ = self.model(low_imgs, inference=True)
                    
                    preds, targets = [], []

                    for b in range(B):
                        # -------- Predictions --------
                        p = det_out[b].permute(1, 0)  # [8400, 84]

                        boxes_xywh = p[:, :4]         # FIX 1: This is xywh, not xyxy!
                        cls_scores, labels = p[:, 4:].max(dim=1)

                        # Convert predicted xywh to xyxy mathematically 
                        pred_xy = boxes_xywh[:, :2]
                        pred_wh = boxes_xywh[:, 2:]
                        boxes_xyxy = torch.cat((pred_xy - pred_wh / 2, pred_xy + pred_wh / 2), dim=1)

                        # FIX 2: Lower threshold so torchmetrics can build a full PR curve
                        mask = cls_scores > 0.001
                        boxes, scores, labels = boxes_xyxy[mask], cls_scores[mask], labels[mask]

                        # Apply NMS only if boxes exist
                        if len(boxes) > 0:
                            keep = nms(boxes, scores, 0.5)
                            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

                        preds.append({
                            "boxes": boxes,
                            "scores": scores,
                            "labels": labels
                        })

                        # -------- Ground Truth --------
                        gt_mask = det_targets['batch_idx'] == b
                        gt_boxes_xywh = det_targets['bboxes'][gt_mask]   # normalized xywh
                        gt_labels = det_targets['cls'][gt_mask]

                        # FIX 3: Robust GT scaling using the actual batch tensor dimensions
                        if len(gt_boxes_xywh) > 0:
                            gt_xy = gt_boxes_xywh[:, :2]
                            gt_wh = gt_boxes_xywh[:, 2:]
                            gt_boxes_xyxy = torch.cat((gt_xy - gt_wh / 2, gt_xy + gt_wh / 2), dim=1)
                            
                            # Scale normalized [0-1] coordinates to absolute pixel dimensions
                            gt_boxes_xyxy[:, [0, 2]] *= W
                            gt_boxes_xyxy[:, [1, 3]] *= H
                        else:
                            gt_boxes_xyxy = torch.empty((0, 4), device=self.device)

                        targets.append({
                            "boxes": gt_boxes_xyxy,
                            "labels": gt_labels
                        })

                    # Update the torchmetrics object
                    self.map_metric.update(preds, targets)

                # Accumulate Losses
                total_val_loss += total_loss.item()
                total_det_loss += det_loss.item()
                total_enh_loss += enhance_loss.item()

                # Calculate PSNR
                total_psnr += psnr(llie_out, high_imgs).item()

                # Calculate SSIM (iterate batch)
                for i in range(B):
                    total_ssim += ssim_metric(llie_out[i], high_imgs[i])
                    total_ssim_count += 1
                
                pbar.set_postfix({'ValLoss': f'{total_loss.item():.4f}'})

        n_batches = len(self.val_loader)
        
        map_results = self.map_metric.compute()
        
        return {
            'val_loss_total': total_val_loss / n_batches,
            'val_loss_det': total_det_loss / n_batches,
            'val_loss_enh': total_enh_loss / n_batches,
            'val_psnr': total_psnr / n_batches,
            'val_ssim': total_ssim / total_ssim_count if total_ssim_count > 0 else 0,
            'val_prec_prox' : map_results['map'].item(), 
            'val_mAP50': map_results['map_50'].item(),
            'val_mAP': map_results['map'].item(),
            'val_recall': map_results['mar_100'].item(),
        }
        
    # def validate(self):
    #     self.model.eval()
    #     total_val_loss, total_det_loss, total_enh_loss = 0, 0, 0
    #     total_psnr, total_ssim = 0, 0
    #     total_ssim_count = 0
        
    #     # New manual metric accumulators
    #     total_precision, total_recall, total_map50 = 0.0, 0.0, 0.0
    #     total_images_eval = 0

    #     pbar = tqdm(self.val_loader, desc="Validation", leave=False)

    #     with torch.no_grad():
    #         for idx, batch_data in enumerate(pbar):
    #             low_imgs = batch_data['images_low'].to(self.device)
    #             high_imgs = batch_data['images_high'].to(self.device)
    #             B, _, H, W = low_imgs.shape 

    #             det_targets = {
    #                 'cls': batch_data['cls'].to(self.device),
    #                 'bboxes': batch_data['bboxes'].to(self.device),
    #                 'batch_idx': batch_data['batch_idx'].to(self.device)
    #             }

    #             with torch.autocast(device_type=self.device.type, enabled=(self.device_type == 'cuda')):
    #                 total_loss, det_loss, enhance_loss, _, llie_out, _ = compute_total_loss(
    #                     self.model, low_imgs, high_imgs, det_targets, 
    #                     self.det_loss_fn, self.enhance_loss_fn, 
    #                 )
                    
    #                 det_out, _, _ = self.model(low_imgs, inference=True)

    #                 for b in range(B):
    #                     # -------- Predictions --------
    #                     p = det_out[b].permute(1, 0)
    #                     boxes_xywh = p[:, :4]
    #                     cls_scores, labels = p[:, 4:].max(dim=1)

    #                     pred_xy = boxes_xywh[:, :2]
    #                     pred_wh = boxes_xywh[:, 2:]
    #                     boxes_xyxy = torch.cat((pred_xy - pred_wh / 2, pred_xy + pred_wh / 2), dim=1)

    #                     # Threshold & NMS
    #                     mask = cls_scores > 0.01 # Keep this low for the PR curve calculation
    #                     boxes, scores = boxes_xyxy[mask], cls_scores[mask]

    #                     if len(boxes) > 0:
    #                         keep = nms(boxes, scores, 0.5)
    #                         boxes = boxes[keep]

    #                     # -------- Ground Truth --------
    #                     gt_mask = det_targets['batch_idx'] == b
    #                     gt_boxes_xywh = det_targets['bboxes'][gt_mask]

    #                     if len(gt_boxes_xywh) > 0:
    #                         gt_xy = gt_boxes_xywh[:, :2]
    #                         gt_wh = gt_boxes_xywh[:, 2:]
    #                         gt_boxes_xyxy = torch.cat((gt_xy - gt_wh / 2, gt_xy + gt_wh / 2), dim=1)
                            
    #                         gt_boxes_xyxy[:, [0, 2]] *= W
    #                         gt_boxes_xyxy[:, [1, 3]] *= H
    #                     else:
    #                         gt_boxes_xyxy = torch.empty((0, 4), device=self.device)

    #                     # -------- Compute Manual Metrics --------
    #                     prec, rec = compute_pr_loop(boxes, gt_boxes_xyxy, iou_thresh=0.5)
    #                     total_precision += prec
    #                     total_recall += rec
    #                     total_map50 += (prec * rec) # Your proxy calculation
    #                     total_images_eval += 1

    #             # Accumulate Losses
    #             total_val_loss += total_loss.item()
    #             total_det_loss += det_loss.item()
    #             total_enh_loss += enhance_loss.item()

    #             # Calculate PSNR/SSIM
    #             total_psnr += psnr(llie_out, high_imgs).item()
    #             for i in range(B):
    #                 total_ssim += ssim_metric(llie_out[i], high_imgs[i])
    #                 total_ssim_count += 1
                
    #             pbar.set_postfix({'ValLoss': f'{total_loss.item():.4f}'})

    #     n_batches = len(self.val_loader)
        
    #     # Average out the custom metrics
    #     avg_prec = total_precision / total_images_eval if total_images_eval > 0 else 0.0
    #     avg_rec = total_recall / total_images_eval if total_images_eval > 0 else 0.0
    #     avg_map50 = total_map50 / total_images_eval if total_images_eval > 0 else 200
        
    #     return {
    #         'val_loss_total': total_val_loss / n_batches,
    #         'val_loss_det': total_det_loss / n_batches,
    #         'val_loss_enh': total_enh_loss / n_batches,
    #         'val_psnr': total_psnr / n_batches,
    #         'val_ssim': total_ssim / total_ssim_count if total_ssim_count > 0 else 0,
    #         'val_prec_prox': avg_prec, 
    #         'val_recall': avg_rec,
    #         'val_mAP50': avg_map50,
    #     }
        
    def fit(self):
        epochs = self.epochs
        print(f"Starting training on {self.device} for {epochs} epochs...")
        
        # Initialize Scheduler now that we know total epochs
        total_steps = epochs * len(self.train_loader)
        warmup_steps = int(self.warmup_epochs * len(self.train_loader))
        # self.scheduler = CosineAnnealingWarmupLR(self.optimizer, total_steps, warmup_steps)
        
        max_lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=max_lrs, 
            total_steps=total_steps,
            pct_start=0.1,        # Spends 10% of training warming up, 90% cooling down
            anneal_strategy='cos', # Smooth cosine curve
            div_factor=25.0,      # Initial LR starts at max_lr / 25
            final_div_factor=1e4  # Final LR ends at Initial LR / 10000
        )

        for epoch in range(epochs):
            # 1. Train
            train_metrics = self._train_epoch(epoch, epochs)
            
            # 2. Validate
            val_metrics = self.validate()
            
            # 3. Merge Metrics
            epoch_log = {
                'epoch': epoch,
                **train_metrics,
                **val_metrics,
                'lr_backbone': self.optimizer.param_groups[0]['lr'],
                'lr_det': self.optimizer.param_groups[1]['lr'],
                'lr_dec': self.optimizer.param_groups[2]['lr']
            }
            self.history.append(epoch_log)

            # 4. Checkpoint
            self.saver.check_and_save(self.model, epoch_log, epoch)

            # 5. Print Summary
            print(f"\n--- Epoch {epoch} Summary ({train_metrics['epoch_duration']:.2f}s) ---")
            print(f"Train Loss: {train_metrics['train_loss_total']:.4f} | Val Loss: {val_metrics['val_loss_total']:.4f}")
            print(f"Enhancement: PSNR={val_metrics['val_psnr']:.2f} | SSIM={val_metrics['val_ssim']:.3f}")
            print(f"Detection: mAP@0.5={val_metrics['val_mAP50']:.3f}")
            print("-------------------------------------------\n")

        # Final Save
        final_path = f'final_{self.name}_epoch{epochs}.pt'
        torch.save(self.model.state_dict(), final_path)
        print(f"Training Complete. Final model saved to {final_path}")
        return self.history
    
