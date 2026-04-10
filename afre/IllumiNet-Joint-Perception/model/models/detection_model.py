import yaml
import torch

from .base_model import BaseModel, BaseModel_MTL
from model.modules import DetectionHead
from model.utils.loss import DetectionLoss, MTLLoss

from model.misc import parse_config
from model.modules import init_weights
from model.utils.ops import nms


class DetectionModel(BaseModel):
    def __init__(self, config:str, verbose:bool=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        config = config if isinstance(config, dict) else yaml.safe_load(open(config, 'r'))
        in_channels = config.get('in_channels', 3)

        self.model, self.save_idxs = parse_config(config, verbose=verbose)
        self.model.to(self.device)

        self.inplace = config.get('inplace', True)

        detect_head = self.model[-1]
        # Calculate stride for detection head
        if isinstance(detect_head, DetectionHead):
            detect_head.inplace = True
            s = 256
            detect_head.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, in_channels, s, s, device=self.device))])
            self.stride = detect_head.stride
            detect_head._bias_init()

        # Initialize weights
        init_weights(self)

        # Initialize loss function
        self.loss_gains = config.get('loss_gains', None)
        self.loss_fn = DetectionLoss(self, self.device)

    def postprocess(self, preds: torch.Tensor):
        return nms(preds)
    

# In detection_model.py
# class DetectionModel_MTL(BaseModel_MTL):
#     def __init__(self, config: str, verbose: bool = True, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         config = config if isinstance(config, dict) else yaml.safe_load(open(config, 'r'))
#         in_channels = config.get('in_channels', 3)

#         self.model, self.save_idxs = parse_config(config, verbose=verbose)
#         self.model.to(self.device)

#         self.inplace = config.get('inplace', True)

#         # Find DetectionHead
#         detect_head = None
#         for m in self.model:
#             if isinstance(m, DetectionHead):
#                 detect_head = m
#                 break
        
#         if detect_head is None:
#             raise ValueError("No DetectionHead found in model config!")

#         # --- FIX 1: Correct Stride Calculation (Bypass postprocess) ---
#         if isinstance(detect_head, DetectionHead):
#             detect_head.inplace = True
#             s = 256
            
#             # Run dummy forward pass
#             dummy_input = torch.zeros(1, in_channels, s, s, device=self.device)
            
#             # CRITICAL FIX: Use _predict instead of forward
#             # This avoids calling 'postprocess' on the raw list of tensors
#             preds = self._predict(dummy_input)
            
#             det_out = preds[0] 
            
#             # Calculate stride
#             detect_head.stride = torch.tensor([s / x.shape[-2] for x in det_out])
#             self.stride = detect_head.stride
#             detect_head._bias_init()

#         # Initialize weights
#         init_weights(self)

#         self.loss_gains = config.get('loss_gains', None)
#         self.loss_fn = MTLLoss(self, self.device, balance=[0.6, 0.4])

#     def postprocess(self, preds: torch.Tensor):
#         """
#         Handles decoding of raw YOLOv8 output, fixes negative boxes, and applies NMS.
#         """
#         # 1. Handle Tuple Input
#         if isinstance(preds, tuple):
#             preds = preds[0]

#         # 2. Ensure Format is (Batch, 8400, 84)
#         if hasattr(preds, 'shape') and preds.shape[1] != 8400: 
#              preds = preds.transpose(1, 2)
        
#         # 3. Separate Boxes and Classes
#         box_preds = preds[..., :4].clone()
#         cls_preds = preds[..., 4:]

#         # --- FIX 2: Decode Raw/Negative Boxes ---
#         # If boxes are negative (raw regression logits), force them positive and scale
#         if box_preds[..., 2:4].min() < 0:
#             device = box_preds.device
#             # Create strides aligned with 8400 anchors (6400 P3 + 1600 P4 + 400 P5)
#             stride_8 = torch.full((6400, 1), 8.0, device=device)
#             stride_16 = torch.full((1600, 1), 16.0, device=device)
#             stride_32 = torch.full((400, 1), 32.0, device=device)
#             strides = torch.cat([stride_8, stride_16, stride_32], dim=0)
            
#             # Decode: ABS(width) * Stride
#             box_preds[..., 2:4] = torch.abs(box_preds[..., 2:4]) * strides
            
#             # If x,y are normalized (0-1), scale them up (heuristic fallback)
#             # Assuming 640x640 inference if mostly normalized
#             if box_preds[..., 0].max() < 2.0:
#                  box_preds[..., :4] *= 640.0 
        
#         # 4. Convert xywh -> xyxy
#         x, y, w, h = box_preds[..., 0], box_preds[..., 1], box_preds[..., 2], box_preds[..., 3]
#         x1 = x - w / 2
#         y1 = y - h / 2
#         x2 = x + w / 2
#         y2 = y + h / 2
#         decoded_boxes = torch.stack([x1, y1, x2, y2], dim=-1)

#         # 5. Prepare for NMS
#         # Calculate scores to filter before sending to NMS (optimization)
#         scores, class_ids = torch.max(cls_preds, dim=-1)
        
#         # Quick filter for empty images
#         if not (scores > 0.05).any():
#             return [] # Return empty list if no detections
            
#         final_preds = torch.cat([decoded_boxes, scores.unsqueeze(-1), class_ids.float().unsqueeze(-1)], dim=-1)
        
#         # 6. Apply NMS
#         return nms(final_preds)


class DetectionModel_MTL(BaseModel_MTL):
    def __init__(self, config: str, verbose: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        config = config if isinstance(config, dict) else yaml.safe_load(open(config, 'r'))
        in_channels = config.get('in_channels', 3)

        self.model, self.save_idxs = parse_config(config, verbose=verbose)
        self.model.to(self.device)

        self.inplace = config.get('inplace', True)
        
        self.conf_threshold = 0.5

        # --- FIX 1: Find the DetectionHead dynamically ---
        # Don't assume it's at [-1]. Iterate and find it.
        detect_head = None
        for m in self.model:
            if isinstance(m, DetectionHead):
                detect_head = m
                break
        
        if detect_head is None:
            raise ValueError("No DetectionHead found in model config!")

        # --- FIX 2: Correct Stride Calculation ---
        if isinstance(detect_head, DetectionHead):
            detect_head.inplace = True
            s = 256
            
            # Run dummy forward pass
            dummy_input = torch.zeros(1, in_channels, s, s, device=self.device)
            preds = self._predict(dummy_input)
            
            # Unpack the tuple to get only the detection output
            # (det_out, llie_out) = preds
            det_out = preds[0] 
            
            # Now calculate stride using det_out
            detect_head.stride = torch.tensor([s / x.shape[-2] for x in det_out])
            self.stride = detect_head.stride
            detect_head._bias_init()

        # Initialize weights
        init_weights(self)

        self.loss_gains = config.get('loss_gains', None)
        # Note: We will need to update DetectionLoss later to handle the tuple input too!
        # self.loss_fn = DetectionLoss(self, self.device)
        self.loss_fn = MTLLoss(self, self.device, balance=[0.6, 0.4])

    # def postprocess(self, preds: torch.Tensor):
    #     """
    #     Handles decoding of raw YOLOv8 output and applies NMS.
    #     Input preds: (batch, 4 + nc, 8400) or (batch, 8400, 4 + nc)
    #     """
    #     # 1. Handle Tuple Input (if coming from predict wrapper)
    #     if isinstance(preds, tuple):
    #         preds = preds[0] # Get just the detection tensor

    #     # 2. Ensure Format is (Batch, 8400, 84)
    #     if preds.shape[1] != 8400: 
    #          preds = preds.transpose(1, 2)
        
    #     # 3. Separate Boxes and Classes
    #     box_preds = preds[..., :4].clone()
    #     cls_preds = preds[..., 4:]

    #     # 2. Get Max Score (Filter noise early)
    #     scores, class_ids = torch.max(cls_preds, dim=-1)
    #     mask = scores > self.conf_threshold
        
    #     box_preds = box_preds[mask]
    #     scores = scores[mask]
    #     class_ids = class_ids[mask]
        
    #     if len(box_preds) > 0:
    #         # 3. ROBUST SCALING LOGIC
    #         # Check 1: Are x,y normalized (0-1)? -> Scale by Image Size
    #         if box_preds[:, 0:2].max() <= 1.5:
    #             box_preds[:, :4] *= 640.0
            
    #         # Check 2: Are w,h tiny (Grid space)? -> Scale by Stride
    #         # (We estimate stride based on box location simply for recovery)
    #         # But safer strategy: if width is suspiciously small (< 20px) but x is large (>100px)
    #         elif box_preds[:, 2].mean() < 20.0 and box_preds[:, 0].mean() > 50.0:
    #             # Heuristic: Multiply w,h by 16 (Average stride) to inflate them
    #             # Or better: just fix the width to a known anchor size? No, that's bad.
    #             # LET'S USE THE STRIDE TENSOR CORRECTLY:
                
    #             # We need to map the masked boxes BACK to their original index to find the stride
    #             # This is hard after masking.
    #             pass 

    #             # --- SIMPLIFIED FIX FOR YOU ---
    #             # Since you are debugging, let's just use the robust post-process
    #             # FROM THE MODEL CLASS we wrote earlier. 
    #             # Delete your manual pre-processing in this script and call model.postprocess()
                
    #     # 4. DECODING FIX (The "Negative Box" Fix)
    #     # Check if boxes are raw/negative (like we saw in debug: -9.23)
    #     # if box_preds[..., 2:4].min() < 0:
    #     #     # Create strides aligned with 8400 anchors
    #     #     device = box_preds.device
    #     #     stride_8 = torch.full((6400, 1), 8.0, device=device)
    #     #     stride_16 = torch.full((1600, 1), 16.0, device=device)
    #     #     stride_32 = torch.full((400, 1), 32.0, device=device)
    #     #     strides = torch.cat([stride_8, stride_16, stride_32], dim=0)
            
    #     #     # Apply Absolute Value or Exp to fix negatives, and scale by stride
    #     #     # using abs() is safer for raw regression outputs that drifted negative
    #     #     box_preds[..., 2:4] = torch.abs(box_preds[..., 2:4]) * strides
            
    #     #     # Ensure x,y are also scaled if they are small (normalized)
    #     #     # (Based on your logs, x,y were already pixels, so this might not be needed, 
    #     #     #  but checking doesn't hurt)
    #     #     if box_preds[..., 0].max() < 2.0:
    #     #          box_preds[..., :4] *= 640.0 # Or use img size
        
    #     # 5. Convert xywh -> xyxy (NMS expects xyxy)
    #     # We can implement a quick helper or import one
    #     x, y, w, h = box_preds[..., 0], box_preds[..., 1], box_preds[..., 2], box_preds[..., 3]
    #     x1 = x - w / 2
    #     y1 = y - h / 2
    #     x2 = x + w / 2
    #     y2 = y + h / 2
    #     decoded_boxes = torch.stack([x1, y1, x2, y2], dim=-1)

    #     # 6. Re-assemble for NMS
    #     # The 'nms' function likely expects [x1, y1, x2, y2, conf, cls] or similar
    #     # Since we don't see your 'nms' code, let's assume it handles the standard YOLO shape
    #     # If your 'nms' function is simple, we might need to compute scores here:
        
    #     scores, class_ids = torch.max(cls_preds, dim=-1)
        
    #     # Filter before NMS to save speed
    #     mask = scores > 0.05
    #     if not mask.any():
    #         return []
            
    #     final_preds = torch.cat([decoded_boxes, scores.unsqueeze(-1), class_ids.float().unsqueeze(-1)], dim=-1)
        
    #     # Now call your original NMS
    #     return nms(final_preds)
    
    def postprocess(self, preds: torch.Tensor):
        """
        Handles decoding of raw YOLOv8 output and applies NMS.
        Input preds: (batch, 4 + nc, 8400) or (batch, 8400, 4 + nc)
        """
        # 1. Handle Tuple Input
        if isinstance(preds, tuple):
            preds = preds[0]

        # 2. Ensure Format is (Batch, 8400, 84)
        if preds.shape[1] != 8400: 
             preds = preds.transpose(1, 2)
        
        # 3. Separate Boxes and Classes
        box_preds = preds[..., :4].clone() # [B, 8400, 4]
        cls_preds = preds[..., 4:]         # [B, 8400, 80]

        # --- THE FIX IS HERE ---
        # We need max score PER ANCHOR (dim=-1), not per class (dim=1)
        scores, class_ids = torch.max(cls_preds, dim=-1) # Output shape: [B, 8400]
        
        print(f"DEBUG: Max Confidence Score in Batch: {scores.max().item():.4f}")
        
        # 4. Filter by Confidence (Vectorized)
        mask = scores > 0.0001 # Use your threshold
        
        # If nothing detected, return empty
        if not mask.any():
            return []

        # Apply mask
        # Note: masking flattens the batch dim if batch > 1, but for inference B=1 it's fine
        box_preds = box_preds[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        if len(box_preds) == 0:
            return []

        # 5. ROBUST SCALING (Mixed Space Fix)
        # Check: Are w,h in Grid Space? (Small values)
        # Standard YOLOv8 grid vals are usually < 100, while x,y in pixels are > 100
        if box_preds[:, 2:4].max() < 100:
             # Re-create strides just for the remaining boxes? 
             # No, it's hard to match indices after masking.
             # BETTER STRATEGY: Scale BEFORE masking or use a global stride map.
             
             # Since we already masked, let's use the Heuristic fallback which is safe:
             # "If width is small (<50) but center_x is large (>50), it's definitely unscaled."
             pass

        # RE-DOING SCALING BEFORE MASKING (Safer & Cleaner)
        # Let's rewind the mask step for a micro-second to apply scaling to the full tensor
        # (This is more robust than trying to guess strides on masked data)
        
        # --- RESTARTING LOGIC FOR SAFETY ---
        box_preds_full = preds[..., :4].clone()
        
        # Create Strides (1, 8400, 1)
        device = preds.device
        stride_8 = torch.full((6400, 1), 8.0, device=device)
        stride_16 = torch.full((1600, 1), 16.0, device=device)
        stride_32 = torch.full((400, 1), 32.0, device=device)
        strides = torch.cat([stride_8, stride_16, stride_32], dim=0).unsqueeze(0) # [1, 8400, 1]

        # Apply Scaling to W/H only
        # We use ABS() to fix negative preds, and multiply by stride
        box_preds_full[..., 2:4] = torch.abs(box_preds_full[..., 2:4]) * strides
        
        # Now apply the mask to the SCALED boxes
        box_preds = box_preds_full[mask]
        
        # 6. Convert xywh -> xyxy
        x, y, w, h = box_preds[:, 0], box_preds[:, 1], box_preds[:, 2], box_preds[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        decoded_boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        # 7. Clip to Image Size (assuming 640x640)
        decoded_boxes.clamp_(0, 640)

        # 8. NMS
        final_preds = torch.cat([decoded_boxes, scores.unsqueeze(-1), class_ids.float().unsqueeze(-1)], dim=-1)
        
        return nms(final_preds.unsqueeze(0))