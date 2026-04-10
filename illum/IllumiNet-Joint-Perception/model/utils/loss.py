import torch
import torch.nn as nn

from model.modules import make_anchors, dist2bbox, bbox2dist, xywh2xyxy
from .metrics import bbox_iou, df_loss
from .tal import TaskAlignedAssigner

from typing import Dict, Tuple


# TODO 
# Since the new LLIE Decoder removes the last residual addition, 
# we need to implement the last residual addition. 
# ex: 
# loss(in_img + res_map, gt_img)

class BaseLoss:
    def __init__(self, device:str):
        self.device = device

    def compute_loss(self, batch:torch.Tensor, preds:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BboxLoss(BaseLoss):
    def __init__(self, reg_max:int, device:str, use_dfl:bool=False):
        super().__init__(device)

        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def compute_loss(
        self,
        pred_box_dist:torch.Tensor,
        pred_boxes:torch.Tensor,
        target_boxes:torch.Tensor,
        anchor_points:torch.Tensor,
        target_scores:torch.Tensor,
        mask:torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(dim=-1)[mask].unsqueeze(dim=-1)
        iou = bbox_iou(pred_boxes[mask], target_boxes[mask], xywh=False)

        target_scores_sum = max(target_scores.sum(), 1)
        iou_loss = ((1 - iou) * weight).sum() / target_scores_sum

        if self.use_dfl:
            gt_ltrb = bbox2dist(target_boxes, anchor_points, self.reg_max)
            dfl_loss = df_loss(pred_box_dist[mask].view(-1, self.reg_max+1), gt_ltrb[mask]) * weight
            dfl_loss = dfl_loss.sum() / target_scores_sum
        else:
            dfl_loss = torch.tensor(0.0).to(self.device)

        return iou_loss, dfl_loss


from model.modules.head import DetectionHead

class DetectionLoss(BaseLoss):
    def __init__(self, model, device:str):
        super().__init__(device)
        
        # detect_head = model.model[-1]
        detect_head = None
        # Check if 'model' is the wrapper class or the inner nn.Sequential
        model_list = model.model if hasattr(model, 'model') else model
        
        for m in model_list:
            if isinstance(m, DetectionHead):
                detect_head = m
                break
        
        if detect_head is None:
            raise AttributeError("DetectionLoss could not find a DetectionHead in the model.")

        self.nc = detect_head.nc
        self.n_outputs = detect_head.n_outputs
        self.reg_max = detect_head.reg_max
        self.stride = detect_head.stride

        self.loss_gains = model.loss_gains

        # Projects predicted boxes to different scales
        self.proj = torch.arange(self.reg_max, device=self.device, dtype=torch.float)

        # Sigmoid + Binary Cross Entropy Loss (-log(sigmoid(x)))
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

        self.bbox_loss = BboxLoss(self.reg_max-1, self.device, use_dfl=True)

        self.tal_assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0).to(self.device)

    def preprocess(self, targets:torch.Tensor, batch_size:int, scale_tensor:torch.Tensor) -> torch.Tensor:
        """
        Preprocesses target boxes to match predicted boxes batch size
        """
        # No bboxes in image
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device)
        
        im_idxs = targets[:,0]
        _, counts = im_idxs.unique(return_counts=True)

        # Row idxs correspond to predicted idxs, col idxs correspond to targets
        out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
        for i in range(batch_size):
            mask = (im_idxs == i)
            n_matches = mask.sum()
            if n_matches > 0:
                # Add cls and bbox targets to output at matching indices
                out[i, :n_matches] = targets[mask, 1:]

        # Convert boxes from xywh to xyxy
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def decode_bbox(self, anchor_points:torch.Tensor, pred_box_dist:torch.Tensor):
        """
        Decodes bounding box coordinates from anchor points and predicted
        box distribution. Returns bounding box coordinates in (xyxy) format.
        """
        b, a, c = pred_box_dist.shape  # (batch, anchors, channels)
        # Reshape to (batch, anchors, 4, reg_max) then softmax
        # along reg_max dim and mul by (reg_max,) -> (b,a,4)
        pred_boxes = pred_box_dist.view(b, a, 4, c//4).softmax(dim=3) @ self.proj
        return dist2bbox(pred_boxes, anchor_points, xywh=False)

    def compute_loss(self, batch:Dict[str,torch.Tensor], preds:torch.Tensor):
        pred_box_dist, pred_cls = torch.cat(
            [xi.view(preds[0].shape[0], self.n_outputs, -1) for xi in preds], dim=2
        ).split((4*self.reg_max, self.nc), dim=1)

        pred_cls = pred_cls.permute(0, 2, 1).contiguous()
        pred_box_dist = pred_box_dist.permute(0, 2, 1).contiguous()

        batch_size = pred_box_dist.shape[0]
        im_size = torch.tensor(preds[0].shape[2:], device=self.device) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(preds, self.stride)
        pred_boxes = self.decode_bbox(anchor_points, pred_box_dist)  # (b, h*w, 4) in (xyxy)

        # (batch_idx, cls, xywh box)
        targets = torch.cat((batch['batch_idx'].view(-1,1), batch['cls'].view(-1,1), batch['bboxes']), dim=1).to(self.device)
        targets = self.preprocess(targets, batch_size, scale_tensor=im_size[[1,0,1,0]])
        # cls, xyxy box
        gt_cls, gt_boxes = targets.split((1,4), dim=2)
        gt_mask = gt_boxes.sum(dim=2, keepdim=True) > 0  # mask to filter out (0,0,0,0) boxes (just used to pad tensor)

        _, target_boxes, target_scores, mask = self.tal_assigner(
            pred_cls.detach().sigmoid(), pred_boxes.detach() * stride_tensor, anchor_points * stride_tensor, gt_cls, gt_boxes, gt_mask
        )
        
        cls_loss = self.bce_loss(pred_cls, target_scores).sum() / max(target_scores.sum(), 1)

        if mask.sum() > 0:
            iou_loss, dfl_loss = self.bbox_loss.compute_loss(
                pred_box_dist, pred_boxes, target_boxes/stride_tensor, anchor_points, target_scores, mask
            )
        else:
            iou_loss = torch.tensor(0.0).to(self.device)
            dfl_loss = torch.tensor(0.0).to(self.device)

        loss = self.loss_gains['cls']*cls_loss + self.loss_gains['iou']*iou_loss + self.loss_gains['dfl']*dfl_loss

        loss_items = torch.stack([cls_loss, iou_loss, dfl_loss]).detach()
        
        return loss * batch_size, loss_items


# utils/loss.py

class MTLLoss(BaseLoss):
    def __init__(self, model, device, balance=[0.6, 0.4]):
        super().__init__(device)
        self.balance = balance  # [Detection Weight, LLIE Weight]
        
        # 1. Initialize the Standard Detection Loss
        # We wrap the existing class
        self.det_loss_fn = DetectionLoss(model, device)
        
        # 2. Initialize LLIE Loss (MAE / L1)
        self.llie_loss_fn = nn.L1Loss()

    def compute_loss(self, batch, preds):
        """
        preds: Tuple(det_preds, llie_res_map)
               det_preds -> List of 3 tensors from YOLO head
               llie_res_map -> Tensor (B, 3, H, W) from LLIE Decoder
        """
        # 1. Unpack the tuple from our modified model
        det_preds, llie_res_map = preds

        # ----------------------------------------------------
        # TASK A: Object Detection Loss
        # ----------------------------------------------------
        # This calls the DetectionLoss.compute_loss we just modified
        det_loss, det_items = self.det_loss_fn.compute_loss(batch, det_preds)

        # ----------------------------------------------------
        # TASK B: LLIE Loss (With Residual Addition)
        # ----------------------------------------------------
        # We need the Input Image (Low Light) and Target Image (Normal Light)
        # Ensure they are on the correct device
        img = batch['img'].to(self.device)   # Input (Low Light)
        gt_img = batch['gt_img'].to(self.device) # Target (Normal Light)

        # [CRITICAL STEP] Reconstruct the enhanced image here
        # Enhanced = Input + Predicted_Residual
        enhanced_img = img + llie_res_map
        
        # Compute MAE Loss between Enhanced and Ground Truth
        llie_loss = self.llie_loss_fn(enhanced_img, gt_img)

        # Scale LLIE loss by batch size to match detection loss scale logic 
        # (Since DetectionLoss returns loss * batch_size)
        llie_loss = llie_loss * batch['batch_idx'].max() + 1 # Approximate batch size scaling
        # Or simpler: just keep it mean and let the weights handle it. 
        # Let's stick to the user's weighted sum logic directly:

        # ----------------------------------------------------
        # COMBINE LOSSES
        # ----------------------------------------------------
        # Total = 0.6 * Det + 0.4 * LLIE
        total_loss = (self.balance[0] * det_loss) + (self.balance[1] * llie_loss)

        # ----------------------------------------------------
        # LOGGING
        # ----------------------------------------------------
        # We concatenate the detection items (box, cls, dfl) with the llie loss
        # Result vector: [box_loss, cls_loss, dfl_loss, llie_loss]
        loss_items = torch.cat([det_items, llie_loss.unsqueeze(0).detach()])

        return total_loss, loss_items