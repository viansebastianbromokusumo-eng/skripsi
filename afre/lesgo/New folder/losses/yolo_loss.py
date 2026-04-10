import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils.yolo_helpers import bbox2dist, make_anchors, dist2bbox, xywh2xyxy_tensor, xywh2xyxy


def bbox_iou(box1:torch.Tensor, box2:torch.Tensor, xywh:bool=True, eps:float=1e-10):
    """
    Calculate IoU between two bounding boxes

    Args:
        box1: (Tensor) with shape (..., 1 or n, 4)
        box2: (Tensor) with shape (..., n, 4)
        xywh: (bool) True if box coordinates are in (xywh) else (xyxy)

    Returns:
        iou: (Tensor) with IoU
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, dim=-1), box2.chunk(4, dim=-1)
        b1_x1, b1_y1, b1_x2, b1_y2 = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
        b2_x1, b2_y1, b2_x2, b2_y2 = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2
    else:
        (b1_x1, b1_y1, b1_x2, b1_y2), (b2_x1, b2_y1, b2_x2, b2_y2) = box1.chunk(4, dim=-1), box2.chunk(4, dim=-1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    intersection = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(min=0) * \
                   (torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)).clamp(min=0)
    
    union = w1 * h1 + w2 * h2 - intersection + eps

    iou = intersection / union

    return iou

def df_loss(pred_box_dist:torch.Tensor, targets:torch.Tensor):
    """
    Sum of left and right DFL losses
    """
    target_left = targets.long()
    target_right = target_left + 1
    weight_left = target_right - targets
    weight_right = 1 - weight_left

    dfl_left = F.cross_entropy(pred_box_dist, target_left.view(-1), reduction='none').view(target_left.shape) * weight_left
    dfl_right = F.cross_entropy(pred_box_dist, target_right.view(-1), reduction='none').view(target_right.shape) * weight_right

    return torch.mean(dfl_left + dfl_right, dim=-1, keepdim=True)

def nms(preds:torch.Tensor, confidence_thresh:float=0.25, iou_thresh:float=0.45) -> list[torch.Tensor]:
    """
    Non-Maximum Suppression for predicted boxes and classes

    Args:
        preds (Tensor): Predictions from model of shape (bs, 4 + num_classes, num_boxes)
        confidence_thresh (float, optional): Confidence threshold. Defaults to 0.25
        iou_thresh (float, optional): IoU threshold. Defaults to 0.45

    Returns:
        list[Tensor]: list of tensors of shape (num_boxes, 6) containing boxes with
            (x1, y1, x2, y2, confidence, class)
    """
    b, nc, _ = preds.shape
    nc -= 4

    # max confidence score among boxes
    xc = preds[:,4:].amax(dim=1) > confidence_thresh

    # (b, 4+nc, a) -> (b, a, 4+nc)
    preds = preds.transpose(-1, -2)

    preds[..., :4] = xywh2xyxy(preds[..., :4])

    out = [torch.zeros((0,6), device=preds.device)] * b

    for i, x in enumerate(preds):
        # take max cls confidence score
        # only consider predictions with confidence > confidence_thresh
        x = x[xc[i]]

        # If there are no remaining predictions, move to next image
        if not x.shape[0]:
            continue

        box, cls = x.split((4, nc), dim=1)

        confidences, cls_idxs = cls.max(dim=1, keepdim=True)
        x = torch.cat((box, confidences, cls_idxs.float()), dim=1)

        keep_idxs = torchvision.ops.nms(x[:,:4], x[:,4], iou_thresh)

        out[i] = x[keep_idxs]

    return out

def anchors_in_gt_boxes(anchor_points:torch.Tensor, gt_boxes:torch.Tensor, eps:float=1e-8):
    """
    Returns mask for positive anchor centers that are in GT boxes

    Args:
        anchor_points (Tensor): Anchor points of shape (n_anchors, 2)
        gt_boxes (Tensor): GT boxes of shape (batch_size, n_boxes, 4)
        
    Returns:
        mask (Tensor): Mask of shape (batch_size, n_boxes, n_anchors)
    """
    n_anchors = anchor_points.shape[0]
    batch_size, n_boxes, _ = gt_boxes.shape
    lt, rb = gt_boxes.view(-1, 1, 4).chunk(chunks=2, dim=2)
    box_deltas = torch.cat((anchor_points.unsqueeze(0) - lt, rb - anchor_points.unsqueeze(0)), dim=2).view(batch_size, n_boxes, n_anchors, -1)
    return (torch.amin(box_deltas, dim=3) > eps).to(anchor_points.device)

def select_highest_iou(mask:torch.Tensor, ious:torch.Tensor, num_max_boxes:int):
    """
    Select GT box with highest IoU for each anchor

    Args:
        mask (Tensor): Mask of shape (batch_size, num_max_boxes, n_anchors)
        ious (Tensor): IoU of shape (batch_size, num_max_boxes, n_anchors)

    Returns:
        target_gt_box_idxs (Tensor): Index of GT box with highest IoU for each anchor of shape (batch_size, n_anchors)
        fg_mask (Tensor): Mask of shape (batch_size, n_anchors) where 1 indicates positive anchor
        mask (Tensor): Mask of shape (batch_size, num_max_boxes, n_anchors) where 1 indicates positive anchor
    """
    # sum over n_max_boxes dim to get num GT boxes assigned to each anchor
    # (batch_size, num_max_boxes, n_anchors) -> (batch_size, n_anchors)
    fg_mask = mask.sum(dim=1)

    if fg_mask.max() > 1:
        # If 1 anchor assigned to more than one GT box, select the one with highest IoU
        max_iou_idx = ious.argmax(dim=1)  # (batch_size, n_anchors)

        # mask for where there are more than one GT box assigned to anchor
        multi_gt_mask = (fg_mask.unsqueeze(1) > 1).expand(-1, num_max_boxes, -1)  # (batch_size, num_max_boxes, n_anchors)

        # mask for GT box with highest IoU
        max_iou_mask = torch.zeros_like(mask, dtype=torch.bool)
        max_iou_mask.scatter_(dim=1, index=max_iou_idx.unsqueeze(1), value=1)

        mask = torch.where(multi_gt_mask, max_iou_mask, mask)
        fg_mask = mask.sum(dim=1)

    target_gt_box_idxs = mask.argmax(dim=1)  # (batch_size, n_anchors)
    return target_gt_box_idxs, fg_mask, mask


class TaskAlignedAssigner(nn.Module):
    """
    Task-aligned assigner for object detection
    """
    def __init__(self, topk:int=10, num_classes:int=80, alpha:float=1.0, beta:float=6.0, eps:float=1e-8, device:str='cuda'):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.device = device # Use the device passed during initialization

        self.bg_idx = num_classes  # no object (background)

    @torch.no_grad()
    def forward(
        self,
        pred_scores:torch.Tensor,
        pred_boxes:torch.Tensor,
        anchor_points:torch.Tensor,
        gt_labels:torch.Tensor,
        gt_boxes:torch.Tensor,
        gt_mask:torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assignment works in 4 steps:
        1. Compute alignment metric between all predicted bboxes (at all scales) and GT
        2. Select top-k bbox as candidates for each GT
        3. Limit positive sample's center in GT (anchor-free detector can only predict positive distances)
        4. If anchor box is assigned to multiple GT, select the one with highest IoU
        """
        num_max_boxes = gt_boxes.shape[1]

        # If there are no GT boxes, all boxes are background
        if num_max_boxes == 0:
            # Create all-background tensors directly on self.device
            return (torch.full_like(pred_scores[..., 0], self.bg_idx, device=self.device),
                    torch.zeros_like(pred_boxes, device=self.device),
                    torch.zeros_like(pred_scores, device=self.device))

        mask, align_metrics, ious = self.get_positive_mask(
            pred_scores, pred_boxes, anchor_points, gt_labels, gt_boxes, gt_mask
        )

        # Select GT box with highest IoU for each anchor
        target_gt_box_idxs, fg_mask, mask = select_highest_iou(mask, ious, num_max_boxes)

        target_labels, target_boxes, target_scores = self.get_targets(gt_labels, gt_boxes, target_gt_box_idxs, fg_mask)

        # Normalize
        align_metrics *= mask
        positive_align_metrics = align_metrics.amax(dim=-1, keepdim=True)  # (batch_size, num_max_boxes)
        positive_ious = (ious * mask).amax(dim=-1, keepdim=True)  # (batch_size, num_max_boxes)
        align_metrics_norm = (align_metrics * positive_ious / (positive_align_metrics + self.eps)).amax(dim=-2).unsqueeze(-1)
        target_scores = target_scores * align_metrics_norm

        return target_labels, target_boxes, target_scores, fg_mask.bool()

    def get_positive_mask(self, pred_scores, pred_boxes, anchor_points, gt_labels, gt_boxes, gt_mask):
        mask_anchors_in_gt = anchors_in_gt_boxes(anchor_points, gt_boxes)

        alignment_metrics, ious = self.get_alignment_metric(pred_scores, pred_boxes, gt_labels, gt_boxes, mask_anchors_in_gt * gt_mask)

        topk_mask = self.get_topk_candidates(alignment_metrics, mask=gt_mask.expand(-1, -1, self.topk))

        # merge masks (batch_size, num_max_boxes, n_anchors)
        mask = topk_mask * mask_anchors_in_gt * gt_mask

        return mask, alignment_metrics, ious

    def get_alignment_metric(self, pred_scores, pred_boxes, gt_labels, gt_boxes, mask):
        """
        Compute alignment metric
        """
        batch_size, n_anchors, _ = pred_scores.shape
        num_max_boxes = gt_boxes.shape[1]

        # Use self.device consistently
        # print('target device: ', self.device)

        # Create new tensors directly on self.device
        ious = torch.zeros((batch_size, num_max_boxes, n_anchors), dtype=pred_boxes.dtype, device=self.device)
        box_cls_scores = torch.zeros((batch_size, num_max_boxes, n_anchors), dtype=pred_scores.dtype, device=self.device)

        # Create indexing tensors directly on self.device
        batch_idxs = torch.arange(batch_size, device=self.device) \
                               .unsqueeze_(1) \
                               .expand(-1, num_max_boxes) \
                               .to(torch.long)
        
        class_idxs = gt_labels.squeeze(-1).to(torch.long).to(self.device)

        # Ensure all tensors for indexing are on self.device
        mask = mask.to(self.device)
        pred_scores = pred_scores.to(self.device) # Ensure indexed tensor is on device
        pred_boxes = pred_boxes.to(self.device)   # Ensure indexed tensor is on device
        
        # Scores for each grid for each GT cls
        # This line was the source of the error
        box_cls_scores[mask] = pred_scores[batch_idxs, :, class_idxs][mask]  # (bs, num_max_boxes, num_anchors)

        masked_pred_boxes = pred_boxes.unsqueeze(1).expand(-1, num_max_boxes, -1, -1)[mask]
        masked_gt_boxes = gt_boxes.unsqueeze(2).expand(-1, -1, n_anchors, -1)[mask]
        
        # We know gt_boxes is on self.device, so masked_gt_boxes is too
        # We moved pred_boxes to self.device, so masked_pred_boxes is too
        # No need for extra .to(self.device) calls here
        
        ious[mask] = bbox_iou(masked_gt_boxes, masked_pred_boxes, xywh=False).squeeze(-1).clamp_(min=0).to(ious.dtype)

        alignment_metrics = box_cls_scores.pow(self.alpha) * ious.pow(self.beta)

        return alignment_metrics, ious

    def get_topk_candidates(self, alignment_metrics:torch.Tensor, mask:torch.Tensor):
        """
        Select top-k candidates for each GT
        """
        # (batch_size, num_max_boxes, topk)
        topk_metrics, topk_idxs = torch.topk(alignment_metrics, self.topk, dim=-1, largest=True)

        if mask is None:
            mask = (topk_metrics.amax(dim=-1, keepdim=True) > self.eps).expand_as(topk_idxs)
        
        topk_idxs.masked_fill_(~mask, 0)

        # Create new tensors directly on self.device
        counts = torch.zeros(alignment_metrics.shape, dtype=torch.int8, device=self.device)
        increment = torch.ones_like(topk_idxs[:,:,:1], dtype=torch.int8, device=self.device)

        for i in range(self.topk):
            counts.scatter_add_(dim=-1, index=topk_idxs[:,:,i:i+1], src=increment)

        counts.masked_fill_(counts > 1, 0)

        return counts.to(alignment_metrics.dtype)

    def get_targets(self, gt_labels:torch.Tensor, gt_boxes:torch.Tensor, target_gt_box_idx:torch.Tensor, mask:torch.Tensor):
        """
        Get target labels, bboxes, scores for positive anchor points.
        """
        batch_size, num_max_boxes, _ = gt_boxes.shape
        _, num_anchors = target_gt_box_idx.shape

        # Create new tensors directly on self.device
        batch_idxs = torch.arange(batch_size, device=self.device).unsqueeze(-1)

        # target_gt_box_idx is on device from previous steps
        target_gt_box_idx = target_gt_box_idx + batch_idxs * num_max_boxes

        target_labels = gt_labels.long().flatten()[target_gt_box_idx]  # (batch_size, num_anchors)
        target_labels.clamp_(min=0, max=self.num_classes)

        target_boxes = gt_boxes.view(-1, 4)[target_gt_box_idx]  # (batch_size, num_anchors, 4)

        # Create new tensor directly on self.device
        target_scores = torch.zeros((batch_size, num_anchors, self.num_classes), dtype=torch.int64, device=self.device)
        target_scores.scatter_(dim=2, index=target_labels.unsqueeze(-1), value=1)

        scores_mask = mask.unsqueeze(-1).expand(-1, -1, self.num_classes)
        target_scores = torch.where(scores_mask > 0, target_scores, 0)

        return target_labels, target_boxes, target_scores


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        # All inputs are already on self.device from DetectionLoss
        weight = target_scores.sum(dim=-1)[mask].unsqueeze(dim=-1)
        iou = bbox_iou(pred_boxes[mask], target_boxes[mask], xywh=False)

        target_scores_sum = max(target_scores.sum(), 1)
        iou_loss = ((1 - iou) * weight).sum() / target_scores_sum

        if self.use_dfl:
            gt_ltrb = bbox2dist(target_boxes, anchor_points, self.reg_max)
            weight = target_scores.sum(dim=-1)[mask].unsqueeze(dim=-1)
            
            dfl_loss = df_loss(pred_box_dist[mask].view(-1, self.reg_max+1), gt_ltrb[mask]) * weight
            dfl_loss = dfl_loss.sum() / target_scores_sum
        else:
            dfl_loss = torch.tensor(0.0, device=self.device) # Explicit device

        return iou_loss, dfl_loss


class DetectionLoss(BaseLoss):
    def __init__(self, model, device:str, loss_gain=None):
        super().__init__(device)
        
        default_loss_gain = {'iou': 7.5, 'cls': 0.5, 'dfl': 1.5}
        
        detect_head = model.head

        self.nc = detect_head.nc
        self.n_outputs = detect_head.n_outputs
        self.reg_max = detect_head.reg_max
        self.stride = detect_head.stride
        self.loss_gains = loss_gain if loss_gain else default_loss_gain

        # Projects predicted boxes to different scales, on self.device
        self.proj = torch.arange(self.reg_max, device=self.device, dtype=torch.float)

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.bbox_loss = BboxLoss(self.reg_max-1, self.device, use_dfl=True)

        # Move the assigner module to self.device
        self.tal_assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0, device=self.device).to(self.device)

    def preprocess(self, targets:torch.Tensor, batch_size:int, scale_tensor:torch.Tensor) -> torch.Tensor:
        """
        Preprocesses target boxes to match predicted boxes batch size
        """
        if targets.shape[0] == 0:
            # Create tensor directly on self.device
            return torch.zeros(batch_size, 0, 5, device=self.device)
        
        im_idxs = targets[:,0]
        _, counts = im_idxs.unique(return_counts=True)

        # Create tensor directly on self.device
        out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
        for i in range(batch_size):
            mask = (im_idxs == i)
            n_matches = mask.sum()
            if n_matches > 0:
                out[i, :n_matches] = targets[mask, 1:]

        # scale_tensor is already on self.device from __call__
        out[..., 1:5] = xywh2xyxy_tensor(out[..., 1:5].mul_(scale_tensor))
        return out

    def decode_bbox(self, anchor_points:torch.Tensor, pred_box_dist:torch.Tensor):
        """
        Decodes bounding box coordinates.
        """
        b, a, c = pred_box_dist.shape
        # self.proj is already on self.device
        pred_boxes = pred_box_dist.view(b, a, 4, c//4).softmax(dim=3) @ self.proj
        return dist2bbox(pred_boxes, anchor_points, xywh=False)

    def __call__(self, batch:dict[str,torch.Tensor], preds:torch.Tensor):
        
        pred_box_dist, pred_cls = torch.cat(
            [xi.view(preds[0].shape[0], self.n_outputs, -1) for xi in preds], dim=2
        ).split((4*self.reg_max, self.nc), dim=1)

        pred_cls = pred_cls.permute(0, 2, 1).contiguous()
        pred_box_dist = pred_box_dist.permute(0, 2, 1).contiguous()

        # 🚨 CRITICAL FIX: Move all parts of the prediction to self.device
        pred_box_dist = pred_box_dist.to(self.device)
        pred_cls = pred_cls.to(self.device) # This was missing!
        
        batch_size = pred_box_dist.shape[0]
        # Create new tensors directly on self.device
        im_size = torch.tensor(preds[0].shape[2:], device=self.device) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(preds, torch.tensor(self.stride, device=self.device))

        # Ensure anchors are on self.device
        anchor_points = anchor_points.to(self.device)
        stride_tensor = stride_tensor.to(self.device)
        
        pred_boxes = self.decode_bbox(anchor_points, pred_box_dist)  # (b, h*w, 4) in (xyxy)

        # Move targets to self.device
        targets = torch.cat((batch['batch_idx'].view(-1,1), batch['cls'].view(-1,1), batch['bboxes']), dim=1).to(self.device)
        targets = self.preprocess(targets, batch_size, scale_tensor=im_size[[1,0,1,0]])
        
        gt_cls, gt_boxes = targets.split((1,4), dim=2)
        gt_mask = gt_boxes.sum(dim=2, keepdim=True) > 0
        
        # All inputs to tal_assigner are now on self.device
        _, target_boxes, target_scores, mask = self.tal_assigner(
            pred_cls.detach().sigmoid(), pred_boxes.detach() * stride_tensor, anchor_points * stride_tensor, gt_cls, gt_boxes, gt_mask
        )
        
        cls_loss = self.bce_loss(pred_cls, target_scores).sum() / max(target_scores.sum(), 1)

        if mask.sum() > 0:
            iou_loss, dfl_loss = self.bbox_loss.compute_loss(
                pred_box_dist, pred_boxes, target_boxes/stride_tensor, anchor_points, target_scores, mask
            )
        else:
            # Create tensors directly on self.device
            iou_loss = torch.tensor(0.0, device=self.device)
            dfl_loss = torch.tensor(0.0, device=self.device)

        loss = self.loss_gains['cls']*cls_loss + self.loss_gains['iou']*iou_loss + self.loss_gains['dfl']*dfl_loss

        return loss * batch_size
        # return loss