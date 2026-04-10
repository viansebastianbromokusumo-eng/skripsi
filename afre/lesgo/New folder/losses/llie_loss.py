import torch 
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity


class UnsupervisedLLIELoss(nn.Module):
    def __init__(self, weight_exp=10, weight_col=5, weight_tv=200, weight_spa=1):
        super(UnsupervisedLLIELoss, self).__init__()
        self.weight_exp = weight_exp
        self.weight_col = weight_col
        self.weight_tv = weight_tv
        self.weight_spa = weight_spa

    def forward(self, input_img, enhanced_img, illu_map):
        # input_img: The original low-light image
        # enhanced_img: The output of your model
        # illu_map: The light curve/map your model estimated (if applicable)
        
        loss_spa = self.spatial_consistency_loss(input_img, enhanced_img)
        loss_col = self.color_constancy_loss(enhanced_img)
        loss_exp = self.exposure_control_loss(enhanced_img)
        loss_tv = self.illumination_smoothness_loss(illu_map)
        
        total_loss = (self.weight_spa * loss_spa + 
                      self.weight_col * loss_col + 
                      self.weight_exp * loss_exp + 
                      self.weight_tv * loss_tv)
        return total_loss

    def spatial_consistency_loss(self, x, y):
        # Enforces that edges in Output (y) match edges in Input (x)
        # 1. Create 4 kernels for Up, Down, Left, Right neighbors
        kernel_left = torch.FloatTensor([[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0,-1,0],[0,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0,0,0],[0,1,0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)

        # Average over channels to get 1-channel intensity
        x_mean = torch.mean(x, 1, keepdim=True)
        y_mean = torch.mean(y, 1, keepdim=True)

        d_org_left = F.conv2d(x_mean, kernel_left, padding=1)
        d_org_right = F.conv2d(x_mean, kernel_right, padding=1)
        d_org_up = F.conv2d(x_mean, kernel_up, padding=1)
        d_org_down = F.conv2d(x_mean, kernel_down, padding=1)

        d_enh_left = F.conv2d(y_mean, kernel_left, padding=1)
        d_enh_right = F.conv2d(y_mean, kernel_right, padding=1)
        d_enh_up = F.conv2d(y_mean, kernel_up, padding=1)
        d_enh_down = F.conv2d(y_mean, kernel_down, padding=1)

        d_left = torch.pow(d_org_left - d_enh_left, 2)
        d_right = torch.pow(d_org_right - d_enh_right, 2)
        d_up = torch.pow(d_org_up - d_enh_up, 2)
        d_down = torch.pow(d_org_down - d_enh_down, 2)
        
        return d_left.mean() + d_right.mean() + d_up.mean() + d_down.mean()

    def exposure_control_loss(self, x, mean_val=0.6):
        # Split into patches (e.g., 16x16) and force mean to be ~0.6
        x = torch.mean(x, 1, keepdim=True)
        mean = F.avg_pool2d(x, 16, stride=16)
        return torch.mean(torch.pow(mean - mean_val, 2))

    def color_constancy_loss(self, x):
        # Gray World Assumption: R, G, B means should be similar
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        d_rg = torch.pow(mr - mg, 2)
        d_rb = torch.pow(mr - mb, 2)
        d_gb = torch.pow(mg - mb, 2)
        return torch.sqrt(torch.pow(d_rg, 2) + torch.pow(d_rb, 2) + torch.pow(d_gb, 2)).mean()

    def illumination_smoothness_loss(self, x):
        # Total Variation on the Illumination Map
        # Ensures the light curve doesn't jitter
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        
        h_tv = torch.pow((x[:,:,1:,:] - x[:,:,:h_x-1,:]), 2).sum()
        w_tv = torch.pow((x[:,:,:,1:] - x[:,:,:,:w_x-1]), 2).sum()
        
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

def mean_squared_error(pred_img: torch.Tensor, gt_img: torch.Tensor) -> torch.Tensor: 
    assert pred_img.shape[-2:] == gt_img.shape[-2:], \
        f"Height and width of predicted and ground truth images must match. " \
        f"Predicted: {pred_img.shape[-2:]}, Ground Truth: {gt_img.shape[-2:]}"
    
    # Ensure both tensors have the same full shape
    assert pred_img.shape == gt_img.shape, \
        f"Predicted and ground truth image tensors must have the same shape. " \
        f"Predicted: {pred_img.shape}, Ground Truth: {gt_img.shape}"

    loss = torch.mean((pred_img - gt_img) ** 2)
    return loss

def mean_abs_error(pred_img: torch.Tensor, gt_img: torch.Tensor) -> torch.Tensor: 
    assert pred_img.shape[-2:] == gt_img.shape[-2:], \
        f"Height and width of predicted and ground truth images must match. " \
        f"Predicted: {pred_img.shape[-2:]}, Ground Truth: {gt_img.shape[-2:]}"
    
    # Ensure both tensors have the same full shape
    assert pred_img.shape == gt_img.shape, \
        f"Predicted and ground truth image tensors must have the same shape. " \
        f"Predicted: {pred_img.shape}, Ground Truth: {gt_img.shape}"

    loss = torch.mean(torch.abs(pred_img - gt_img))
    return loss

def smooth_l1(pred_img: torch.Tensor, gt_img: torch.Tensor, beta: float = 1.0) -> torch.Tensor: 
    assert pred_img.shape[-2:] == gt_img.shape[-2:], \
        f"Height and width of predicted and ground truth images must match. " \
        f"Predicted: {pred_img.shape[-2:]}, Ground Truth: {gt_img.shape[-2:]}"
    
    # Ensure both tensors have the same full shape
    assert pred_img.shape == gt_img.shape, \
        f"Predicted and ground truth image tensors must have the same shape. " \
        f"Predicted: {pred_img.shape}, Ground Truth: {gt_img.shape}"

    diff = torch.abs(pred_img - gt_img)
    
    # Calculate loss where abs(diff) < beta (L2 part)
    l2_loss_part = 0.5 * (diff ** 2) / beta
    # Calculate loss where abs(diff) >= beta (L1 part)
    l1_loss_part = diff - 0.5 * beta
    
    # Combine the two parts based on the condition
    loss = torch.where(diff < beta, l2_loss_part, l1_loss_part)
    
    return torch.mean(loss)

def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR).
    Assumes pred and target are tensors (B, C, H, W) or (C, H, W) in the range [0, data_range].
    """
    # Use torch's math functions
    if pred.shape != target.shape:
        raise ValueError("Predicted and target tensors must have the same shape.")
        
    # Calculate Mean Squared Error (MSE)
    mse = F.mse_loss(pred, target, reduction='mean')
    
    # Handle perfect match (MSE=0)
    if mse.item() == 0:
        return torch.tensor(100.0, dtype=pred.dtype, device=pred.device)
    
    # PSNR = 10 * log10(MAX_I^2 / MSE)
    # Since our tensors are normalized to [0, data_range], MAX_I is data_range.
    # We use torch.log10 to keep the result on the GPU/tensor format
    return 10 * torch.log10(data_range**2 / mse)

def ssim_metric(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Computes Structural Similarity Index (SSIM) for a single image (C, H, W).
    NOTE: This implementation requires NumPy/Skimage and handles only a single image.
    If you need batch SSIM, you must loop over the batch dimension outside this function.
    """
    if len(pred.shape) != 3:
        raise ValueError("SSIM metric requires a single image input (C, H, W).")
        
    # Convert to CPU numpy and permute C, H, W -> H, W, C for Skimage/NumPy
    pred_np = pred.detach().cpu().permute(1, 2, 0).numpy()
    target_np = target.detach().cpu().permute(1, 2, 0).numpy()

    # Calculate SSIM
    # channel_axis=2 is used for HWC format
    # multichannel is deprecated in favor of channel_axis
    # average=True is default, returning a scalar SSIM value
    try:
        score = structural_similarity(
            pred_np, 
            target_np, 
            channel_axis=2, 
            data_range=data_range,
            win_size=11, 
            gaussian_weights=True
        )
        return score
    except Exception as e:
        print(f"SSIM Calculation Failed: {e}")
        return 0.0