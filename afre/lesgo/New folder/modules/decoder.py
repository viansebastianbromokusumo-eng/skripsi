import torch
import torch.nn as nn
from modules.arb_components import ARB_Refined


class ResidualBlock(nn.Module):
    """
    残差块 (Residual Block)
    Refactored to use conditional logic in forward pass instead of nn.Identity.
    """
    def __init__(self, channels: int, use_arb: bool = False):
        super().__init__()
        
        self.use_arb = use_arb
        
        # 1. Conditionally register ARB module
        if use_arb:
            # Only define and register ARB if needed
            self.arb = ARB_Refined(channels, channels)
        
        # 2. Main convolutional path (always registered)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
            
        # 2. Main path convolutions
        out = self.conv1(x) # Note: If ARB is used, it pre-processes the input to conv1
        out = self.conv2(out)
        
        # 1. Apply ARB conditionally
        if self.use_arb:
            # We apply ARB to the input feature map 'x'
            out = self.arb(out)
        else:
            # If ARB is off, the primary path starts with the input 'x'
            out = out
        
        # 3. Residual connection and final SiLU
        out = out + identity
        
        return nn.SiLU(inplace=True)(out)
        

# --- 2. UpBlock (Upsampling Module) ---]
class UpBlock(nn.Module):
    """
    上采样模块结构 (Upsampling Module Structure) - Figure 2(b)
    由若干个残差块以及一个反卷积构成，实现分辨率翻倍、通道数减半。
    Consists of several residual blocks and a deconvolution (ConvTranspose2d)
    to double the resolution and halve the number of channels.
    """
    def __init__(self, in_channels: int, out_channels: int, num_residual_blocks: int = 2, use_arb: bool = False):
        super().__init__()
        
        # 1. Residual Blocks: Perform feature processing at the lower resolution
        residual_layers = [
            ResidualBlock(in_channels, use_arb=use_arb) for _ in range(num_residual_blocks)
        ]
        self.residuals = nn.Sequential(*residual_layers)
        
        # self.arb = ARB_Refined(in_channels, out_channels) if use_arb else nn.Identity()
        
        # 2. Deconvolution (Upsampling): Doubles resolution and reduces channels
        # Kernel size 3x3, stride 2, padding 1, output_padding 1
        # This doubles H/W and reduces channels from in_channels to out_channels
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                stride=2, 
                padding=1, 
                output_padding=1, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Residual Processing
        residual_out = self.residuals(x)
        
        # 2. Upsampling (Deconvolution)
        out = self.deconv(residual_out)
        
        # Output is now (H*2, W*2, C_out)
        return out


# --- 3. LowLightEnhancementDecoder (Full Decoder) ---
class LowLightEnhancementDecoder(nn.Module):
    """
    低照度增强解码器 (Low-light Enhancement Decoder) - Figure 2(a)
    基于U-Net结构，利用主干网络的前3个阶段特征进行图像复原。
    U-Net based structure utilizing the first 3 stages of backbone features
    (Dark3, Dark2, Stem) for image restoration.
    """
    def __init__(self, base_channels: int, use_arb: bool = False):
        super().__init__()
        if use_arb:
            print('ARB LLIE Decoder Initialized')
        else:
            print('Night Dec initalized')
        
        C = base_channels # Channel count for Dark3 (e.g., C=256)
        
        # Up_block1: Input C (from Dark3) -> Output C/2 (Up1)
        # Resolution: W/8 -> W/4
        self.up_block1 = UpBlock(in_channels=C, out_channels=C // 2, use_arb=use_arb)
        
        # Up_block2: Input C/2 (Fused Up1 + Dark2) -> Output C/4 (Up2)
        # Resolution: W/4 -> W/2
        # Note: The input channel size for UpBlock2 must match the fused size (C/2)
        self.up_block2 = UpBlock(in_channels=C // 2, out_channels=C // 4, use_arb=use_arb)

        # Up_block3: Input C/4 (Fused Up2 + Stem) -> Output C/8 (Up3)
        # Resolution: W/2 -> W
        # Note: The input channel size for UpBlock3 must match the fused size (C/4)
        self.up_block3 = UpBlock(in_channels=C // 4, out_channels=C // 8, use_arb=use_arb)
        
        # Final Convolution: C/8 channels -> 3 channels (RGB)
        self.final_conv = nn.Sequential(
            nn.Conv2d(C // 8, 3, kernel_size=3, padding=1),
            # nn.Tanh()
            # nn.Sigmoid()
        )

    def forward(self, in_img: torch.Tensor, dark3: torch.Tensor, dark2: torch.Tensor, stem: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dark3 (torch.Tensor): Feature map from Dark3 (smallest spatial size, largest channels C)
            dark2 (torch.Tensor): Feature map from Dark2 (mid spatial size, C/2 channels)
            stem (torch.Tensor): Feature map from Stem (largest spatial size, C/4 channels)
        """
        
        # 1. Process Dark3 (C, H/8, W/8)
        up1_raw = self.up_block1(dark3) # -> (C/2, H/4, W/4)
        
        # 2. Fuse with Dark2 (Skip Connection)
        # The text specifies addition (相加)
        up2_input = up1_raw + dark2 # -> (C/2, H/4, W/4)
        up2_raw = self.up_block2(up2_input) # -> (C/4, H/2, W/2)
        
        # 3. Fuse with Stem (Skip Connection)
        up3_input = up2_raw + stem # -> (C/4, H/2, W/2)
        up3_raw = self.up_block3(up3_input) # -> (C/8, H, W)
        
        # 4. Final Output Convolution
        res_map = self.final_conv(up3_raw) # -> (3, H, W)
        
        out_add = res_map + in_img
        output = torch.clamp(out_add, 0, 1)

        return output, res_map


class GradientScaler(torch.autograd.Function):
    """
    Custom autograd function to scale gradients during the backward pass.
    Forward pass: Identity function.
    Backward pass: Multiplies the incoming gradient by 'scale'.
    """
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Scale the gradient flowing backward
        # Returning None for 'scale' because it doesn't need a gradient
        return grad_output * ctx.scale, None

def scale_gradient(x, scale):
    """Helper function to apply GradientScaler cleanly."""
    return GradientScaler.apply(x, scale)


class LowLightEnhancementDecoder_Scale(nn.Module):
    """
    低照度增强解码器 (Low-light Enhancement Decoder) - Figure 2(a)
    基于U-Net结构，利用主干网络的前3个阶段特征进行图像复原。
    U-Net based structure utilizing the first 3 stages of backbone features
    (Dark3, Dark2, Stem) for image restoration.
    """
    def __init__(self, base_channels: int, use_arb: bool = False):
        super().__init__()
        if use_arb:
            print('ARB LLIE Decoder Initialized')
        else:
            print('Night Dec initalized')
        
        C = base_channels # Channel count for Dark3 (e.g., C=256)
        
        # Up_block1: Input C (from Dark3) -> Output C/2 (Up1)
        # Resolution: W/8 -> W/4
        self.up_block1 = UpBlock(in_channels=C, out_channels=C // 2, use_arb=use_arb)
        
        # Up_block2: Input C/2 (Fused Up1 + Dark2) -> Output C/4 (Up2)
        # Resolution: W/4 -> W/2
        # Note: The input channel size for UpBlock2 must match the fused size (C/2)
        self.up_block2 = UpBlock(in_channels=C // 2, out_channels=C // 4, use_arb=use_arb)

        # Up_block3: Input C/4 (Fused Up2 + Stem) -> Output C/8 (Up3)
        # Resolution: W/2 -> W
        # Note: The input channel size for UpBlock3 must match the fused size (C/4)
        self.up_block3 = UpBlock(in_channels=C // 4, out_channels=C // 8, use_arb=use_arb)
        
        # Final Convolution: C/8 channels -> 3 channels (RGB)
        self.final_conv = nn.Sequential(
            nn.Conv2d(C // 8, 3, kernel_size=3, padding=1),
            # nn.Tanh()
            # nn.Sigmoid()
        )

    def forward(self, in_img: torch.Tensor, dark3: torch.Tensor, dark2: torch.Tensor, stem: torch.Tensor, grad_scale: float = 0.1) -> torch.Tensor:
        """
        Args:
            dark3 (torch.Tensor): Feature map from Dark3 (smallest spatial size, largest channels C)
            dark2 (torch.Tensor): Feature map from Dark2 (mid spatial size, C/2 channels)
            stem (torch.Tensor): Feature map from Stem (largest spatial size, C/4 channels)
        """
        
       # --- Gradient Scaling Barrier ---
        # Forward pass: these remain mathematically identical to the inputs.
        # Backward pass: gradients passing through here are multiplied by grad_scale.
        dark3_scaled = scale_gradient(dark3, grad_scale)
        dark2_scaled = scale_gradient(dark2, grad_scale)
        stem_scaled = scale_gradient(stem, grad_scale)
        
        # 1. Process Dark3 (C, H/8, W/8)
        up1_raw = self.up_block1(dark3_scaled) 
        
        # 2. Fuse with Dark2 (Skip Connection)
        up2_input = up1_raw + dark2_scaled 
        up2_raw = self.up_block2(up2_input) 
        
        # 3. Fuse with Stem (Skip Connection)
        up3_input = up2_raw + stem_scaled 
        up3_raw = self.up_block3(up3_input) 
        
        # 4. Final Output Convolution
        res_map = self.final_conv(up3_raw) 
        
        out_add = res_map + in_img
        output = torch.clamp(out_add, 0, 1)

        return output, res_map
