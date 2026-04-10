import torch
import torch.nn as nn
import torch.nn.functional as F
from .arb import ARB_Refined


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


class LowLightEnhancementDecoder_Yol(nn.Module):
    # args in YAML: [64, True]
    # inc from YOLO: [64, 128, 256] (Automatically passed)
    def __init__(self, inc, base_channels: int, use_arb: bool = False):
        super().__init__()
        
        # 1. Unpack Input Channels (Order matches YAML 'from': [0, 2, 4])
        # ch_stem=64, ch_dark2=128, ch_dark3=256
        self.ch_stem, self.ch_dark2, self.ch_dark3 = inc 
        
        # 2. Dynamic Channel Assignment
        # We must use the ACTUAL input channel size for the first block, 
        # NOT the manual base_channels, or shapes won't match.
        C = self.ch_dark3 

        print(f'Using ARB in Decoder: {use_arb}')
        print(f"Initializing LLIE Decoder. Input channels: {inc}")
        print(f"Processing Dark3 ({C}) -> Dark2 ({self.ch_dark2}) -> Stem ({self.ch_stem})")

        # Up_block1: Takes Dark3 (256) -> Outputs 128 (matches Dark2 for addition)
        self.up_block1 = UpBlock(in_channels=C, out_channels=self.ch_dark2, use_arb=use_arb)
        
        # Up_block2: Takes Dark2 size (128) -> Outputs 64 (matches Stem for addition)
        self.up_block2 = UpBlock(in_channels=self.ch_dark2, out_channels=self.ch_stem, use_arb=use_arb)

        # Up_block3: Takes Stem size (64) -> Outputs base_channels (e.g., 32 or 64)
        # This is where we can finally use your 'base_channels' arg if you want specific output size
        self.up_block3 = UpBlock(in_channels=self.ch_stem, out_channels=base_channels, use_arb=use_arb)
        
        # Final Convolution: base_channels -> 3 channels (RGB Residual)
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            # nn.Tanh() # Recommended if you want residual to be in range [-1, 1]
        )

    def forward(self, x):
        # x is a LIST of 3 tensors sent by YOLO [layer0, layer2, layer4]
        # Unpack them (Order matters!)
        stem, dark2, dark3 = x
        
        # 1. Process Dark3
        up1_raw = self.up_block1(dark3) 
        
        # 2. Fuse with Dark2 (Skip Connection)
        # Ensure up1_raw and dark2 have same H,W via interpolation if needed, 
        # but since they are from YOLO backbone, sizes should align naturally (H/8 -> H/4).
        if up1_raw.shape != dark2.shape:
             up1_raw = F.interpolate(up1_raw, size=dark2.shape[2:], mode='bilinear', align_corners=False)
        
        up2_input = up1_raw + dark2 
        up2_raw = self.up_block2(up2_input) 
        
        # 3. Fuse with Stem (Skip Connection)
        if up2_raw.shape != stem.shape:
             up2_raw = F.interpolate(up2_raw, size=stem.shape[2:], mode='bilinear', align_corners=False)

        up3_input = up2_raw + stem 
        up3_raw = self.up_block3(up3_input) 
        
        # 4. Final Output Convolution
        # NOTE: This is likely H/2 resolution (Stem size). 
        # You might need one final upsample to get to H (original image size)
        res_map = self.final_conv(up3_raw)
        
        # UPSAMPLE TO ORIGINAL IMAGE SIZE (Since Stem is H/2)
        # We assume target size is 2x current size
        # res_map = F.interpolate(res_map, scale_factor=2, mode='bilinear', align_corners=False)

        # RETURN ONLY RES_MAP
        # We cannot add in_img here because we don't have it.
        # We will handle the addition in the Loss function.
        return res_map