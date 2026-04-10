import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=nn.SiLU):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.conv(x))


class NCU(nn.Module):
    def __init__(self, channels):
        super(NCU, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.silu = nn.SiLU(inplace=True)
    def forward(self, F_in):
        # return self.relu(self.conv(F_in))
        return self.silu(self.conv(F_in))
        

class CAU(nn.Module):
    def __init__(self, channels, r=4):
        super(CAU, self).__init__()
        # ... (CAU implementation as before, returning F_in * F_ca)
        self.channels = channels
        self.f_ca1 = nn.Sequential(
            nn.Conv2d(channels, channels // r, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(inplace=True)
            nn.SiLU(inplace=True)
        )
        self.f_ca2 = nn.Conv2d(channels // r, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, F_in):
        F_caa = F.avg_pool2d(F_in, F_in.size()[2:])
        F_cam = F.max_pool2d(F_in, F_in.size()[2:])
        F_caA = self.f_ca1(F_caa)
        F_caM = self.f_ca1(F_cam)
        F_ca_weights = torch.sigmoid(self.f_ca2(F_caA) + self.f_ca2(F_caM))
        # The output F_ca is the input F_in multiplied by the weights
        return F_in * F_ca_weights


class SAU(nn.Module):
    def __init__(self):
        super(SAU, self).__init__()
        self.f_sa = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, F_in):
        F_saa = torch.mean(F_in, dim=1, keepdim=True)
        F_sam, _ = torch.max(F_in, dim=1, keepdim=True)
        F_saAM = torch.cat([F_saa, F_sam], dim=1)
        F_sa_weights = torch.sigmoid(self.f_sa(F_saAM))
        # The output F_sa is the input F_in multiplied by the weights
        return F_in * F_sa_weights


# --- Refined ARB Class ---
class ARB_Refined(nn.Module):
    """
    Refined Attention-oriented Residual Block (ARB) based on the image structure.
    """
    def __init__(self, in_channels, out_channels):
        super(ARB_Refined, self).__init__()
        
        # 0. Initial feature map extraction (F_in) from the overall input (x_in)
        # This is the "Convolutional (3x3)" block on the far left.
        # It includes the ReLU from Eq. 1: F_in = ReLU(f_in(x_in))
        self.f_in = BasicConv(in_channels, out_channels, kernel_size=3, padding=1)
        
        # ARB components operate on F_in
        self.NCU = NCU(out_channels)
        self.CAU = CAU(out_channels, r=4)
        self.SAU = SAU() 

        # 1. Post-Fusion Convolution (The first 3x3 conv after the "+" fusion)
        self.conv_post_fusion = BasicConv(out_channels, out_channels, 
                                          kernel_size=3, padding=1, 
                                          activation=None) # No ReLU here in the diagram

        # 2. Final Attention Branch (The dashed box)
        # Final Convolution (The second 3x3 conv)
        self.conv_final = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # Batch Normalization
        self.BN = nn.BatchNorm2d(out_channels) 
        # ReLU (applied after the final residual connection in the standard ResNet view,
        # but Eq. 8 puts it at the end of the attention path, F_arb = ReLU(BN(...)))
        
    def forward(self, x_in):
        # 1. Initial Feature Extraction (F_in) - This is the primary residual connection input
        F_in = self.f_in(x_in) 
        
        # 2. Three Parallel Branches (NCU, CAU, SAU)
        F_nc = self.NCU(F_in)
        F_ca = self.CAU(F_in)
        F_sa = self.SAU(F_in)

        # 3. Fusion (First Addition)
        fused_features = F_nc + F_ca + F_sa

        # 4. Post-Fusion Convolution (The one you spotted!)
        F_mid = self.conv_post_fusion(fused_features)

        # 5. Final Path (Attention Branch F(x) or F_arb_path)
        F_arb_path = self.BN(self.conv_final(F_mid))

        # 6. Final Residual Connection (Second Addition)
        # Output = F_arb_path + F_in
        F_out = F_arb_path + F_in
        
        # 7. Final Activation (ReLU)
        # output = F.relu(F_out, inplace=True)
         
        output = F.silu(F_out, inplace=True)
        
        return output
