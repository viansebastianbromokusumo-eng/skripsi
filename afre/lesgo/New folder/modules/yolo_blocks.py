import torch
import torch.nn as nn 


class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        activation=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if activation else nn.identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = shortcut

    def forward(self, x):
        x_in = x  # for skip connection
        x = self.conv1(x)
        x = self.conv2(x)
        if self.shortcut:
            x += x_in
        return x


# YOLOV8 C2F Block
# Combines high-level features with contextual information to improve detection acc
# It effectively combines the benefits of two concepts:
# 1. Cross-Stage Partial (CSP) Network: Splits input feature map into 2 branches, where 1 goes thru series of computational layers (bottlenecks) 
#                                       and the other as a bypass. This allows the network to learn both shallow and deep feats simultaneously, 
#                                       enhancing acc and reduce redundant gradient info
# 2. Bottlenecks: The repeating Bottleneck layers allow the model to process features efficiently by compressing the channel dimension before 
#                 applying the main computations, then expanding it again.
# In short: C2f maximizes feature reuse and learning by explicitly combining the results of multiple computational stages within a single block.
class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_bottlenecks, shortcut=True):
        super().__init__()
        self.mid_channels = out_channels // 2
        self.num_bottlenecks = num_bottlenecks
        self.shortcut = shortcut
        self.conv1 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # sequence of bottleneck layers
        self.m = nn.ModuleList(
            [
                Bottleneck(self.mid_channels, self.mid_channels, self.shortcut)
                for _ in range(num_bottlenecks)
            ]
        )

        self.conv2 = Conv(
            (num_bottlenecks + 2) * out_channels // 2,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        x = self.conv1(x)

        # split x along channel dimension
        x1, x2 = x[:, : x.shape[1] // 2, :, :], x[:, : x.shape[1] // 2, :, :]

        # list of outputs
        outputs = [x1, x2]  # x1 is fed through the bottlenecks

        for i in range(self.num_bottlenecks):
            x1 = self.m[i](x1)  # [batch_size, 0.5 out_channel, w, h]
            outputs.insert(0, x1)

        outputs = torch.cat(
            outputs, dim=1
        )  # [batch_size, 0.5 out_channel(num_bottlenecks + 2), w, h]
        out = self.conv2(outputs)

        return out
    

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, pool_kernel_size=5):
        super().__init__()
        hidden_channel = in_channels // 2

        self.conv1 = Conv(
            in_channels, hidden_channel, kernel_size=1, stride=1, padding=0
        )

        # concatenate output of maxpool and feed to conv2
        self.conv2 = Conv(
            4 * hidden_channel, out_channels, kernel_size=1, stride=1, padding=0
        )

        # maxpool is applied at 3 different scales
        self.maxpool = nn.MaxPool2d(
            kernel_size=pool_kernel_size,
            stride=1,
            padding=pool_kernel_size // 2,
            dilation=1,
            ceil_mode=False,
        )

    def forward(self, x):
        x = self.conv1(x)

        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        y = torch.cat([x, y1, y2, y3], dim=1)

        y = self.conv2(y)

        return y
    
  
class Upsample(nn.Module): 
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode 
        
    def forward(self, x): 
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
    

class DFL(nn.Module): 
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch 
        
        self.conv = nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x) # DFL only has ch parameters
    
    def forward(self, x): 
        # x must have num_channels = 4 * ch, x = [bs, 4 * ch, c]
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(1, 2)
        
        # take softmax on channel dimension ot get dist probabilities 
        x = x.softmax(1)            #[ b, ch, 4, a]
        x = self.conv(x)            #[ b, 1, 4, a]
        return x.view(b, 4, a)      #[ b, 4, a]
    