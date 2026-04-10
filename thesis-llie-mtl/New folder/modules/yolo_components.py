import torch
import torch.nn as nn
from math import log
from modules.yolo_blocks import Conv, C2f, SPPF, DFL, Upsample
from modules.arb_components import ARB_Refined
from utils.yolo_helpers import yolo_params, make_anchors, dist2bbox


# YOLOv8 Clone Backbone
class BackBone(nn.Module): 
    def __init__(self, version, in_channels=3, shortcut=True):
        super().__init__()
        depth, width, ratio = yolo_params(version)
        
        # conv layers
        self.conv0 = Conv(in_channels, int(64 * width), kernel_size=3, stride=2, padding=1)
        self.conv1 = Conv(int(64 * width), int(128 * width), kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv(int(128 * width), int(256 * width), kernel_size=3, stride=2, padding=1)
        self.conv5 = Conv(int(256 * width), int(512 * width), kernel_size=3, stride=2, padding=1)
        self.conv7 = Conv(int(512 * width), int(512 * width * ratio), kernel_size=3, stride=2, padding=1)
        
        # c2f layers 
        self.c2f_2 = C2f(int(128 * width), int(128 * width), num_bottlenecks=int(3 * depth), shortcut=True)
        self.c2f_4 = C2f(int(256 * width), int(256 * width), num_bottlenecks=int(6 * depth), shortcut=True)
        self.c2f_6 = C2f(int(512 * width), int(512 * width), num_bottlenecks=int(6 * depth), shortcut=True)
        self.c2f_8 = C2f(int(512 * width * ratio), int(512 * width * ratio), num_bottlenecks=int(3 * depth), shortcut=True)
        
        # sppf 
        self.sppf = SPPF(int(512 * width * ratio), int(512 * width * ratio))
        
    def forward(self, x): 
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.c2f_2(x)
        x = self.conv3(x)
        
        out1 = self.c2f_4(x) # keep for output 
        
        x = self.conv5(out1)
        
        out2 = self.c2f_6(x) # keep for output 
        
        x = self.conv7(out2)
        x = self.c2f_8(x)
        
        out3 = self.sppf(x)
        
        return out1, out2, out3


# Custom YOLOv8 Backbone. Returns 4 stages from the backbone
# reference paper requires h/2, h/4 and h/8 for decoding, however to follow YOLO backbone it would make sense to do it after 
# each C2F, as it plays a crucial role in maximizing feature learning
class BackBone_4(nn.Module): 
    def __init__(self, version, in_channels=3, shortcut=True):
        super().__init__()
        depth, width, ratio = yolo_params(version)
        
        # conv layers
        self.conv0 = Conv(in_channels, int(64 * width), kernel_size=3, stride=2, padding=1)
        self.conv1 = Conv(int(64 * width), int(128 * width), kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv(int(128 * width), int(256 * width), kernel_size=3, stride=2, padding=1)
        self.conv5 = Conv(int(256 * width), int(512 * width), kernel_size=3, stride=2, padding=1)
        self.conv7 = Conv(int(512 * width), int(512 * width * ratio), kernel_size=3, stride=2, padding=1)
        
        # c2f layers 
        self.c2f_2 = C2f(int(128 * width), int(128 * width), num_bottlenecks=int(3 * depth), shortcut=True)
        self.c2f_4 = C2f(int(256 * width), int(256 * width), num_bottlenecks=int(6 * depth), shortcut=True)
        self.c2f_6 = C2f(int(512 * width), int(512 * width), num_bottlenecks=int(6 * depth), shortcut=True)
        self.c2f_8 = C2f(int(512 * width * ratio), int(512 * width * ratio), num_bottlenecks=int(3 * depth), shortcut=True)
        
        # sppf 
        self.sppf = SPPF(int(512 * width * ratio), int(512 * width * ratio))
        
    def forward(self, x):

        x = self.conv0(x)              # H/2
        x = self.conv1(x)              # H/4
        
        stem = self.c2f_2(x)           # H/4
        
        x = self.conv3(stem)
        dark2 = self.c2f_4(x)          # H/4

        x = self.conv5(dark2)
        dark3 = self.c2f_6(x)          # H/8

        x = self.conv7(dark3)
        dark4 = self.c2f_8(x)          # H/16

        out = self.sppf(dark4)         # H/32

        return stem, dark2, dark3, out
        
        
# Custom YOLOv8 Backbone. Returns all 5 stages from the backbone
# this backbone follows the paper's requirement of h/2, h/4 and h/8
class BackBone_5(nn.Module): 
    def __init__(self, version, in_channels=3, shortcut=True):
        super().__init__()
        depth, width, ratio = yolo_params(version)
        
        # conv layers
        self.conv0 = Conv(in_channels, int(64 * width), kernel_size=3, stride=2, padding=1)
        self.conv1 = Conv(int(64 * width), int(128 * width), kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv(int(128 * width), int(256 * width), kernel_size=3, stride=2, padding=1)
        self.conv5 = Conv(int(256 * width), int(512 * width), kernel_size=3, stride=2, padding=1)
        self.conv7 = Conv(int(512 * width), int(512 * width * ratio), kernel_size=3, stride=2, padding=1)
        
        # c2f layers 
        self.c2f_2 = C2f(int(128 * width), int(128 * width), num_bottlenecks=int(3 * depth), shortcut=True)
        self.c2f_4 = C2f(int(256 * width), int(256 * width), num_bottlenecks=int(6 * depth), shortcut=True)
        self.c2f_6 = C2f(int(512 * width), int(512 * width), num_bottlenecks=int(6 * depth), shortcut=True)
        self.c2f_8 = C2f(int(512 * width * ratio), int(512 * width * ratio), num_bottlenecks=int(3 * depth), shortcut=True)
        
        # sppf 
        self.sppf = SPPF(int(512 * width * ratio), int(512 * width * ratio))
        
    def forward(self, x):
        stem = self.conv0(x)              # H/2
        x = self.conv1(stem)              # H/2
        
        dark2 = self.c2f_2(x)             # H/4
        
        x = self.conv3(dark2)
        dark3 = self.c2f_4(x)             # H/4
        
        x = self.conv5(dark3)
        dark4 = self.c2f_6(x)             # H/8
        
        x = self.conv7(dark4)
        x = self.c2f_8(x)          # H/16
        
        out = self.sppf(x)         # H/32
        return stem, dark2, dark3, dark4, out
    
    
class BackBone_5_ARB(nn.Module): 
    def __init__(self, version, in_channels=3, use_arb=False, shortcut=True):
        super().__init__()
        depth, width, ratio = yolo_params(version)
        
        self.use_arb = use_arb
        
        if use_arb: 
            print('ARB Backbone Initialized')
        else: 
            print('NO ARB in Backbone')
            
        # Standard layers (always present)
        self.conv0 = Conv(in_channels, int(64 * width), kernel_size=3, stride=2, padding=1)
        self.conv1 = Conv(int(64 * width), int(128 * width), kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv(int(128 * width), int(256 * width), kernel_size=3, stride=2, padding=1)
        self.conv5 = Conv(int(256 * width), int(512 * width), kernel_size=3, stride=2, padding=1)
        self.conv7 = Conv(int(512 * width), int(512 * width * ratio), kernel_size=3, stride=2, padding=1)
        
        self.c2f_2 = C2f(int(128 * width), int(128 * width), num_bottlenecks=int(3 * depth), shortcut=True)
        self.c2f_4 = C2f(int(256 * width), int(256 * width), num_bottlenecks=int(6 * depth), shortcut=True)
        self.c2f_6 = C2f(int(512 * width), int(512 * width), num_bottlenecks=int(6 * depth), shortcut=True)
        self.c2f_8 = C2f(int(512 * width * ratio), int(512 * width * ratio), num_bottlenecks=int(3 * depth), shortcut=True)
        
        self.sppf = SPPF(int(512 * width * ratio), int(512 * width * ratio))
        
        # --- CONDITIONAL ARB MODULES ---
        if use_arb:
            # ARB for C2f_2
            self.arb_c2f2 = ARB_Refined(int(128 * width), int(128 * width))
            # ARB for C2f_4 (Dark3 output)
            self.arb_c2f4 = ARB_Refined(int(256 * width), int(256 * width))
            # ARB for C2f_6 (Dark4 output)
            self.arb_c2f6 = ARB_Refined(int(512 * width), int(512 * width))
            # ARB for C2f_8 
            self.arb_c2f8 = ARB_Refined(int(512 * width * ratio), int(512 * width * ratio))
            # ARB for SPPF (Final output) - Note: The original code commented this out, leaving it defined for completeness
            # self.arb_sppf = ARB_Refined(int(512 * width * ratio), int(512 * width * ratio))
        
    def forward(self, x):
        stem = self.conv0(x) # H/2
        x = self.conv1(stem) # H/2
        
        dark2 = self.c2f_2(x) # H/4
        if self.use_arb:
            dark2 = self.arb_c2f2(dark2) # <-- Conditional ARB 1
        
        x = self.conv3(dark2)
        dark3 = self.c2f_4(x)# H/4
        if self.use_arb:
            dark3 = self.arb_c2f4(dark3) # <-- Conditional ARB 2
        
        x = self.conv5(dark3)
        dark4 = self.c2f_6(x) # H/8
        if self.use_arb:
            dark4 = self.arb_c2f6(dark4) # <-- Conditional ARB 3
        
        x = self.conv7(dark4)
        x = self.c2f_8(x) # H/16
        if self.use_arb:
            x = self.arb_c2f8(x) # <-- Conditional ARB 4
        
        out = self.sppf(x) # H/32
        # If the ARB_SPPF was meant to be included:
        # if self.use_arb and hasattr(self, 'arb_sppf'):
        #     out = self.arb_sppf(out)        # <-- Conditional ARB 5 (Final)
        
        return stem, dark2, dark3, dark4, out


class Neck(nn.Module): 
    def __init__(self, version):
        super().__init__()
        d, w, r = yolo_params(version)
        
        self.up = Upsample() # no trainable params 
        self.c2f_1 = C2f(in_channels=int(512 * w * (1 + r)), out_channels=int(512 * w), num_bottlenecks=int(3 * d), shortcut=False)
        self.c2f_2 = C2f(in_channels=int(768 * w), out_channels=int(256 * w), num_bottlenecks=int(3 * d), shortcut=False)
        self.c2f_3 = C2f(in_channels=int(768 * w), out_channels=int(512 * w), num_bottlenecks=int(3 * d), shortcut=False)
        self.c2f_4 = C2f(in_channels=int(512 * w * (1 + r)), out_channels=int(512 * w * r), num_bottlenecks=int(3 * d), shortcut=False)
        
        self.conv_1 = Conv(in_channels=int(256 * w), out_channels=int(256 * w), kernel_size=3, stride=2, padding=1)
        self.conv_2 = Conv(in_channels=int(512 * w), out_channels=int(512 * w), kernel_size=3, stride=2, padding=1)
        
    # x_res1, 2, 3 is output from backbone
    # 1 --> 3 = smallest to largest channels
    def forward(self, x_res1, x_res2, x_res3): 
        res1 = x_res3                       # for residual connection 
        
        x = self.up(x_res3)
        x = torch.cat([x, x_res2], dim=1)   # dim=1, --> along channel dimension
        res2 = self.c2f_1(x)                # for residual connection
        x = self.up(res2)
        x = torch.cat([x, x_res1], dim=1)
        
        out1 = self.c2f_2(x)
        
        x = self.conv_1(out1)
        x = torch.cat([x, res2], dim=1)
        
        out2 = self.c2f_3(x)
        
        x = self.conv_2(out2)
        x = torch.cat([x, res1], dim=1)
        
        out3 = self.c2f_4(x)
        
        return out1, out2, out3
    

class Head(nn.Module): 
    def __init__(self, version, ch=16, num_classes=80):
        super().__init__()
        self.ch = ch 
        self.coords = self.ch * 4 
        self.nc = num_classes
        self.no = self.coords + self.nc 
        self.stride = torch.zeros(3) 
        d, w, r = yolo_params(version=version)
        
        # for bounding box
        self.box = nn.ModuleList([
            # 1st det head
            nn.Sequential(
                Conv(int(256 * w), self.coords, kernel_size=3, stride=1, padding=1),
                Conv(self.coords, self.coords, kernel_size=3, stride=1, padding=1), 
                nn.Conv2d(self.coords, self.coords, kernel_size=1, stride=1)
                ),
            
            # 2nd det head
            nn.Sequential(
                Conv(int(512 * w), self.coords, kernel_size=3, stride=1, padding=1),
                Conv(self.coords, self.coords, kernel_size=3, stride=1, padding=1), 
                nn.Conv2d(self.coords, self.coords, kernel_size=1, stride=1)
                ),
            
            # 3rd det head
            nn.Sequential(
                Conv(int(512 * w * r), self.coords, kernel_size=3, stride=1, padding=1),
                Conv(self.coords, self.coords, kernel_size=3, stride=1, padding=1), 
                nn.Conv2d(self.coords, self.coords, kernel_size=1, stride=1)
                ),
        ])
        
        self.cls = nn.ModuleList([
            nn.Sequential(
                Conv(int(256 * w), self.nc, kernel_size=3, stride=1, padding=1),
                Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                Conv(int(512 * w), self.nc, kernel_size=3, stride=1, padding=1),
                Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                Conv(int(512 * w * r), self.nc, kernel_size=3, stride=1, padding=1),
                Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)
            ),
        ])
        
        self.dfl = DFL() 
    
    def make_anchors(self, x, strides, offset=0.5): 
        # x = list of feature maps: x=x[0], ..., x[n - 1], in our case n = num_detection_heads = 3, each having shapes [bs, ch, w, h]
        # each feature map x[i] gives output[i] = w * h anchor coords + w * h stride values
        # strides = list of stride values indicating how much the spatial resolution of the feature map is reduced compared to the original image 
        
        assert x is not None 
        anchor_tensor, stride_tensor = [], [] 
        dtype, device = x[0].dtype, x[0].device
        
        for i, stride in enumerate(strides): 
            _, _, h, w = x[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + offset # x coords of anchor centers 
            sy = torch.arange(end=h, device=device, dtype=dtype) + offset # y coords of anchor centers 
            sy, sx = torch.meshgrid(sy, sx)
            anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_tensor), torch.cat(stride_tensor)
        
    def forward(self, x, inference=False): 
        # x = output of Neck = list of 3 tensors with different resolutions and channel dimensions
        #   x[0] = [bs, ch0, w0, h0], x[1] = [bs, ch1, w1, h1], x[2] = [bs, ch2, w2, h2]
        for i in range(len(self.box)):              # det head 
            box = self.box[i](x[i])                 # [bs, num_coords, w, h]
            cls = self.cls[i](x[i])                 # [bs, num_classes, w, h]
            x[i] = torch.cat((box, cls), dim=1)     # [bs, num_coords + num_classes, w, h]
        
        # if training no DFL output
        # if self.training: 
        #     return x                                # [3, bs, num_coords + num_classes, w, h]
        if not inference: 
            return x 
        # in inference, dfl produces refined bounding boc coords
        anchors, strides = (i.transpose(0, 1) for i in self.make_anchors(x, self.stride))
        
        # concatenante preds from all det layers 
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2) #[bs, 4 * self.ch + self.nc, sum_i(h[i] w[i])]

        # split out preds for box and cls 
                # box = [bs, 4 * self.ch, sum_i(h[i] w[i])]
                # clas = [bs, self.nc, sum_i(h[i] w[i])]
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)
        
        a, b = self.dfl(box).chunk(2, 1)
        a = anchors.unsqueeze(0) - a
        b = anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)
        
        return torch.cat(tensors=(box * strides, cls.sigmoid()), dim=1)   


class Head_2(nn.Module): 
    anchors = torch.empty(0)
    strides = torch.empty(0)
    shape = None
    
    def __init__(self, version, ch=16, num_classes=80):
        super().__init__()
        self.ch = ch 
        self.coords = self.ch * 4 
        self.nc = num_classes
        self.no = self.coords + self.nc 
        self.stride = torch.zeros(3) 
        self.reg_max = 16 
        self.n_outputs = 4 * self.reg_max + self.nc
        d, w, r = yolo_params(version=version)
        
        # for bounding box
        self.box = nn.ModuleList([
            # 1st det head
            nn.Sequential(
                Conv(int(256 * w), self.coords, kernel_size=3, stride=1, padding=1),
                Conv(self.coords, self.coords, kernel_size=3, stride=1, padding=1), 
                nn.Conv2d(self.coords, self.coords, kernel_size=1, stride=1)
                ),
            
            # 2nd det head
            nn.Sequential(
                Conv(int(512 * w), self.coords, kernel_size=3, stride=1, padding=1),
                Conv(self.coords, self.coords, kernel_size=3, stride=1, padding=1), 
                nn.Conv2d(self.coords, self.coords, kernel_size=1, stride=1)
                ),
            
            # 3rd det head
            nn.Sequential(
                Conv(int(512 * w * r), self.coords, kernel_size=3, stride=1, padding=1),
                Conv(self.coords, self.coords, kernel_size=3, stride=1, padding=1), 
                nn.Conv2d(self.coords, self.coords, kernel_size=1, stride=1)
                ),
        ])
        
        self.cls = nn.ModuleList([
            nn.Sequential(
                Conv(int(256 * w), self.nc, kernel_size=3, stride=1, padding=1),
                Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                Conv(int(512 * w), self.nc, kernel_size=3, stride=1, padding=1),
                Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                Conv(int(512 * w * r), self.nc, kernel_size=3, stride=1, padding=1),
                Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)
            ),
        ])
        
        self.dfl = DFL() 
        
    def forward(self, x, inference=False): 
        # x = output of Neck = list of 3 tensors with different resolutions and channel dimensions
        #   x[0] = [bs, ch0, w0, h0], x[1] = [bs, ch1, w1, h1], x[2] = [bs, ch2, w2, h2]
        for i in range(len(self.box)):              # det head 
            box = self.box[i](x[i])                 # [bs, num_coords, w, h]
            cls = self.cls[i](x[i])                 # [bs, num_classes, w, h]
            x[i] = torch.cat((box, cls), dim=1)     # [bs, num_coords + num_classes, w, h]
        
        # if training no DFL output
        # if self.training: 
        #     return x                                # [3, bs, num_coords + num_classes, w, h]
        if not inference: 
            return x 
        shape = x[0].shape  # (batch, channels, height, width)

        # (batch, 4*reg_max + nc, n_layers*height*width)
        x_cat = torch.cat([xi.view(shape[0], self.n_outputs, -1) for xi in x], dim=2)

        if self.shape != shape:
            print(self.strides)
            self.anchors, self.strides = make_anchors(x, self.stride)
            print(self.strides)
            self.anchors.transpose_(0, 1)
            self.strides.transpose_(0, 1)
            self.shape = shape

        # (batch, 4*reg_max, n_layers*height*width), (batch, nc, n_layers*height*width)
        box, cls = x_cat.split((4*self.reg_max, self.nc), dim=1)
        # (batch, 4, n_layers*height*width) (ltrb) -> (xywh)
        bbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), dim=1) * self.strides

        # (batch, 4+nc, n_layers*height*width)
        out = torch.cat((bbox, torch.sigmoid(cls)), dim=1)
        return out
    
    def _bias_init(self) -> None:
        """
        Initialize biases for Conv2d layers

        Must set stride before calling this method
        """
        if self.stride is None:
            raise ValueError('stride is not set')
        
        for b_list, c_list, s in zip(self.box, self.cls, self.stride):
            b_list[-1].bias.data[:] = 1.0
            c_list[-1].bias.data[:self.nc] = log(5/(self.nc*(640/s)**2))
