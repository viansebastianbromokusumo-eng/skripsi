import torch
import torch.nn as nn
from modules.yolo_components import BackBone_5_ARB, Neck, Head_2
from modules.decoder import LowLightEnhancementDecoder
from utils.load_pretrained import remap_and_load, backbone_mapping, neck_mapping, head_mapping, yolo_state


class IllumiNet_b5_ARB(nn.Module): 
    def __init__(self, back_use_arb, dec_arb):
        super().__init__()
        # print('h/2, h/4, h/8 Framework initialized')
        print('Backbone ARB: ', back_use_arb)
        print('Decoder ARB', dec_arb)
        self.backbone = BackBone_5_ARB(version='nano', use_arb=back_use_arb)
        self.decoder = LowLightEnhancementDecoder(base_channels=64, use_arb=dec_arb)
        self.neck = Neck(version='nano')
        self.head = Head_2(version='nano')
        
        self.head.stride = torch.tensor([8., 16., 32.])
        
        self.backbone_w = remap_and_load(yolo_state, self.backbone, backbone_mapping, "Backbone")
        self.neck_w = remap_and_load(yolo_state, self.neck, neck_mapping, "Neck")
        self.head_w = remap_and_load(yolo_state, self.head, head_mapping, "Head")
        
        self.head._bias_init()
        
        print('pretrain weights loaded')
        
    
    def forward(self, x, inference): 
        stem, dark2, dark3, dark4, out = self.backbone(x)
        
        llie_out, llie_residuals = self.decoder(in_img=x, stem=stem, dark2=dark2, dark3=dark3)
        
        neck_out1, neck_out2, neck_out3 = self.neck(dark3, dark4, out)
        
        det_out = self.head([neck_out1, neck_out2, neck_out3], inference=inference)
        
        return det_out, llie_out, llie_residuals