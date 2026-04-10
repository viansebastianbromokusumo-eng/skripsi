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

class DetectionModel_MTL(BaseModel_MTL):
    def __init__(self, config: str, verbose: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        config = config if isinstance(config, dict) else yaml.safe_load(open(config, 'r'))
        in_channels = config.get('in_channels', 3)

        self.model, self.save_idxs = parse_config(config, verbose=verbose)
        self.model.to(self.device)

        self.inplace = config.get('inplace', True)

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
            preds = self.forward(dummy_input)
            
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

    def postprocess(self, preds: torch.Tensor):
        # Preds is now (det_out, llie_out) because _predict returns a tuple
        # But wait! Look at BaseModel.predict:
        # if self.mode == 'eval':
        #    det_out, llie_out = preds
        #    det_out = self.postprocess(det_out)
        
        # So 'preds' passed HERE is actually just 'det_out'.
        # We can keep this standard NMS.
        return nms(preds)