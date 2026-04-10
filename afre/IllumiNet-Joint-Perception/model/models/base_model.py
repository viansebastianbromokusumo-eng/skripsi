import torch
import torch.nn as nn

from model.modules import Conv, C2f, SPPF, DetectionHead, LowLightEnhancementDecoder_Yol
from model.utils.loss import BaseLoss

from typing import Union

class BaseModel(nn.Module):
    model:nn.ModuleList
    save_idxs:set
    loss_fn:BaseLoss

    def __init__(self, mode='train', device='cpu'):
        super().__init__()

        assert mode in ('train', 'eval'), f'Invalid mode: {mode}'
        self.mode = mode

        self.device = device

        self.model = None
        self.save_idxs = set()

    def load(self, weights:Union[dict, nn.Module]):
        state_dict = weights.float().state_dict() if isinstance(weights, nn.Module) else weights
        self.load_state_dict(state_dict)

    def forward(self, x:torch.Tensor, *args, **kwargs):
        return self.predict(x, *args, **kwargs)

    def predict(self, x:torch.Tensor, *args, **kwargs):
        preds = self._predict(x, *args, **kwargs)
        return self.postprocess(preds) if self.mode == 'eval' else preds
    
    def _predict(self, x:torch.Tensor, *args, **kwargs):
        outputs = []
        for module in self.model:
            # If not just using previous module output
            if module.f != -1:
                # Get list of inputs for module
                x = outputs[module.f] if isinstance(module.f, int) else [x if i == -1 else outputs[i] for i in module.f]
                # Don't concat if module is DetectionHead (it takes in a list)
                if isinstance(x, list) and not isinstance(module, DetectionHead):
                    x = torch.cat(x, dim=1)
            x = module(x)

            outputs.append(x if module.i in self.save_idxs else None)

        return x
    
    def loss(self, batch:torch.Tensor):
        preds = self.forward(batch['images'].to(self.device))
        return self.loss_fn.compute_loss(batch, preds)
    
    def postprocess(self, preds:torch.Tensor):
        return preds

    def save(self, path:str):
        torch.save(self.state_dict(), path)
     
        
class BaseModel_MTL(nn.Module):
    model: nn.ModuleList
    save_idxs: set

    def __init__(self, mode='train', device='cpu'):
        super().__init__()
        assert mode in ('train', 'eval'), f'Invalid mode: {mode}'
        self.mode = mode
        self.device = device
        self.model = None
        self.save_idxs = set()

    # def load(self, weights):
    #     # Allow loading partial weights (strict=False) so we can load YOLO weights 
    #     # into the MTL model without crashing on the missing LLIE keys.
    #     state_dict = weights.float().state_dict() if isinstance(weights, nn.Module) else weights
    #     try:
    #         self.load_state_dict(state_dict, strict=False) 
    #         print("Weights loaded successfully (strict=False).")
    #     except Exception as e:
    #         print(f"Warning during loading: {e}")
    
    def load(self, weights):
        """
        Smart loading that handles Index Mismatches between Official YOLOv8 and MTL Model.
        """
        # 1. Load Checkpoint
        if isinstance(weights, str) and os.path.exists(weights):
            ckpt = torch.load(weights, map_location=self.device)
            state_dict = ckpt['model'].float().state_dict() if 'model' in ckpt else ckpt
        else:
            state_dict = weights

        # 2. Get Your Model's State Dict
        my_dict = self.state_dict()
        
        # 3. Smart Key Matching
        new_state_dict = {}
        matched_keys = 0
        
        # Detect the Head Index in YOUR model (Likely 18)
        my_head_idx = None
        for key in my_dict.keys():
            if 'dfl' in key or 'cv2' in key and 'model.18' in key: # Heuristic for head
                my_head_idx = int(key.split('.')[1])
                break
        
        # Detect Head Index in PRETRAINED model (Likely 22)
        pt_head_idx = 22 # Standard for YOLOv8n
        
        print(f"DEBUG: Smart Load - Mapping Pretrained Head {pt_head_idx} -> My Head {my_head_idx}")

        for k, v in state_dict.items():
            # Standard copy
            if k in my_dict and v.shape == my_dict[k].shape:
                new_state_dict[k] = v
                matched_keys += 1
            
            # Handle Head Remapping (e.g., model.22 -> model.18)
            elif k.startswith(f'model.{pt_head_idx}.') and my_head_idx is not None:
                new_key = k.replace(f'model.{pt_head_idx}.', f'model.{my_head_idx}.')
                if new_key in my_dict and v.shape == my_dict[new_key].shape:
                    new_state_dict[new_key] = v
                    matched_keys += 1
                    
        # 4. Load
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Smart Weights Loaded: {matched_keys} keys matched.")
        if len(missing) > 0:
            print(f"   (LLIE keys expected missing. Detection Head should be loaded now.)")
            
    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.predict(x, *args, **kwargs)

    def predict(self, x: torch.Tensor, *args, **kwargs):
        # 1. Get raw model output (Tuple of det_out, llie_out)
        preds = self._predict(x, *args, **kwargs)
        
        # 2. If in EVAL mode, apply post-processing to Detection only
        if self.mode == 'eval':
            det_out, llie_out = preds
            det_out = self.postprocess(det_out) # Decode boxes & NMS
            return det_out, llie_out
        
        # 3. In TRAIN mode, return raw predictions for loss calculation
        return preds

    def _predict(self, x: torch.Tensor, *args, **kwargs):
        """Internal prediction loop that returns raw layer outputs."""
        outputs = []
        det_out = None
        llie_out = None

        for module in self.model:
            # Input Gathering Logic
            if module.f != -1:
                x = outputs[module.f] if isinstance(module.f, int) else [x if i == -1 else outputs[i] for i in module.f]
                
                # MTL Specific: Do not concat for specific heads if they need lists
                if isinstance(x, list) and not isinstance(module, (DetectionHead, LowLightEnhancementDecoder_Yol)):
                    x = torch.cat(x, dim=1)
            
            # Run Module
            x = module(x)
            outputs.append(x if module.i in self.save_idxs else None)

            # Capture Specific Outputs
            if isinstance(module, DetectionHead):
                det_out = x
            elif isinstance(module, LowLightEnhancementDecoder_Yol):
                llie_out = x

        # Safety fallback
        if det_out is None and llie_out is None:
            return x

        return det_out, llie_out
    
    def loss(self, batch: torch.Tensor):
        current_device = next(self.parameters()).device
        preds = self.forward(batch['img'].to(current_device))
        return self.loss_fn.compute_loss(batch, preds)
    
    def postprocess(self, preds: torch.Tensor):
        """Default postprocess (can be overridden)"""
        return preds

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        
                
# class BaseModel_MTL(nn.Module):
#     model:nn.ModuleList
#     save_idxs:set
#     loss_fn:BaseLoss

#     def __init__(self, mode='train', device='cpu'):
#         super().__init__()

#         assert mode in ('train', 'eval'), f'Invalid mode: {mode}'
#         self.mode = mode

#         self.device = device

#         self.model = None
#         self.save_idxs = set()

#     # def load(self, weights:Union[dict, nn.Module]):
#     #     state_dict = weights.float().state_dict() if isinstance(weights, nn.Module) else weights
#     #     self.load_state_dict(state_dict)

#     def load(self, weights):
#             # Allow loading partial weights (strict=False) so we can load YOLO weights 
#             # into the MTL model without crashing on the missing LLIE keys.
#             state_dict = weights.float().state_dict() if isinstance(weights, nn.Module) else weights
#             try:
#                 self.load_state_dict(state_dict, strict=False) 
#                 print("Weights loaded successfully (strict=False).")
#             except Exception as e:
#                 print(f"Warning during loading: {e}")
            
#     def forward(self, x:torch.Tensor, *args, **kwargs):
#         return self.predict(x, *args, **kwargs)

#     def predict(self, x: torch.Tensor, *args, **kwargs):
#         # Changed: helper to unpack tuple
#         preds = self._predict(x, *args, **kwargs)
        
#         # If we are in EVAL mode, we postprocess ONLY the detection part
#         if self.mode == 'eval':
#             det_out, llie_out = preds
#             det_out = self.postprocess(det_out)
#             return det_out, llie_out
        
#         # In TRAIN mode, return raw predictions for loss calculation
#         return preds

#     def _predict(self, x: torch.Tensor, *args, **kwargs):
#         outputs = []
#         det_out = None
#         llie_out = None

#         for module in self.model:
#             # 1. Input Gathering Logic
#             if module.f != -1:
#                 x = outputs[module.f] if isinstance(module.f, int) else [x if i == -1 else outputs[i] for i in module.f]
                
#                 # CRITICAL CHANGE HERE: 
#                 # Add LowLightEnhancementDecoder to the "Do Not Concat" list.
#                 # It needs the list [stem, dark2, dark3], not a fused tensor.
#                 if isinstance(x, list) and not isinstance(module, (DetectionHead, LowLightEnhancementDecoder_Yol)):
#                     x = torch.cat(x, dim=1)
            
#             # 2. Run Module
#             x = module(x)
#             outputs.append(x if module.i in self.save_idxs else None)

#             # 3. Capture Specific Outputs
#             # We explicitly look for our two heads. 
#             if isinstance(module, DetectionHead):
#                 det_out = x
#             elif isinstance(module, LowLightEnhancementDecoder_Yol):
#                 llie_out = x

#         # Safety fallback: if we didn't find specific heads (e.g. testing just backbone), return last x
#         if det_out is None and llie_out is None:
#             return x

#         # Return tuple of (Detection, LLIE)
#         return det_out, llie_out
    
#     def loss(self, batch:torch.Tensor):
#         # preds = self.forward(batch['img'].to(self.device))
#         # New
#         # Use the device of the first parameter in the model
#         current_device = next(self.parameters()).device
#         preds = self.forward(batch['img'].to(current_device))
#         return self.loss_fn.compute_loss(batch, preds)
    
#     def postprocess(self, preds:torch.Tensor):
#         return preds

#     def save(self, path:str):
#         torch.save(self.state_dict(), path)
        
         