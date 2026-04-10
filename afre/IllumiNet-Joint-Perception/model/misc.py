# import logging

# import torch
# import torch.nn as nn

# from .modules import Conv, C2f, SPPF, DetectionHead, LowLightEnhancementDecoder_Yol

# from typing import Tuple

# def parse_config(config_dict:dict, verbose=False) -> Tuple[nn.Module, set]:
#     if verbose:
#         log = logging.getLogger("yolo")
#         logging.basicConfig(level=logging.INFO)
    
#     depth, width, max_channels = config_dict['scale']

#     num_classes = config_dict['num_classes']

#     channels = [config_dict.get('in_channels', 3)]

#     modules = []
#     save_idxs = set()

#     if verbose:
#         log.info(f'{"idx":>4} | {"Module Type":>14} | {"Input idx(s)":>12} | Args')
#         log.info('-'*60)

#     # Loop through backbone and head layers
#     for i, (module, f, r, args) in enumerate(config_dict['backbone']+config_dict['head']):
#         module = getattr(torch.nn, module[3:]) if module.startswith('nn.') else globals()[module]
#         if module in (Conv, C2f, SPPF):
#             # Get input/output channel sizes
#             c_in = channels[f] if isinstance(f, int) else sum([channels[idx] for idx in f])
#             c_out = args[0]

#             if c_out != num_classes:
#                 c_out = int(min(c_out, max_channels) * width)

#             if module == C2f:
#                 args = [c_in, c_out, max(round(r*depth), 1), *args[1:]]
#             else:
#                 args = [c_in, c_out, *args[1:]]

#         elif module in (DetectionHead,):
#             args.append([channels[idx] for idx in f])

#         if verbose:
#             log.info(f'{i:>4} | {module.__name__:>14} | {str(f):>12} | {args}')

#         m_ = module(*args)
#         modules.append(m_)
#         m_.i, m_.f = i, f

#         save_idxs.update([f] if isinstance(f, int) else f)

#         # Remove initial channel amount
#         # (only needed for first Conv layer)
#         if i == 0:
#             channels = []
#         channels.append(c_out)

#     save_idxs.remove(-1)

#     if verbose:
#         log.info(f' Will save at indices: {save_idxs}')

#     return nn.Sequential(*modules), save_idxs

import logging
import torch
import torch.nn as nn
from typing import Tuple

# Ensure these imports are correct based on your file structure
from .modules import Conv, C2f, SPPF, DetectionHead, LowLightEnhancementDecoder_Yol



# def parse_config(config_dict: dict, verbose=False) -> Tuple[nn.Module, set]:
#     if verbose:
#         log = logging.getLogger("yolo")
#         logging.basicConfig(level=logging.INFO)
    
#     depth, width, max_channels = config_dict['scale']
#     num_classes = config_dict['num_classes']
    
#     channels = [config_dict.get('in_channels', 3)]
#     modules = []
#     save_idxs = set()

#     if verbose:
#         log.info(f'{"idx":>4} | {"Module Type":>25} | {"Input idx(s)":>12} | Args')
#         log.info('-' * 80)

#     layers = config_dict['backbone'] + config_dict['head']

#     for i, (module_name, f, r, args) in enumerate(layers):
#         # 1. RESOLVE MODULE CLASS
#         if module_name.startswith('nn.'):
#             module = getattr(torch.nn, module_name[3:])
#         elif module_name == 'LowLightEnhancementDecoder_Yol':
#             module = LowLightEnhancementDecoder_Yol
#         else:
#             try:
#                 module = globals()[module_name]
#             except KeyError:
#                 raise KeyError(f"Class '{module_name}' not found. Check imports in model/misc.py")

#         def get_ch(idx):
#             # If idx is -1, get the last added output
#             if idx == -1: return channels[-1]
#             # If idx is 0 or positive, we usually want the OUTPUT of that layer.
#             # channels list structure: [input_img, layer0_out, layer1_out...]
#             # So layer 0 output is at index 1.
#             return channels[idx + 1]
        
#         # 2. CALCULATE INPUT CHANNELS & ARGS
        
#         # --- CASE A: Standard YOLO Modules (Conv, C2f, SPPF) ---
#         if module in (Conv, C2f, SPPF):
#             c_in = channels[f] if isinstance(f, int) else sum([channels[idx] for idx in f])
#             c_out = args[0]
#             if c_out != num_classes:
#                 c_out = int(min(c_out, max_channels) * width)

#             if module == C2f:
#                 args = [c_in, c_out, max(round(r * depth), 1), *args[1:]]
#             else:
#                 args = [c_in, c_out, *args[1:]]

#         # --- CASE B: Detection Head ---
#         elif module in (DetectionHead,):
#             args.append([channels[idx] for idx in f])
#             # The head doesn't really output channels for the next layer in a standard way, 
#             # but we can set it to null or the last output.
#             c_out = None 

#         # --- CASE C: LLIE Decoder ---
#         elif module == LowLightEnhancementDecoder_Yol:
#             c_in_list = [channels[idx] for idx in f]
#             args = [c_in_list, *args]
#             c_out = 3 

#         # --- CASE D: NN.UPSAMPLE (The Fix) ---
#         elif module is nn.Upsample:
#             # Upsample preserves channels.
#             # args from YAML: [None, 2, 'nearest'] -> Matches (size, scale_factor, mode)
#             # We do NOT inject c_in/c_out.
#             c_out = channels[f]
#             # args are passed as-is to nn.Upsample constructor

#         # --- CASE E: Default Fallback ---
#         else:
#             c_out = args[0]
#             if c_out is None:
#                 c_out = channels[f] # Fallback: preserve channels
#             args = [channels[f], c_out, *args[1:]]

#         # 3. INSTANTIATE MODULE
#         m_ = module(*args)
#         m_.i, m_.f = i, f
#         modules.append(m_)

#         # 4. UPDATE CHANNEL TRACKING
#         save_idxs.update([f] if isinstance(f, int) else f)
#         channels.append(c_out)

#         if verbose:
#             # Clean up args string for printing
#             print_args = args if module is not nn.Upsample else args
#             log.info(f'{i:>4} | {module_name:>25} | {str(f):>12} | {print_args}')

#     save_idxs.discard(-1)

#     if verbose:
#         log.info(f' Will save at indices: {save_idxs}')

#     return nn.Sequential(*modules), save_idxs

def parse_config(config_dict: dict, verbose=False) -> Tuple[nn.Module, set]:
    if verbose:
        log = logging.getLogger("yolo")
        logging.basicConfig(level=logging.INFO)
    
    depth, width, max_channels = config_dict['scale']
    num_classes = config_dict['num_classes']
    
    # channels[0] = Input Image (3)
    # channels[1] = Output of Layer 0
    # channels[2] = Output of Layer 1 ... etc
    channels = [config_dict.get('in_channels', 3)]

    modules = []
    save_idxs = set()

    if verbose:
        log.info(f'{"idx":>4} | {"Module Type":>30} | {"Input idx(s)":>12} | Args')
        log.info('-' * 100)

    layers = config_dict['backbone'] + config_dict['head']

    for i, (module_name, f, r, args) in enumerate(layers):
        # 1. RESOLVE MODULE CLASS
        if module_name.startswith('nn.'):
            module = getattr(torch.nn, module_name[3:])
        elif module_name == 'LowLightEnhancementDecoder_Yol':
            module = LowLightEnhancementDecoder_Yol
        else:
            try:
                module = globals()[module_name]
            except KeyError:
                raise KeyError(f"Class '{module_name}' not found. Check imports in model/misc.py")

        # --- HELPER: Safely get output channels from previous layers ---
        def get_ch(idx):
            # If idx is -1, get the last added output
            if idx == -1: 
                return channels[-1]
            # If idx is 0 or positive, we want the OUTPUT of that layer.
            # Since channels[0] is the input image, Layer N's output is at channels[N+1].
            return channels[idx + 1]

        # 2. CALCULATE INPUT CHANNELS & ARGS
        
        # --- CASE A: Standard YOLO Modules (Conv, C2f, SPPF) ---
        if module in (Conv, C2f, SPPF):
            # Use get_ch() to find input size
            c_in = get_ch(f) if isinstance(f, int) else sum([get_ch(idx) for idx in f])
            
            c_out = args[0]
            if c_out != num_classes:
                c_out = int(min(c_out, max_channels) * width)

            if module == C2f:
                args = [c_in, c_out, max(round(r * depth), 1), *args[1:]]
            else:
                args = [c_in, c_out, *args[1:]]

        # --- CASE B: Detection Head ---
        elif module in (DetectionHead,):
            # Pass list of input channels to the head
            args.append([get_ch(idx) for idx in f])
            c_out = None 

        # --- CASE C: LLIE Decoder ---
        elif module == LowLightEnhancementDecoder_Yol:
            # Pass list of input channels to the decoder
            c_in_list = [get_ch(idx) for idx in f]
            args = [c_in_list, *args]
            # Decoder outputs an RGB image, so 3 channels
            c_out = 3 

        # --- CASE D: NN.UPSAMPLE ---
        elif module is nn.Upsample:
            # Upsample preserves channel count
            c_out = get_ch(f)
            # args are passed as-is (e.g., [None, 2, 'nearest'])

        # --- CASE E: Default Fallback ---
        else:
            c_out = args[0]
            if c_out is None:
                c_out = get_ch(f) # Fallback: preserve channels
            args = [get_ch(f), c_out, *args[1:]]

        # 3. INSTANTIATE MODULE
        m_ = module(*args)
        m_.i, m_.f = i, f
        modules.append(m_)

        # 4. UPDATE CHANNEL TRACKING
        save_idxs.update([f] if isinstance(f, int) else f)
        channels.append(c_out)

        if verbose:
            log.info(f'{i:>4} | {module_name:>30} | {str(f):>12} | {args}')

    save_idxs.discard(-1)

    if verbose:
        log.info(f' Will save at indices: {save_idxs}')

    return nn.Sequential(*modules), save_idxs

