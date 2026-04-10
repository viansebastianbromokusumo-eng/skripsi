import re
from ultralytics import YOLO

def remap_and_load(yolo_state, target_model, mapping, target_state_name=""):
    """
    Remaps YOLOv8 state_dict keys to your custom module keys and loads them safely.
    
    Args:
        yolo_state (dict): YOLOv8 state_dict.
        target_model (nn.Module): Your model module (e.g., backbone, neck, head).
        mapping (dict): Dictionary mapping YOLO key prefixes → your model key prefixes.
        target_state_name (str): Optional name for logging (e.g., 'Backbone', 'Neck', 'Head').
    """
    remapped_state = {}
    target_state = target_model.state_dict()

    for k, v in yolo_state.items():
        for yolo_prefix, my_prefix in mapping.items():
            if k.startswith(yolo_prefix):
                new_k = k.replace(yolo_prefix, my_prefix)

                # Normalize nested layer naming (cv1 -> conv1, etc.)
                new_k = re.sub(r'\.cv1\.', '.conv1.', new_k)
                new_k = re.sub(r'\.cv2\.', '.conv2.', new_k)
                new_k = re.sub(r'\.cv3\.', '.conv3.', new_k)

                # Only keep weights that match by name & shape
                if new_k in target_state and target_state[new_k].shape == v.shape:
                    remapped_state[new_k] = v
                break

    missing, unexpected = target_model.load_state_dict(remapped_state, strict=False)
    
    print(f"✅ Loaded {len(remapped_state)} layers into {target_state_name}.")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    
    return remapped_state


yolo = YOLO('yolov8n.pt')

yolo_state = yolo.model.state_dict() 

# for k in list(yolo_state.keys()): 
#     print(k)
    
print(len(list(yolo_state.keys())))

# --- Define all mappings ---
backbone_mapping = {
    'model.0.': 'conv0.',
    'model.1.': 'conv1.',
    'model.2.': 'c2f_2.',
    'model.3.': 'conv3.',
    'model.4.': 'c2f_4.',
    'model.5.': 'conv5.',
    'model.6.': 'c2f_6.',
    'model.7.': 'conv7.',
    'model.8.': 'c2f_8.',
    'model.9.': 'sppf.'
}

neck_mapping = {
    'model.12.': 'c2f_1.',
    'model.15.': 'c2f_2.',
    'model.16.': 'conv_1.',
    'model.18.': 'c2f_3.',
    'model.19.': 'conv_2.',
    'model.21.': 'c2f_4.',
}

head_mapping = {
    "model.22.cv2.": "box.",
    "model.22.cv3.": "cls.",
    "model.22.dfl.": "dfl.",
}

# backbone_n_5 = BackBone_5(version='nano')
# neck = Neck(version='nano')
# head = Head_2(version='nano')

# # --- Apply to each module ---
# backbone_weights = remap_and_load(yolo_state, backbone_n_5, backbone_mapping, "Backbone")
# neck_weights     = remap_and_load(yolo_state, neck, neck_mapping, "Neck")
# head_weights     = remap_and_load(yolo_state, head, head_mapping, "Head")

# print(len(list(backbone_weights.keys())) + len(list(neck_weights.keys())) + len(list(head_weights.keys())))