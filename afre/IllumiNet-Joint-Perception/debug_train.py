import sys
import os
import torch
import yaml
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset 

# Setup Path
sys.path.append(os.getcwd())

# Import your modules
from model.models.detection_model import DetectionModel_MTL
from model.data.dataset import MTLDataset

# --- CONFIGURATION ---
# Use raw strings
train_config_path = r'model/config/training/fine_tune.yaml'
model_config_path = r'model/config/models/yolov8n.yaml'
yolo_weights_path = r'yolov8n.pt'
save_dir = r'runs/debug'  # <--- NEW: Where to save debug weights

def debug_train():
    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # 2. Load Config Dict
    if not os.path.exists(train_config_path):
        print("Config not found.")
        return
    
    with open(train_config_path, 'r') as f:
        train_config = yaml.safe_load(f)

    # 3. Initialize Model
    print("Initializing Model...")
    model = DetectionModel_MTL(model_config_path, verbose=True) 
    model.to(device)

    # 4. Load Pretrained YOLO Weights
    if os.path.exists(yolo_weights_path):
        print(f"Loading weights from {yolo_weights_path}...")
        
        # --- FIX 1: Open the file (Disabling security check) ---
        ckpt = torch.load(yolo_weights_path, map_location=device, weights_only=False)
        
        # --- FIX 2: Extract the State Dict ---
        state_dict = None
        
        if isinstance(ckpt, dict):
            if 'model' in ckpt:
                if hasattr(ckpt['model'], 'state_dict'):
                    state_dict = ckpt['model'].float().state_dict()
                else:
                    state_dict = ckpt['model']
            else:
                state_dict = ckpt
        elif hasattr(ckpt, 'state_dict'):
            state_dict = ckpt.float().state_dict()
            
        # --- FIX 3: Load into your MTL Model ---
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)
            print("✅ Pretrained weights loaded successfully (Partial Load).")
        else:
            print("❌ Error: Could not extract state_dict from checkpoint.")
            
    else:
        print("No weights found, initializing random weights.")
            
    # 5. Initialize Dataset & Loader
    print("Initializing Data...")
    
    # 1. Initialize full dataset
    full_dataset = MTLDataset(train_config_path, mode='train')

    # 2. Create a subset of the first 500 indices (OR LESS if dataset is smaller)
    limit = min(500, len(full_dataset))
    dataset = Subset(full_dataset, range(limit))

    print(f"Debug Mode: Loaded {len(dataset)} samples out of {len(full_dataset)}")

    # 3. Pass the SUBSET to the DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=MTLDataset.collate_fn
    )

    # 6. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-10)

    # 7. Training Loop (2 Epochs)
    print("\n--- STARTING SANITY CHECK (2 Epochs) ---")
    
    for epoch in range(2):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/2")
        
        for batch in pbar:
            loss, loss_items = model.loss(batch) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 3. GRANULAR LOGGING
            # loss_items structure: [cls, box, dfl, llie]
            # We use .item() to get the float value from the tensor
            cls_loss = loss_items[0].item()
            box_loss = loss_items[1].item()
            dfl_loss = loss_items[2].item()
            llie_loss = loss_items[3].item()
            
            # Calculate a raw "Total Detection" (sum of components) for visualization
            det_total = cls_loss + box_loss + dfl_loss
            
            # Update Progress Bar with all metrics
            pbar.set_postfix({
                'Tot':  f"{loss.item():.2f}",    # The weighted final loss
                'Det':  f"{det_total:.2f}",      # Sum of Box+Cls+DFL
                'Box':  f"{box_loss:.2f}",
                'Cls':  f"{cls_loss:.2f}",
                'DFL':  f"{dfl_loss:.2f}",
                'LLIE': f"{llie_loss:.2f}"
            })
            
    print("\nSanity Check Passed! Model trains without crashing.")

    # --- 8. SAVE MODEL ---
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'debug_mtl_model.pt')
    
    # Save the full state (Model + Optimizer + Epoch) usually better for resuming
    # But for inference check, just model state_dict is enough.
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': 2
    }, save_path)
    
    print(f"✅ Debug Model saved to: {save_path}")

if __name__ == "__main__":
    debug_train()
    

# import sys
# import os
# import torch
# import yaml
# import argparse
# from tqdm import tqdm
# from torch.utils.data import DataLoader

# # Setup Path
# sys.path.append(os.getcwd())

# # Import your modules (Now that __init__ is fixed, this is safe)
# from model.models.detection_model import DetectionModel_MTL  # Or DetectionModel_MTL if you renamed it
# from model.data.dataset import MTLDataset

# # --- CONFIGURATION ---
# # Use raw strings
# train_config_path = r'model/config/training/fine_tune.yaml'
# model_config_path = r'model/config/models/yolov8n.yaml'
# yolo_weights_path = r'model\weights\yolov8n.pt' # Ensure this file exists!

# def debug_train():
#     # 1. Setup Device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Running on: {device}")

#     # 2. Load Config Dict (for learning rate reading)
#     # Check if file exists first
#     if not os.path.exists(train_config_path):
#         print("Config not found.")
#         return
    
#     with open(train_config_path, 'r') as f:
#         train_config = yaml.safe_load(f)

#     # 3. Initialize Model
#     print("Initializing Model...")
#     # NOTE: Ensure your class name matches what is in detection_model.py
#     model = DetectionModel_MTL(model_config_path, verbose=True) 
#     model.to(device)

#     # 4. Load Pretrained YOLO Weights
#     # if os.path.exists(yolo_weights_path):
#     #     print(f"Loading weights from {yolo_weights_path}...")
#     #     model.load(torch.load(yolo_weights_path, map_location=device))
#     # else:
#     #     print("No weights found, initializing random weights.")

#     if os.path.exists(yolo_weights_path):
#             print(f"Loading weights from {yolo_weights_path}...")
            
#             # --- FIX 1: Open the file (Disabling security check) ---
#             # We MUST use weights_only=False because standard YOLO .pt files contain 
#             # the full class definition, not just weights.
#             ckpt = torch.load(yolo_weights_path, map_location=device, weights_only=False)
            
#             # --- FIX 2: Extract the State Dict ---
#             state_dict = None
            
#             # Case A: Checkpoint is a dictionary (Standard YOLOv8 format)
#             if isinstance(ckpt, dict):
#                 # The weights are usually stored under 'model'
#                 if 'model' in ckpt:
#                     # 'model' might be the full object (DetectionModel) or a state_dict
#                     if hasattr(ckpt['model'], 'state_dict'):
#                         state_dict = ckpt['model'].float().state_dict()
#                     else:
#                         state_dict = ckpt['model']
#                 else:
#                     # Maybe the dict IS the state_dict
#                     state_dict = ckpt
                    
#             # Case B: Checkpoint is the model object directly
#             elif hasattr(ckpt, 'state_dict'):
#                 state_dict = ckpt.float().state_dict()
                
#             # --- FIX 3: Load into your MTL Model ---
#             if state_dict is not None:
#                 # We use strict=False because:
#                 # 1. Your model has an extra layer (Index 19: LLIE Decoder) -> Missing in weights
#                 # 2. Your model might have slight differences -> We want to load what matches
#                 model.load_state_dict(state_dict, strict=False)
#                 print("✅ Pretrained weights loaded successfully (Partial Load).")
#             else:
#                 print("❌ Error: Could not extract state_dict from checkpoint.")
                
#     else:
#         print("No weights found, initializing random weights.")
            
#     # 5. Initialize Dataset & Loader
#     print("Initializing Data...")
#     # FIX: Pass the PATH string, not the dict object
#     from torch.utils.data import Subset # <--- Add this import

#     # 1. Initialize full dataset
#     full_dataset = MTLDataset(train_config_path, mode='train')

#     # 2. Create a subset of the first 500 indices
#     # range(500) creates indices [0, 1, ... 499]
#     dataset = Subset(full_dataset, range(500))

#     print(f"Debug Mode: Loaded {len(dataset)} samples out of {len(full_dataset)}")

#     # 3. Pass the SUBSET to the DataLoader
#     dataloader = DataLoader(
#         dataset, 
#         batch_size=2, 
#         shuffle=True, 
#         collate_fn=MTLDataset.collate_fn
#     )

#     # 6. Optimizer
#     optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-10)

#     # 7. Training Loop (2 Epochs)
#     print("\n--- STARTING SANITY CHECK (2 Epochs) ---")
    
#     for epoch in range(2):
#         model.train()
#         pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/2")
        
#         for batch in pbar:
#             # Move data to device (Model.loss handles 'img', but we need to ensure 'gt_img' etc are ready if needed)
#             # Actually, let's let the loss function handle the .to(device) calls to keep loop clean.
            
#             # Forward + Loss
#             # Returns tuple: (total_loss, loss_items_vector)
#             loss, loss_items = model.loss(batch) 
            
#             # Backward
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             # Update Progress Bar
#             # loss_items is [box, cls, dfl, llie]
#             current_loss = loss.item()
#             llie_loss = loss_items[-1].item()
#             pbar.set_postfix({'Total': f"{current_loss:.2f}", 'LLIE': f"{llie_loss:.2f}"})

#     print("\nSanity Check Passed! Model trains without crashing.")

# if __name__ == "__main__":
#     debug_train()