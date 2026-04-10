import sys
import os
import torch
import yaml
import argparse
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

# Ensure root path is in sys.path
sys.path.append(os.getcwd())

# Import your modules
from model.models.detection_model import DetectionModel_MTL
from model.data.dataset import MTLDataset

# Setup Logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def set_freeze_status(model, freeze_backbone=False):
    """
    Freezes or unfreezes layers 0-9 (Backbone).
    """
    # 1. First, set everything to Trainable
    for p in model.parameters():
        p.requires_grad = True

    # 2. If freeze requested, lock the first 10 layers (Backbone)
    if freeze_backbone:
        logger.info("🔒 Freezing Backbone Layers (0-9)")
        for i, m in enumerate(model.model):
            if i < 10: 
                for p in m.parameters():
                    p.requires_grad = False
    else:
        logger.info("🔓 Unfreezing Backbone - Full Model Training")

def main(args):
    # 1. Setup Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running on: {device}")

    # 2. Load Configuration
    if not os.path.exists(args.train_config):
        logger.error(f"Config not found at {args.train_config}")
        return
    
    with open(args.train_config, 'r') as f:
        train_cfg = yaml.safe_load(f)

    # 3. Initialize Model
    logger.info("Initializing Model...")
    model = DetectionModel_MTL(args.model_config, verbose=True)
    model.to(device)
    model.device = device

    # 4. Load Pretrained Weights (Robust Method)
    if args.weights and os.path.exists(args.weights):
        logger.info(f"Loading weights from {args.weights}...")
        
        # Open file (Disable security check for local files)
        ckpt = torch.load(args.weights, map_location=device, weights_only=False)
        
        # Extract State Dict
        state_dict = None
        if isinstance(ckpt, dict):
            if 'model' in ckpt:
                state_dict = ckpt['model'].float().state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
            else:
                state_dict = ckpt
        elif hasattr(ckpt, 'state_dict'):
            state_dict = ckpt.float().state_dict()
            
        # Load into MTL Model (Strict=False for partial load)
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)
            logger.info("✅ Pretrained weights loaded successfully (Partial Load).")
        else:
            logger.error("❌ Error: Could not extract state_dict from checkpoint.")
    else:
        logger.warning("⚠️ No weights found! Training from scratch.")

   # 5. Setup Parameter Groups (Differential Learning Rates)
    # Group 1: Backbone (Layers 0-9)           -> Lower LR (0.1x)
    # Group 2: Detection Head + Neck (10-18)   -> Base LR  (1.0x)
    # Group 3: LLIE Decoder (Layer 19)         -> Base LR  (1.0x) - Can be tuned separately!
    
    # Import the decoder class to identify it robustly
    from model.modules.llie_dec import LowLightEnhancementDecoder_Yol

    backbone_params = []
    det_params = []
    llie_params = []

    for i, m in enumerate(model.model):
        if i < 10: 
            # Backbone (Layers 0-9)
            backbone_params.extend(m.parameters())
        elif isinstance(m, LowLightEnhancementDecoder_Yol):
            # LLIE Decoder (Specific Class)
            llie_params.extend(m.parameters())
        else:
            # Neck & Detection Head (Layers 10-18)
            det_params.extend(m.parameters())

    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5,     'name': 'backbone'}, # Low LR
        {'params': det_params,      'lr': 1e-4,     'name': 'detection'},# Base LR
        {'params': llie_params,     'lr': 1e-3,     'name': 'llie'}      # Base LR
        
    ], weight_decay=train_cfg.get('weight_decay', 1e-10))
    print('Separate LR initialized')

    # 6. Scheduler (Cosine Annealing)
    epochs = train_cfg['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # 7. Data Loaders
    logger.info("Initializing Data...")
    full_dataset = MTLDataset(args.train_config, mode='train')
    
    # Optional Debug Mode: Use only 500 samples
    if args.debug:
        logger.warning("🐛 DEBUG MODE: Using only first 500 samples!")
        limit = min(500, len(full_dataset))
        dataset = Subset(full_dataset, range(limit))
    else:
        dataset = full_dataset

    dataloader = DataLoader(
        dataset, 
        batch_size=train_cfg['batch_size'], 
        shuffle=True, 
        collate_fn=MTLDataset.collate_fn,
        num_workers=4, 
        pin_memory=True
    )
    
    logger.info(f"Loaded {len(dataset)} samples for training.")

    # 8. Training Loop
    start_epoch = 0
    freeze_epochs = 5 # Freeze backbone for first 10 epochs
    save_dir = train_cfg.get('save_dir', 'runs/train')
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"\n--- STARTING TRAINING ({epochs} Epochs) ---")

    for epoch in range(start_epoch, epochs):
        # --- FREEZING STRATEGY ---
        # Freeze at start, Unfreeze after N epochs
        if epoch == 0:
            set_freeze_status(model, freeze_backbone=True)
        elif epoch == freeze_epochs:
            set_freeze_status(model, freeze_backbone=False)
        
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            # Forward + Loss
            loss, loss_items = model.loss(batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            # --- GRANULAR LOGGING ---
            # loss_items structure: [cls, box, dfl, llie]
            cls_loss = loss_items[0].item()
            box_loss = loss_items[1].item()
            dfl_loss = loss_items[2].item()
            llie_loss = loss_items[3].item()
            
            # Det Total for visualization
            det_total = cls_loss + box_loss + dfl_loss
            
            # Update Progress Bar
            pbar.set_postfix({
                'Tot':  f"{loss.item():.2f}", 
                'Det':  f"{det_total:.2f}", 
                'Box':  f"{box_loss:.2f}",
                'Cls':  f"{cls_loss:.2f}",
                'DFL':  f"{dfl_loss:.2f}",
                'LLIE': f"{llie_loss:.2f}",
                'LR_back':   f"{optimizer.param_groups[0]['lr']:.5f}",
                'LR_det':   f"{optimizer.param_groups[1]['lr']:.5f}",
                'LR_LLIE':   f"{optimizer.param_groups[2]['lr']:.5f}"
                
            })

        # Step Scheduler at end of epoch
        scheduler.step()
        
        # Save Checkpoint
        save_freq = train_cfg.get('save_freq', 5)
        if (epoch + 1) % save_freq == 0 or (epoch + 1) == epochs:
            save_path = os.path.join(save_dir, f"mtl_epoch_{epoch+1}.pt")
            
            # Save comprehensive checkpoint
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': train_cfg
            }, save_path)
            
            logger.info(f"Checkpoint saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default=r'model/config/models/yolov8n.yaml', help='path to model config')
    parser.add_argument('--train_config', type=str, default=r'model/config/training/fine_tune.yaml', help='path to training config')
    parser.add_argument('--weights', type=str, default=r'model/weights/yolov8n.pt', help='initial weights path')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--debug', action='store_true', help='Use subset of data for debugging')
    
    args = parser.parse_args()
    main(args)