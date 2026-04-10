import sys
import os

# 1. Force the current directory into Python path to resolve 'model' package
sys.path.append(os.getcwd())

print("Step 1: Importing Torch...")
import torch
print(f"Torch version: {torch.__version__}")

print("Step 2: Importing Dataset class...")
# We use a try/except block to catch the import error cleanly
try:
    # NOTE: Try changing this import to be specific to avoid triggering the whole package
    from model.data.dataset import MTLDataset
    print("Dataset class imported successfully.")
except ImportError as e:
    print("\nCRITICAL IMPORT ERROR:")
    print(e)
    print("\nSUGGESTION: Check your 'model/__init__.py'. It is likely trying to load everything at once.")
    sys.exit(1)

# --- CONFIGURATION ---
# Use raw strings (r'...') for Windows paths
train_config_path = r'model/config/training/fine_tune.yaml'
model_config = 'model\config\models\yolov8n.yaml'
train_config = 'model\config\training\fine_tune.yaml'

data_train = 'datasets\archive\train'
metadata_train = 'datasets\archive\train_metadata.csv'

data_val = 'datasets\archive\val'
metadata_val = 'datasets\archive\val_metadata.csv'

data_test = 'datasets\archive\test'
metadata_test = 'datasets\archive\test_metadata.csv'

def test_pipeline():
    print("\nStep 3: Initializing Dataset...")
    if not os.path.exists(train_config_path):
        print(f"Error: Config file not found at {train_config_path}")
        return

    # Initialize Dataset
    dataset = MTLDataset(config=train_config_path, mode='train')
    print(f"Dataset initialized. Found {len(dataset)} pairs.")

    print("Step 4: Testing DataLoader...")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=MTLDataset.collate_fn
    )

    # Fetch one batch
    batch = next(iter(loader))
    
    print("\n--- BATCH SUCCESS ---")
    print(f"Keys: {batch.keys()}")
    if 'img' in batch:
        print(f"Input Shape: {batch['img'].shape}")
    if 'gt_img' in batch:
        print(f"GT Shape:    {batch['gt_img'].shape}")


def main(args):
    from model.models.detection_model import DetectionModel_MTL
    from tqdm import trange
    from torch.utils.data import DataLoader
    import yaml
    
    yolo_weights = r'C:\Users\vian8\Desktop\Tugas2\LLIE\yol_mtl\yolov8\yolov8n.pt'
    
    train_config = yaml.safe_load(open(train_config_path, 'r'))

    device = torch.device(args.device)
    model = DetectionModel_MTL(args.model_config, device=device)
    
    # if args.weights is not None:
        # model.load(torch.load(args.weights))

    if yolo_weights:
        model.load(torch.load(yolo_weights))

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['lr'], weight_decay=1e-10)
   
    dataset = MTLDataset(train_config, mode='train')
    dataloader = DataLoader(dataset, batch_size=dataset.batch_size, shuffle=True, collate_fn=MTLDataset.collate_fn) #collate_fn=Dataset.collate_fn)

    if args.save:
        save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 train_config['save_dir'],
                                 os.path.splitext(os.path.basename(args.model_config))[0])
        os.makedirs(save_path, exist_ok=True)

    for epoch in trange(train_config['epochs']):
        for batch in dataloader:
            loss = model.loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % train_config['save_freq'] == 0 and args.save:
            model.save(os.path.join(save_path, f'{epoch+1}.pt'))

if __name__ == "__main__":
    test_pipeline()