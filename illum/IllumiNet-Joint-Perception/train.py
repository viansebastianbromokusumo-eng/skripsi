import os
import argparse
import yaml
import torch
from tqdm import trange

from model.models.detection_model import DetectionModel_MTL
from model.data.dataset import MTLDataset #Dataset
from model.modules.llie_dec import LowLightEnhancementDecoder_Yol

from torch.utils.data import DataLoader


# TODO:
# Dont forget to log PSNR and SSIM ,
# also need to ensure that we can see the output enhanced image

def get_args():
    parser = argparse.ArgumentParser(description='YOLOv8 model training')
    parser.add_argument(
        '--model-config',
        type=str,
        default='model/config/models/yolov8n.yaml',
        help='path to model config file'
    )
    parser.add_argument(
        '--weights',
        type=str,
        help='path to weights file'
    )

    parser.add_argument(
        '--train-config',
        type=str,
        default='model/config/training/fine_tune.yaml',
        help='path to training config file'
    )

    dataset_args = parser.add_argument_group('Dataset')
    dataset_args.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='path to dataset config file'
    )
    dataset_args.add_argument(
        '--dataset-mode',
        type=str,
        default='train',
        help='dataset mode'
    )

    parser.add_argument(
        '--device',
        '-d',
        type=str,
        default='cuda',
        help='device to model on'
    )

    parser.add_argument(
        '--save',
        '-s',
        action='store_true',
        help='save trained model weights'
    )

    return parser.parse_args()


def main(args):
    train_config = yaml.safe_load(open(args.train_config, 'r'))

    device = torch.device(args.device)
    model = DetectionModel_MTL(args.model_config, device=device)
    if args.weights is not None:
        model.load(torch.load(args.weights))

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['lr'])
    
    # backbone_params = []
    # neck_head_params = []
    # llie_params = []

    # # Assuming model.model is the nn.ModuleList from parsed YAML
    # for i, m in enumerate(model.model):
    #     # Layers 0-9 are Backbone
    #     if i < 10: 
    #         backbone_params.extend([p for p in m.parameters() if p.requires_grad])
    #     # The last layer (e.g. 19) is likely your LLIE Decoder
    #     elif isinstance(m, LowLightEnhancementDecoder_Yol): 
    #         llie_params.extend([p for p in m.parameters() if p.requires_grad])
    #     # Everything else (Neck, Detection Head)
    #     else: 
    #         neck_head_params.extend([p for p in m.parameters() if p.requires_grad])

    # # Optimizer with groups
    # optimizer = torch.optim.AdamW([
    #     {'params': backbone_params, 'lr': 1e-4}, # 0.1x LR for backbone
    #     {'params': neck_head_params, 'lr': 1e-4},      # 1x LR for detection
    #     {'params': llie_params, 'lr': 1e-3}      # 1.5x LR for LLIE (learn faster)
    # ], lr=1e-5, weight_decay=5e-4)
    
    # Include ALL parameters, regardless of current requires_grad status
    # for i, m in enumerate(model.model):
    #     if i < 10: 
    #         backbone_params.extend(m.parameters()) # Add all
    #     elif isinstance(m, LowLightEnhancementDecoder): 
    #         llie_params.extend(m.parameters())
    #     else: 
    #         neck_head_params.extend(m.parameters())

    # # Initialize Optimizer
    # optimizer = torch.optim.AdamW(...) 

    # # NOW freeze for epoch 0
    # set_freeze_status(model, freeze_backbone=True)

    # print(f"Optimizer: AdamW initialized with 3 parameter groups.")

    # dataset = Dataset(args.dataset, mode=args.dataset_mode, batch_size=train_config['batch_size'])
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


if __name__ == '__main__':
    args = get_args()
    main(args)
    