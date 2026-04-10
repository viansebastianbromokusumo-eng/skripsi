import os
import yaml
from glob import glob
import logging

import cv2
import numpy as np
import torch
from math import ceil

from model.data.utils import pad_to, pad_xywh

from typing import Tuple

log = logging.getLogger("dataset")
logging.basicConfig(level=logging.DEBUG)


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class for loading images and annotations.

    Args:
        config (str): path to dataset config file
        batch_size (optional, int): batch size for dataloader
        mode (optional, str): dataset mode (train, val, test)
        img_size (optional, Tuple[int,int]): image size to pad images to
    """
    def __init__(self, config:str, batch_size:int=8, mode:str='train', img_size:Tuple[int,int]=(640, 640)):
        super().__init__()
        self.config = yaml.safe_load(open(config, 'r'))
        self.dataset_path = os.path.join(os.path.dirname(config), self.config['path'])
        self.batch_size = batch_size
        self.img_size = img_size

        assert mode in ('train', 'val', 'test'), f'Invalid mode: {mode}'
        self.mode = mode

        self.im_files = self.get_image_paths()
        log.debug(f'Found {len(self.im_files)} images in {os.path.join(self.dataset_path, self.config[self.mode])}')

        self.label_files = self.get_label_paths()
        if self.label_files is not None:
            log.debug(f'Found {len(self.label_files)} labels in {os.path.join(self.dataset_path, self.config[self.mode+"_labels"])}')
        else:
            log.debug(f'No labels found in {os.path.join(self.dataset_path, self.config[self.mode+"_labels"])}')

        self.labels = self.get_labels()

        self.seen_idxs = set()

    def get_image_paths(self):
        """
        Get image paths from dataset directory

        Searches recursively for .jpg, .png, and .jpeg files.
        """
        im_dir = os.path.join(self.dataset_path, self.config[self.mode])

        image_paths = glob(os.path.join(im_dir, '*.jpg')) + \
                      glob(os.path.join(im_dir, '*.png')) + \
                      glob(os.path.join(im_dir, '*.jpeg'))
 
        return image_paths
    
    def get_label_paths(self):
        """
        Get label paths from dataset directory

        Uses ids from image paths to find corresponding label files.

        If no label directory is found, returns None.
        """
        label_dir = os.path.join(self.dataset_path, self.config[self.mode+'_labels'])
        if os.path.isdir(label_dir):
            return [os.path.join(label_dir, os.path.splitext(os.path.basename(p))[0]+".txt") for p in self.im_files]
        return None
    
    def get_labels(self):
        """
        Gets labels from label files (assumes COCO formatting)

        Returns a list of dictionaries for each file
            {
                'cls': torch.Tensor of shape (num_boxes,)
                'bboxes': torch.Tensor of shape (num_boxes, 4) in (xywh) format
            }

        If no label files were found, returns a list of empty dictionaries.
        """
        if self.label_files is None:
            return [{} for _ in range(len(self.im_files))]
        labels = []
        for label_file in self.label_files:
            annotations = open(label_file, 'r').readlines()
            cls, boxes = [], []
            for ann in annotations:
                ann = ann.strip('\n').split(' ')
                cls.append(int(ann[0]))

                # box provided in xywh format
                boxes.append(torch.from_numpy(np.array(ann[1:5], dtype=float)))

            labels.append({
                'cls': torch.tensor(cls),
                'bboxes': torch.vstack(boxes)
            })
        return labels
    
    def load_image(self, idx):
        """
        Loads image at specified index and prepares for model input.

        Changes image shape to be specified img_size, but preserves aspect ratio.
        """
        im_file = self.im_files[idx]
        im_id = os.path.splitext(os.path.basename(im_file))[0]
        image = cv2.cvtColor(cv2.imread(im_file), cv2.COLOR_BGR2RGB)

        h0, w0 = image.shape[:2]

        if h0 > self.img_size[0] or w0 > self.img_size[1]:
            # Resize to have max dimension of img_size, but preserve aspect ratio
            ratio = min(self.img_size[0]/h0, self.img_size[1]/w0)
            h, w = min(ceil(h0*ratio), self.img_size[0]), min(ceil(w0*ratio), self.img_size[1])
            image = cv2.resize(image, (h, w), interpolation=cv2.INTER_LINEAR)

        image = image.transpose((2, 0, 1))  # (h, w, 3) -> (3, h, w)
        image = torch.from_numpy(image).float() / 255.0
        
        # Pad image with black bars to desired img_size
        image, pads = pad_to(image, shape=self.img_size)

        h, w = image.shape[-2:]

        return image, pads, (h0,w0), im_id

    def get_image_and_label(self, idx):
        """
        Gets image and annotations at specified index
        """
        label = self.labels[idx]
        if idx in self.seen_idxs:
            return label
        label['images'], label['padding'], label['orig_shapes'], label['ids'] = self.load_image(idx)
        label['bboxes'] = pad_xywh(label['bboxes'], label['padding'], label['orig_shapes'], return_norm=True)
        self.seen_idxs.add(idx)

        return label

    def __len__(self) -> int:
        return len(self.im_files)

    def __getitem__(self, index):
        return self.get_image_and_label(index)
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate function to specify how to combine a list of samples into a batch
        """
        collated_batch = {}
        for k in batch[0].keys():
            if k == "images":
                collated_batch[k] = torch.stack([b[k] for b in batch], dim=0)
            elif k in ('cls', 'bboxes'):
                collated_batch[k] = torch.cat([b[k] for b in batch], dim=0)
            elif k in ('padding', 'orig_shapes', 'ids'):
                collated_batch[k] = [b[k] for b in batch]
        
        collated_batch['batch_idx'] = [torch.full((batch[i]['cls'].shape[0],), i) for i in range(len(batch))]
        collated_batch['batch_idx'] = torch.cat(collated_batch['batch_idx'], dim=0)
                
        return collated_batch



import csv

import logging as log

from torch.utils.data import Dataset

# Assuming these utils exist in your repo based on the previous code
# from .utils import pad_to, pad_xywh 

class MTLDataset(Dataset):
    """
    MTL Dataset for Joint Low-Light Enhancement and Object Detection.
    Reads from metadata CSV files containing paired paths.
    """
    def __init__(self, config:str, batch_size:int=8, mode:str='train', img_size:Tuple[int,int]=(640, 640)):
        super().__init__()
        # Load config to get the dataset root path
        self.config = yaml.safe_load(open(config, 'r'))
        # Base path where the folders (test, train, val) and csvs are located
        # self.dataset_path = os.path.join(os.path.dirname(config), self.config['path'])
        self.dataset_path = self.config['path']
        
        self.batch_size = batch_size
        self.img_size = img_size
        self.mode = mode
        
        assert mode in ('train', 'val', 'test'), f'Invalid mode: {mode}'
        
        # 1. Parsing the CSV Metadata
        # CSV Format: image_low, image_high, label, split
        metadata_file = os.path.join(self.dataset_path, f"{mode}_metadata.csv")
        
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        self.data_entries = []
        with open(metadata_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data_entries.append(row)
        
        log.info(f"Loaded {len(self.data_entries)} pairs from {metadata_file}")

        # 2. Pre-load Labels (Bounding Boxes)
        self.labels = self.get_labels()

        # Cache for seen indices during training
        self.seen_idxs = set()

    def get_labels(self):
        """
        Parses label files referenced in the CSV entries.
        """
        labels = []
        for entry in self.data_entries:
            # Construct full path. Entry paths usually start with './', 
            # so we join carefully or just strip the dot.
            # adjusting path logic based on your csv example: "./test/..."
            rel_path = entry['label']
            if rel_path.startswith('./'):
                rel_path = rel_path[2:]
            
            label_file = os.path.join(self.dataset_path, rel_path)
            
            cls, boxes = [], []
            
            # Handle empty/missing label files (background images)
            if os.path.exists(label_file):
                try:
                    with open(label_file, 'r') as f:
                        annotations = f.readlines()
                        
                    for ann in annotations:
                        ann = ann.strip().split(' ')
                        if len(ann) >= 5: # Basic validation
                            cls.append(int(float(ann[0])))
                            # box provided in xywh format
                            boxes.append(torch.from_numpy(np.array(ann[1:5], dtype=float)))
                except Exception as e:
                    log.warning(f"Error reading label {label_file}: {e}")

            # If boxes exist, stack them; otherwise empty tensor
            if boxes:
                label_dict = {
                    'cls': torch.tensor(cls),
                    'bboxes': torch.vstack(boxes)
                }
            else:
                label_dict = {
                    'cls': torch.tensor([]),
                    'bboxes': torch.zeros((0, 4))
                }
            labels.append(label_dict)
            
        return labels

    def preprocess_image(self, im_path):
        """
        Helper to load and resize a single image.
        Returns: processed_image, padding_info, (h0, w0)
        """
        # Handle path joining
        if im_path.startswith('./'):
            im_path = im_path[2:]
        full_path = os.path.join(self.dataset_path, im_path)
        
        img_bgr = cv2.imread(full_path)
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {full_path}")
            
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = image.shape[:2]

        # Calculate Scale Ratio
        # We resize such that the largest side fits into img_size
        ratio = min(self.img_size[0]/h0, self.img_size[1]/w0)
        h, w = min(ceil(h0*ratio), self.img_size[0]), min(ceil(w0*ratio), self.img_size[1])
        
        # Resize
        if ratio != 1:
            image = cv2.resize(image, (h, w), interpolation=cv2.INTER_LINEAR)
            
        # HWC -> CHW
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float() / 255.0
        
        # Pad to square (or target size)
        # Assuming pad_to returns (padded_img, (pad_left, pad_top, ...))
        # You need to ensure 'pad_to' is available or implement it here
        image, pads = pad_to(image, shape=self.img_size)
        
        return image, pads, (h0, w0)

    def load_image_pair(self, idx):
        """
        Loads BOTH Low-Light and High-Light images.
        CRITICAL: They must undergo identical geometric transformations.
        """
        entry = self.data_entries[idx]
        im_id = os.path.splitext(os.path.basename(entry['image_low']))[0]

        # 1. Load and Transform Low Light Image
        img_low, pads, orig_shape = self.preprocess_image(entry['image_low'])
        
        # 2. Load and Transform High Light (GT) Image
        # We use the EXACT same logic so pixels align perfectly
        img_high, _, _ = self.preprocess_image(entry['image_high'])

        return img_low, img_high, pads, orig_shape, im_id

    def __getitem__(self, index):
        """
        Returns dictionary containing low-light input, high-light target, and labels.
        """
        label = self.labels[index]
        
        # If caching strategy is desired (optional)
        # if index in self.seen_idxs: return label
        
        # Load the pair
        img_low, img_high, padding, orig_shapes, im_id = self.load_image_pair(index)
        
        # Update the dictionary
        label['img'] = img_low          # Input for Backbone
        label['gt_img'] = img_high      # Target for LLIE Loss
        label['padding'] = padding
        label['orig_shapes'] = orig_shapes
        label['ids'] = im_id
        
        # Adjust Bounding Boxes to match the resized/padded image
        # pad_xywh should convert normalized xywh or absolute xywh to the new coordinate system
        # Assuming your existing utils.pad_xywh handles this:
        if 'bboxes' in label and len(label['bboxes']) > 0:
            # Create a copy to avoid modifying the cached label in-place repeatedly if logic changes
            bboxes = label['bboxes'].clone() 
            label['bboxes'] = pad_xywh(bboxes, padding, orig_shapes, return_norm=True)
        
        # self.seen_idxs.add(index) # Uncomment if you want to implement caching logic
        
        return label

    def __len__(self) -> int:
        return len(self.data_entries)

    @staticmethod
    def collate_fn(batch):
        """
        Stacks images and targets into batch tensors.
        """
        collated_batch = {}
        
        # Keys that need simple stacking (Images)
        # We need to stack 'img' (low) AND 'gt_img' (high)
        for key in ['img', 'gt_img']:
            if key in batch[0]:
                 collated_batch[key] = torch.stack([b[key] for b in batch], dim=0)

        # Keys that need concatenation (Labels)
        for key in ['cls', 'bboxes']:
            if key in batch[0]:
                collated_batch[key] = torch.cat([b[key] for b in batch], dim=0)
        
        # Keys that remain as lists (Meta info)
        for key in ['padding', 'orig_shapes', 'ids']:
            if key in batch[0]:
                collated_batch[key] = [b[key] for b in batch]

        # Create batch index for detection (which image does this box belong to?)
        if 'cls' in batch[0]:
            batch_idx = []
            for i, b in enumerate(batch):
                n_boxes = b['cls'].shape[0]
                batch_idx.append(torch.full((n_boxes,), i))
            collated_batch['batch_idx'] = torch.cat(batch_idx, dim=0)
        
        return collated_batch
    
    