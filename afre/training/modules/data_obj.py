import os 
import torch
import math
import cv2
import json 
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from utils.yolo_helpers import pad_to, pad_xywh


class Nikon750Dataset_Direct(Dataset):
    """
    Nikon750 dataset loader (COCO format)
    Compatible with YOLO-style detection heads
    """

    def __init__(
        self,
        img_root: str,
        ann_path: str,
        img_size: tuple[int, int] = (640, 640)
    ):
        super().__init__()
        self.img_root = img_root
        self.img_size = img_size

        # ---- Load COCO JSON ----
        with open(ann_path, "r") as f:
            coco = json.load(f)

        self.images = coco["images"]

        # category mapping: coco_id → continuous id
        self.catid_to_idx = {c["id"]: i for i, c in enumerate(coco["categories"])}

        # image_id → annotations
        self.imgid_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            if ann["iscrowd"] == 0:
                self.imgid_to_anns[ann["image_id"]].append(ann)

        print(f"[Nikon750] Loaded {len(self.images)} images from {ann_path}")

    def __len__(self):
        return len(self.images)

    # -------------------------
    # Image loader
    # -------------------------
    def _load_image(self, img_path: str):
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h0, w0 = image.shape[:2]

        # resize (same logic as your LoLiStreet dataset)
        if h0 > self.img_size[0] or w0 > self.img_size[1]:
            ratio = min(self.img_size[0] / h0, self.img_size[1] / w0)
            h, w = math.ceil(h0 * ratio), math.ceil(w0 * ratio)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            h, w = h0, w0

        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float() / 255.0

        padded, pads = pad_to(image, shape=self.img_size)

        return padded, pads, (h0, w0)
    
    @staticmethod
    def collate_fn(batch):
        out = {}
        out["images"] = torch.stack([b["images"] for b in batch], 0)
        out["cls"] = torch.cat([b["cls"] for b in batch], 0)
        out["bboxes"] = torch.cat([b["bboxes"] for b in batch], 0)

        out["padding"] = [b["padding"] for b in batch]
        out["orig_shapes"] = [b["orig_shapes"] for b in batch]
        out["im_id"] = [b["im_id"] for b in batch]

        batch_idx = []
        for i, b in enumerate(batch):
            batch_idx.append(torch.full((b["cls"].shape[0],), i, dtype=torch.long))
        out["batch_idx"] = torch.cat(batch_idx, 0)

        return out

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_root, img_info["file_name"])

        image, pads, orig_shape = self._load_image(img_path)

        # ----- Load Annotations -----
        anns = self.imgid_to_anns.get(img_info["id"], [])

        if len(anns) == 0:
            cls = torch.empty((0,), dtype=torch.long)
            bboxes = torch.empty((0, 4), dtype=torch.float32)
        else:
            boxes = []
            labels = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                boxes.append([x + w / 2, y + h / 2, w, h])  # xywh (pixel)
                labels.append(self.catid_to_idx[ann["category_id"]])

            bboxes = torch.tensor(boxes, dtype=torch.float32)
            cls = torch.tensor(labels, dtype=torch.long)

            # normalize
            h0, w0 = orig_shape
            bboxes[:, [0, 2]] /= w0
            bboxes[:, [1, 3]] /= h0

            # apply padding (your util)
            bboxes = pad_xywh(
                bboxes,
                pads,
                orig_shape,
                return_norm=True
            )

        return {
            "images": image,
            "cls": cls,
            "bboxes": bboxes,
            "padding": pads,
            "orig_shapes": orig_shape,
            "im_id": img_info["file_name"]
        }


class LoLiStreetDataset_Direct(Dataset):
    """
    Dataset that directly scans directories for Low, High, and Label files.
    It matches files based on the filename found in the 'high' directory.
    """
    def __init__(self, high_dir: str, low_dir: str, labels_dir: str, img_size: tuple[int, int] = (640, 640), transform=None):
        super().__init__()
        self.img_size = img_size
        self.transform = transform if transform else T.ToTensor()
        
        self.high_images = []
        self.low_images = []
        self.labels_files = []

        # 1. Scan the Ground Truth (High) directory
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        try:
            # Get all image files in high dir
            files = sorted([f for f in os.listdir(high_dir) if f.lower().endswith(valid_extensions)])
        except FileNotFoundError:
            print(f"[Error] Directory not found: {high_dir}")
            files = []

        # 2. Match with Low and Labels
        for f_name in files:
            high_path = os.path.join(high_dir, f_name)
            low_path = os.path.join(low_dir, f_name) # Assumes identical filename
            
            # Construct label path: replace image extension with .txt
            file_stem = os.path.splitext(f_name)[0]
            label_path = os.path.join(labels_dir, file_stem + '.txt')

            # 3. Verify existence of all three components
            if os.path.exists(low_path) and os.path.exists(label_path):
                self.high_images.append(high_path)
                self.low_images.append(low_path)
                self.labels_files.append(label_path)
            else:
                # Optional: Print warning if a pair is missing
                # print(f"Skipping {f_name}: Missing low image or label file.")
                pass

        print(f"Found {len(self.high_images)} valid triplets in {high_dir}")

        # Load all labels initially (Cache)
        self.labels_cache = self._load_all_labels()
        
    def __len__(self):
        return len(self.high_images)

    # --- Unchanged Helper Methods ---

    def _load_labels_for_file(self, label_filepath: str) -> dict:
        labels_list = self._read_yolo_labels(label_filepath) 
        if not labels_list:
            return {'cls': torch.empty(0, dtype=torch.long), 'bboxes': torch.empty(0, 4, dtype=torch.float32)}
        label_tensor = torch.tensor(labels_list, dtype=torch.float32)
        return {'cls': label_tensor[:, 0].long(), 'bboxes': label_tensor[:, 1:5]}

    def _load_all_labels(self):
        return [self._load_labels_for_file(f) for f in self.labels_files]

    def _read_yolo_labels(self, label_filepath: str) -> list[list[float]]:
        labels = []
        if not os.path.exists(label_filepath): return labels
        try:
            with open(label_filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        labels.append([float(p) for p in parts])
        except Exception as e:
            print(f"Warning: Error reading labels from {label_filepath}: {e}") 
        return labels

    def _load_single_image(self, img_path: str):
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h0, w0 = image.shape[:2]
        
        # Resize
        if h0 > self.img_size[0] or w0 > self.img_size[1]:
            ratio = min(self.img_size[0] / h0, self.img_size[1] / w0)
            h, w = math.ceil(h0 * ratio), math.ceil(w0 * ratio)
            h, w = min(h, self.img_size[0]), min(w, self.img_size[1]) 
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            h, w = h0, w0
            
        # To Tensor
        image = image.transpose((2, 0, 1)) 
        image_tensor = torch.from_numpy(image).float() / 255.0
        
        padded_image, pads = pad_to(image_tensor, shape=self.img_size)
        
        return padded_image, pads, (h0, w0)
    
    # --- Collate Function (Same as before) ---
    @staticmethod
    def collate_fn(batch: list[dict]):
        collated_batch = {}
        collated_batch['images_high'] = torch.stack([b['images_high'] for b in batch], dim=0)
        collated_batch['images_low'] = torch.stack([b['images_low'] for b in batch], dim=0)
        collated_batch['cls'] = torch.cat([b['cls'] for b in batch], dim=0)
        collated_batch['bboxes'] = torch.cat([b['bboxes'] for b in batch], dim=0)
        for k in ('padding_high', 'orig_shapes_high', 'im_id'):
            collated_batch[k] = [b[k] for b in batch]
        batch_idx_list = [torch.full((batch[i]['cls'].shape[0],), i, dtype=torch.float32) for i in range(len(batch))]
        collated_batch['batch_idx'] = torch.cat(batch_idx_list, dim=0)
        return collated_batch

    def __getitem__(self, idx: int):
        low_img_tensor, low_pads, low_orig_shapes = self._load_single_image(self.low_images[idx])
        high_img_tensor, high_pads, high_orig_shapes = self._load_single_image(self.high_images[idx])
        label = self.labels_cache[idx]
        
        # Adjust Bboxes
        # WARNING: Ensure 'pad_xywh' is available in your scope.
        adjusted_bboxes = pad_xywh(
            label['bboxes'], 
            high_pads, 
            high_orig_shapes, 
            return_norm=True 
        )

        return {
            'images_low': low_img_tensor,
            'images_high': high_img_tensor,
            'cls': label['cls'],
            'bboxes': adjusted_bboxes,
            'padding_high': high_pads,
            'orig_shapes_high': high_orig_shapes,
            'im_id': os.path.splitext(os.path.basename(self.high_images[idx]))[0]
        }

