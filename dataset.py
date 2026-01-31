import torch
import os
import cv2
import json
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

class FaceDataset(Dataset):
    def __init__(self, image_root, annotation_json):
        self.image_root = image_root
        with open(annotation_json, 'r') as f:
            self.data = json.load(f)

    def __getitem__(self, idx):
        item = self.data[idx]
        path = os.path.join(self.image_root, item['filename'])

        if not os.path.exists(path):
            print(f"[WARNING] Image file not found: {path}")
            
            dummy_img = torch.zeros((3, 224, 224), dtype=torch.float32)
            return dummy_img, {"boxes": torch.zeros((0, 4), dtype=torch.float32), "labels": torch.zeros((0,), dtype=torch.int64)}

        img = cv2.imread(path)
        if img is None:
            print(f"[WARNING] Failed to load image: {path}")
            dummy_img = torch.zeros((3, 224, 224), dtype=torch.float32)
            return dummy_img, {"boxes": torch.zeros((0, 4), dtype=torch.float32), "labels": torch.zeros((0,), dtype=torch.int64)}

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = torch.tensor(item['boxes'], dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  
        img = F.to_tensor(img)

        return img, {"boxes": boxes, "labels": labels}

    def __len__(self):
        return len(self.data)
