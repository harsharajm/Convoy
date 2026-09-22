"""
BDD100K dataset for object detection.

Each item: (image_tensor, target)
  target = {"boxes": FloatTensor[N,4] (x1,y1,x2,y2), "labels": LongTensor[N]}
  label 0 is reserved for background (torchvision detection convention),
  so a category's label = OBJECT_CLASSES.index(category) + 1.

No resize here - original image size, original box coordinates.
Model/training pipeline isn't picked yet, so resizing is left to whatever
that pipeline needs later.
"""

import json
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from partition import OBJECT_CLASSES, USABLE_WEATHER, USABLE_TIMEOFDAY

_HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.environ.get("BDD100K_MANIFEST", os.path.join(_HERE, "bdd100k_manifest.csv"))
BOXES_INDEX_PATH = os.environ.get("BDD100K_BOXES_INDEX", os.path.join(_HERE, "boxes_index.json"))
IMAGE_INDEX_PATH = os.environ.get(
    "BDD100K_IMAGE_INDEX",
    "/home/breach/Desktop/DecentralAutonomousDriving/BDDTA/code_bdd100k/data/image_index.json",
)

CATEGORY_TO_LABEL = {cat: i + 1 for i, cat in enumerate(OBJECT_CLASSES)}  # 0 = background

# Faster R-CNN applies its pretrained ImageNet normalization internally.
# Keep dataset outputs as float RGB tensors in [0, 1] so it happens exactly once.
to_tensor = transforms.ToTensor()


class BDD100KDetectionDataset(Dataset):
    def __init__(self, manifest_rows, boxes_index, image_index):
        """manifest_rows: DataFrame slice - rows for one (client, task, split) cell."""
        self.rows = manifest_rows.reset_index(drop=True)
        self.boxes_index = boxes_index
        self.image_index = image_index

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        filename = row["filename"]

        path = self.image_index[filename]
        image = Image.open(path).convert("RGB")
        image = to_tensor(image)

        boxes, labels = [], []
        for box in self.boxes_index.get(filename, []):
            boxes.append([box["x1"], box["y1"], box["x2"], box["y2"]])
            labels.append(CATEGORY_TO_LABEL[box["category"]])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return image, target


def detection_collate_fn(batch):
    """Targets are variable-length, so we can't stack them like the default collate."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_dataloaders(batch_size=8, num_workers=4):
    """Returns loaders[client][task_id] = {"train": DataLoader, "test": DataLoader}"""
    manifest = pd.read_csv(MANIFEST_PATH)
    with open(BOXES_INDEX_PATH) as f:
        boxes_index = json.load(f)
    with open(IMAGE_INDEX_PATH) as f:
        image_index = json.load(f)

    loaders = {}
    for client in USABLE_WEATHER:
        loaders[client] = {}
        client_rows = manifest[manifest["weather"] == client]
        for task_id, timeofday in enumerate(USABLE_TIMEOFDAY):
            cell = client_rows[client_rows["timeofday"] == timeofday]
            train_rows = cell[cell["split"] == "train"]
            test_rows = cell[cell["split"] == "test"]

            train_ds = BDD100KDetectionDataset(train_rows, boxes_index, image_index)
            test_ds = BDD100KDetectionDataset(test_rows, boxes_index, image_index)

            loaders[client][task_id] = {
                "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers, collate_fn=detection_collate_fn),
                "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, collate_fn=detection_collate_fn),
            }
    return loaders


if __name__ == "__main__":
    loaders = build_dataloaders(batch_size=4, num_workers=0)
    images, targets = next(iter(loaders["rainy"][1]["train"]))
    print("batch size:", len(images))
    print("one image shape:", images[0].shape)
    print("one target:", targets[0])
