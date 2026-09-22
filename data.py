"""
Data loading for BDD100K object detection.

Reads the already-built partition (bdd100k_manifest.csv) and box lists
(boxes_index.json) - nothing here re-derives the split.

What each item looks like, matching what torchvision detection models expect:

    image   FloatTensor[3, 720, 1280]  RGB in [0, 1], NOT resized
    target  {"boxes":  FloatTensor[N, 4]  x1,y1,x2,y2 absolute pixels
             "labels": Int64Tensor[N]    OBJECT_CLASSES.index(cat) + 1}

No normalisation and no resize on purpose: Faster R-CNN's internal
GeneralizedRCNNTransform does both, and maps predicted boxes back to these
original coordinates afterwards. Doing it here as well would apply it twice
and silently break every IoU against the ground truth.
"""

import json
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from config import (
    BOXES_INDEX_PATH, CATEGORY_TO_LABEL, CLIENTS, IMAGES_ROOT,
    IMAGE_INDEX_PATH, INDEX_SPLIT_DIRS, MANIFEST_PATH, TASKS,
)
from replay import rarity_weights


# --- image index -----------------------------------------------------------

def build_image_index(images_root=IMAGES_ROOT, out_path=IMAGE_INDEX_PATH):
    """
    filename -> path *relative to images_root*.

    Relative, not absolute: the same index file then works on this machine and
    on the cluster, where only the root differs. The Kaggle extraction nests
    some images one level deeper (100k/train/{trainA,trainB,testA,testB}/),
    so this walks rather than assuming a flat directory.
    """
    index = {}
    for split_dir in INDEX_SPLIT_DIRS:
        root = os.path.join(images_root, split_dir)
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.lower().endswith(".jpg"):
                    full = os.path.join(dirpath, name)
                    index[name] = os.path.relpath(full, images_root)

    with open(out_path, "w") as f:
        json.dump(index, f)
    return index


def load_image_index(images_root=IMAGES_ROOT, path=IMAGE_INDEX_PATH):
    """Loads the index, building it on first use."""
    if not os.path.exists(path):
        return build_image_index(images_root, path)
    with open(path) as f:
        return json.load(f)


# --- manifest / boxes ------------------------------------------------------

def load_manifest(path=MANIFEST_PATH):
    return pd.read_csv(path)


def load_boxes_index(path=BOXES_INDEX_PATH):
    with open(path) as f:
        return json.load(f)


def cell_filenames(manifest, client, task_id, split):
    """The filenames in one (client, task, split) cell."""
    rows = manifest[
        (manifest["weather"] == client)
        & (manifest["timeofday"] == TASKS[task_id])
        & (manifest["split"] == split)
    ]
    return rows["filename"].tolist()


# --- dataset ---------------------------------------------------------------

class BDD100KDetection(Dataset):
    """
    Built from an explicit list of filenames rather than a DataFrame slice, so
    the replay buffer - which is just a list of filenames - can reuse it
    unchanged.
    """

    def __init__(self, filenames, boxes_index, image_index, images_root=IMAGES_ROOT):
        self.filenames = list(filenames)
        # Slice both indexes down to just this cell's files. The full ones cover
        # all 70k images (boxes_index alone is ~120 MB), and every DataLoader
        # worker forks a copy of whatever the dataset holds - with five clients'
        # loaders alive at once and persistent_workers on, handing each worker
        # the whole index costs GBs of RAM for data it never reads. The values
        # are shared references, so slicing itself is cheap.
        self.boxes_index = {f: boxes_index[f] for f in self.filenames
                            if f in boxes_index}
        self.image_index = {f: image_index[f] for f in self.filenames
                            if f in image_index}
        self.images_root = images_root

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        filename = self.filenames[i]

        path = os.path.join(self.images_root, self.image_index[filename])
        with Image.open(path) as img:
            image = TF.to_tensor(img.convert("RGB"))   # [0,1], no normalisation

        boxes, labels = [], []
        for box in self.boxes_index.get(filename, []):
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            # Faster R-CNN's loss produces NaN on a degenerate box, and a few
            # BDD100K annotations are zero-width/height once cast to float32.
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(CATEGORY_TO_LABEL[box["category"]])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            # carried through so the replay buffer can score/evict by filename
            "filename": filename,
        }
        return image, target


def detection_collate(batch):
    """
    Detection batches stay as lists: images differ in size after the model's
    internal resize, and targets are variable-length, so neither stacks.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


def make_loader(filenames, boxes_index, image_index, batch_size, shuffle,
                num_workers, images_root=IMAGES_ROOT):
    ds = BDD100KDetection(filenames, boxes_index, image_index, images_root)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=detection_collate,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def load_batch(filenames, boxes_index, image_index, images_root=IMAGES_ROOT):
    """
    One collated batch straight from filenames - no DataLoader, no sampler, no
    pin-memory thread. For the small one-off batches (replay, gossip gate)
    that get built and immediately consumed on every single step: going
    through the full loader machinery for those paid DataLoader/thread setup
    cost every step with nothing to amortize it against.
    """
    if not filenames:
        return None
    ds = BDD100KDetection(filenames, boxes_index, image_index, images_root)
    return detection_collate([ds[i] for i in range(len(ds))])


# --- the loaders the training loop actually asks for -----------------------

class DataRegistry:
    """
    Holds the three big shared objects (manifest, boxes, image index) so they
    are parsed once instead of once per cell - boxes_index.json alone is 125 MB.
    """

    def __init__(self, images_root=IMAGES_ROOT):
        self.manifest = load_manifest()
        self.boxes_index = load_boxes_index()
        self.image_index = load_image_index(images_root)
        self.images_root = images_root
        self._cells = {}        # (client, task, split) -> filenames
        self._rarity = None     # global class rarity, identical for every client

    def filenames(self, client, task_id, split):
        """Memoised: the manifest never changes, but the training loop asks for
        the same cells repeatedly (every task boundary re-reads every test cell
        seen so far), and each miss is a full pandas scan of all 70k rows."""
        key = (client, task_id, split)
        if key not in self._cells:
            self._cells[key] = cell_filenames(self.manifest, client, task_id, split)
        return self._cells[key]

    def rarity_weights(self):
        """
        Global class rarity, computed once.

        It is a full pass over all 1.29M boxes and the answer does not depend on
        the client, but every ReplayBuffer used to recompute it in its
        constructor - five identical scans of the whole index at startup.
        """
        if self._rarity is None:
            self._rarity = rarity_weights(self.boxes_index)
        return self._rarity

    def loader(self, filenames, batch_size, shuffle, num_workers):
        return make_loader(filenames, self.boxes_index, self.image_index,
                           batch_size, shuffle, num_workers, self.images_root)

    def batch(self, filenames):
        """One collated batch, no DataLoader - see load_batch()."""
        return load_batch(filenames, self.boxes_index, self.image_index,
                          self.images_root)

    def build_dataloaders(self, batch_size, num_workers):
        """loaders[client][task_id] = {"train": DataLoader, "test": DataLoader}"""
        loaders = {}
        for client in CLIENTS:
            loaders[client] = {}
            for task_id in range(len(TASKS)):
                loaders[client][task_id] = {
                    "train": self.loader(self.filenames(client, task_id, "train"),
                                         batch_size, True, num_workers),
                    "test": self.loader(self.filenames(client, task_id, "test"),
                                        batch_size, False, num_workers),
                }
        return loaders

    def pooled_test_loader(self, client, batch_size, num_workers):
        """
        One client's 3 task test splits as a single loader - what Layer E's
        cross-client matrix evaluates against.
        """
        names = []
        for task_id in range(len(TASKS)):
            names.extend(self.filenames(client, task_id, "test"))
        return self.loader(names, batch_size, False, num_workers)


if __name__ == "__main__":
    reg = DataRegistry()
    print(f"manifest rows      : {len(reg.manifest)}")
    print(f"images indexed     : {len(reg.image_index)}")
    print(f"images with boxes  : {len(reg.boxes_index)}")

    names = reg.filenames("rainy", 1, "train")
    print(f"\nrainy / night / train: {len(names)} images")

    ds = BDD100KDetection(names[:4], reg.boxes_index, reg.image_index)
    image, target = ds[0]
    print(f"image  : {tuple(image.shape)}  dtype={image.dtype} "
          f"range=[{image.min():.2f}, {image.max():.2f}]")
    print(f"boxes  : {tuple(target['boxes'].shape)}")
    print(f"labels : {target['labels'].tolist()}")
