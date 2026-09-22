"""
BDD100K partition pipeline - object detection version.

Same client/task split as before (weather = client, timeofday = task),
but now keeps the real bounding boxes instead of just a yes/no per class.
"""

import json
import os
import numpy as np
import pandas as pd

BDD100K_ROOT = os.environ.get(
    "BDD100K_ROOT",
    "/home/breach/Desktop/DecentralAutonomousDriving/BDDTA/bdd100k_data",
)

_LABELS_DIR = os.path.join(BDD100K_ROOT, "labels")
TRAIN_JSON = os.path.join(_LABELS_DIR, "bdd100k_labels_images_train.json")
VAL_JSON = os.path.join(_LABELS_DIR, "bdd100k_labels_images_val.json")

OBJECT_CLASSES = [
    "car", "traffic sign", "traffic light", "person", "truck",
    "bus", "bike", "rider", "motor", "train",
]

USABLE_WEATHER = ["clear", "overcast", "rainy", "snowy", "partly cloudy"]
USABLE_TIMEOFDAY = ["daytime", "night", "dawn/dusk"]

MIN_CELL_SIZE = 500
TEST_FRACTION = 0.2
SEED = 42


def build_manifest_and_boxes():
    """
    Returns:
      manifest: one row per usable image (filename, weather, timeofday)
      boxes: {filename: [{"category": str, "x1","y1","x2","y2": float}, ...]}
    """
    rows = []
    boxes = {}
    for path in (TRAIN_JSON, VAL_JSON):
        with open(path) as f:
            data = json.load(f)
        for img in data:
            weather = img["attributes"]["weather"]
            timeofday = img["attributes"]["timeofday"]
            if weather not in USABLE_WEATHER or timeofday not in USABLE_TIMEOFDAY:
                continue

            img_boxes = []
            for label in img.get("labels", []):
                if label["category"] not in OBJECT_CLASSES:
                    continue
                box = label.get("box2d")
                if box is None:
                    continue
                img_boxes.append({
                    "category": label["category"],
                    "x1": box["x1"], "y1": box["y1"],
                    "x2": box["x2"], "y2": box["y2"],
                })

            rows.append({"filename": img["name"], "weather": weather, "timeofday": timeofday})
            boxes[img["name"]] = img_boxes

    return pd.DataFrame(rows), boxes


def assign_split(manifest: pd.DataFrame) -> pd.DataFrame:
    """80/20 train/test split, done independently inside each (weather, timeofday) cell."""
    rng = np.random.default_rng(SEED)
    manifest = manifest.copy()
    manifest["split"] = "train"
    for (_, _), cell in manifest.groupby(["weather", "timeofday"]):
        idx = cell.index.to_numpy()
        rng.shuffle(idx)
        n_test = int(len(idx) * TEST_FRACTION)
        manifest.loc[idx[:n_test], "split"] = "test"
    return manifest


def bdd_partition(manifest: pd.DataFrame):
    """Returns partition[client][task_id] = {"train": [filenames], "test": [filenames]}"""
    partition = {}
    for client in USABLE_WEATHER:
        partition[client] = {}
        client_rows = manifest[manifest["weather"] == client]
        for task_id, timeofday in enumerate(USABLE_TIMEOFDAY):
            cell = client_rows[client_rows["timeofday"] == timeofday]
            partition[client][task_id] = {
                "train": cell.loc[cell["split"] == "train", "filename"].tolist(),
                "test": cell.loc[cell["split"] == "test", "filename"].tolist(),
            }
    return partition


def cell_count_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """weather x timeofday cross-tab, train/test/total counts, flags cells under MIN_CELL_SIZE."""
    records = []
    for client in USABLE_WEATHER:
        for task_id, timeofday in enumerate(USABLE_TIMEOFDAY):
            cell = manifest[(manifest["weather"] == client) & (manifest["timeofday"] == timeofday)]
            n_train = (cell["split"] == "train").sum()
            n_test = (cell["split"] == "test").sum()
            total = n_train + n_test
            records.append({
                "client (weather)": client,
                "task_id": task_id,
                "task (timeofday)": timeofday,
                "train": n_train,
                "test": n_test,
                "total": total,
                "below_floor": total < MIN_CELL_SIZE,
            })
    return pd.DataFrame(records)


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))

    manifest, boxes = build_manifest_and_boxes()
    manifest = assign_split(manifest)

    manifest.to_csv(os.path.join(out_dir, "bdd100k_manifest.csv"), index=False)
    with open(os.path.join(out_dir, "boxes_index.json"), "w") as f:
        json.dump(boxes, f)

    table = cell_count_table(manifest)
    pd.set_option("display.width", 120)
    print(table.to_string(index=False))
    print(f"\ncells below {MIN_CELL_SIZE}-image floor: {table['below_floor'].sum()} / {len(table)}")

    n_with_boxes = sum(1 for v in boxes.values() if v)
    print(f"images with at least one box: {n_with_boxes} / {len(boxes)}")
