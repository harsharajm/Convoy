"""
Run logger: one timestamped file per run, plus console output.

Only progress milestones go here (already-throttled prints in train.py:
every 10 scheduler slices, final per-client/system/cross-client summaries).
Nothing per-batch - that would make the file grow unbounded over a run of
thousands of steps.
"""

import logging
import os
from datetime import datetime


def get_logger(mode, seed, log_dir="results/logs", tag=""):
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{tag}_{mode}_seed{seed}" if tag else f"{mode}_seed{seed}"
    path = os.path.join(log_dir, f"{stem}_{stamp}.log")

    logger = logging.getLogger(f"{stem}_{stamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"log file: {path}")
    return logger
