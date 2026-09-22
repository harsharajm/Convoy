"""
Replay buffer for continual learning.

Stores image *references* (filenames), never decoded tensors. 2000 full-res
BDD100K frames as float32 tensors would be ~22 GB per client and ~110 GB
across five; as filenames it is a few hundred kilobytes, and the images are
re-read through the normal dataset path when they are replayed.

Eviction follows design_choices.pdf 2.1.1: score each image by the rarest
class it contains, keep the rare ones, drop the common ones. Without this the
buffer fills with `car` frames - car is 56.22% of all instances, so uniform
eviction would leave the tail classes with almost no replay coverage.
"""

import math
import random
from collections import Counter

from config import BUFFER_CAPACITY, OBJECT_CLASSES


def class_instance_counts(boxes_index):
    """Global instances per class - the rarity scale everything is scored on."""
    counts = Counter()
    for boxes in boxes_index.values():
        for box in boxes:
            counts[box["category"]] += 1
    return counts


def rarity_weights(boxes_index):
    """
    class -> rarity weight, higher = rarer.

    -log(frequency) rather than a raw rank, so the gap between `car` (56%) and
    `train` (0.01%) is expressed as the several orders of magnitude it
    actually is, instead of being flattened to "9 places apart".
    """
    counts = class_instance_counts(boxes_index)
    total = sum(counts.values())
    weights = {}
    for cat in OBJECT_CLASSES:
        n = counts.get(cat, 0)
        # unseen class: treat as rarer than anything observed
        weights[cat] = math.log(total / n) if n else math.log(total)
    return weights


class ReplayBuffer:
    """
    Fixed-capacity set of filenames, each scored by the rarest class in it.

    A newly offered image only displaces a stored one if it is rarer, so the
    buffer drifts toward the tail classes as training proceeds and never
    silently loses them to a flood of common frames.
    """

    def __init__(self, boxes_index, capacity=BUFFER_CAPACITY, seed=0, weights=None):
        self.capacity = capacity
        self.boxes_index = boxes_index
        # `weights` is global and client-independent, so the caller can compute
        # it once and share it across clients instead of paying a full pass over
        # the box index per buffer.
        self.weights = rarity_weights(boxes_index) if weights is None else weights
        self.rng = random.Random(seed)
        self.entries = {}      # filename -> priority

    # --- scoring ---

    def priority(self, filename):
        """An image is as rare as the rarest class in it."""
        boxes = self.boxes_index.get(filename, [])
        if not boxes:
            return 0.0
        return max(self.weights.get(b["category"], 0.0) for b in boxes)

    # --- filling ---

    def add(self, filename):
        if filename in self.entries:
            return False
        p = self.priority(filename)

        if len(self.entries) < self.capacity:
            self.entries[filename] = p
            return True

        # Full: displace the most common stored image, but only if this one is
        # genuinely rarer - otherwise the buffer would churn on ties.
        victim = min(self.entries, key=self.entries.get)
        if p > self.entries[victim]:
            del self.entries[victim]
            self.entries[filename] = p
            return True
        return False

    def add_many(self, filenames):
        return sum(self.add(f) for f in filenames)

    # --- reading ---

    def sample(self, n):
        """n filenames, without replacement (or all of them if fewer)."""
        names = list(self.entries)
        if not names:
            return []
        if n >= len(names):
            return names
        return self.rng.sample(names, n)

    def filenames(self):
        return list(self.entries)

    def __len__(self):
        return len(self.entries)

    # --- reporting ---

    def class_coverage(self):
        """How many buffered images contain each class - the check that the
        eviction policy is doing what it claims."""
        cov = Counter()
        for filename in self.entries:
            present = {b["category"] for b in self.boxes_index.get(filename, [])}
            for cat in present:
                cov[cat] += 1
        return cov

    def state_dict(self):
        return {"capacity": self.capacity, "entries": dict(self.entries)}

    def load_state_dict(self, state):
        self.capacity = state["capacity"]
        self.entries = dict(state["entries"])
