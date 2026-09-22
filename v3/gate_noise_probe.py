"""
Is the gate measuring a neighbour, or is it measuring sampling noise?

gate_loss runs the detector in train() mode, which is required to get a loss
dict at all. But train() mode also subsamples: roi_heads draws ~512 proposals
per image at random, and (in v3, with the RPN unfrozen) the RPN draws a random
fg/bg anchor set. So two calls on the *same model and same batch* do not
return the same number.

Across v1 and v2 the gate's measured signal was tiny - excess_median 0.0058,
i.e. a neighbour scores 0.6% worse than the client itself. If repeat-call
noise on one fixed model is the same order, then `excess` is noise and no
amount of gate tuning can help.

Compares, on one fixed batch:
  * spread of repeated gate_loss calls on ONE model   (pure noise)
  * gap between two different clients' models         (the signal)

Eval only. Nothing trains, nothing is written back.
"""
import argparse

import numpy as np
import torch

import gossip
from config import GATE_BATCH_SIZE
from data import DataRegistry
from model import build_model


def load_client(path, client, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model().to(device)
    m.load_state_dict(ck["heads"][client], strict=False)
    return m


def repeat(model, batch, device, n):
    return np.array([gossip.gate_loss(model, batch, device) for _ in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="../results/runs/gp3fo2_local_only_seed42_heads.pt")
    ap.add_argument("--own", default="clear")
    ap.add_argument("--other", default="rainy")
    ap.add_argument("--repeats", type=int, default=12)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reg = DataRegistry()
    names = reg.filenames(args.own, 0, "train")[:GATE_BATCH_SIZE]
    batch = reg.batch(names)

    own = load_client(args.heads, args.own, device)
    other = load_client(args.heads, args.other, device)

    for label, keys in [("v2 keys (head only)",
                         ("loss_classifier", "loss_box_reg")),
                        ("v3 keys (all four)",
                         ("loss_classifier", "loss_box_reg",
                          "loss_objectness", "loss_rpn_box_reg"))]:
        gossip.GATE_LOSS_KEYS = keys
        a = repeat(own, batch, device, args.repeats)
        b = repeat(other, batch, device, args.repeats)

        noise = a.std() / a.mean()                      # same model, same batch
        signal = (b.mean() - a.mean()) / a.mean()       # what `excess` sees

        print(f"\n--- {label} ---")
        print(f"  own   mean {a.mean():.4f}  std {a.std():.4f}  "
              f"[{a.min():.4f}, {a.max():.4f}]")
        print(f"  other mean {b.mean():.4f}  std {b.std():.4f}")
        print(f"  repeat-call noise (cv) : {noise:.4f}")
        print(f"  neighbour signal (excess): {signal:+.4f}")
        print(f"  signal / noise         : {abs(signal) / noise:.2f}"
              if noise > 0 else "  noise is zero")

    print("\nFor reference, the gp3fo2 run measured excess_median 0.0058.")
    print("A signal/noise near or below 1 means the gate cannot see neighbours.")


if __name__ == "__main__":
    main()
