"""Does the warm-up actually ramp both groups, and land on target?"""
import config
from model import build_model, build_optimizer, build_warmup

m = build_model()
o = build_optimizer(m)
w = build_warmup(o)
print("LR_WARMUP", config.LR_WARMUP, "factor", config.LR_WARMUP_FACTOR)
print("targets  ", [g["lr"] for g in o.param_groups], "(rpn, head)")
seen = []
for step in range(config.LR_WARMUP + 5):
    lrs = [round(g["lr"], 8) for g in o.param_groups]
    if step in (0, 1, 25, 50, 99, 100, 101, 104):
        seen.append((step, lrs))
    o.step(); w.step()
for s, l in seen:
    print(f"  step {s:>4d}  rpn {l[0]:.6f}  head {l[1]:.6f}  ratio {l[1]/l[0]:.1f}")
final = [g["lr"] for g in o.param_groups]
assert abs(final[0] - config.LR_PRETRAINED) < 1e-9, final
assert abs(final[1] - config.LR_HEAD) < 1e-9, final
print("OK: both groups reach target, 10x ratio held throughout")
