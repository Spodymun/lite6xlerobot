#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def add_batch(x):
    if x.dim()==1: return x.unsqueeze(0)
    if x.dim()==2: return x.unsqueeze(0)
    return x

def load_policy(ckpt_dir: Path):
    pm = ckpt_dir/"pretrained_model"
    pm = pm if pm.is_dir() else ckpt_dir
    cfg = PreTrainedConfig.from_pretrained(str(pm))
    policy = ACTPolicy.from_pretrained(str(pm), config=cfg)
    policy.eval()
    return policy

def infer_a0(policy, state, env_vec):
    batch = {
        "observation.state": add_batch(state),
        "observation.environment_state": add_batch(env_vec),
    }
    with torch.no_grad():
        out = policy.model(batch)
        act = out[0] if isinstance(out, tuple) else out
    a0 = act[0,0].detach().cpu().numpy() if act.dim()==3 else act[0].detach().cpu().numpy()
    return a0

def mad(a,b):  # mean abs diff
    return float(np.mean(np.abs(a-b)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (..../checkpoints/last or ..../005000)")
    ap.add_argument("--ds", default="records/throws/fs_TRAIN_READY", help="dataset root")
    ap.add_argument("--slot_x", type=int, default=4)
    ap.add_argument("--slot_y", type=int, default=5)
    ap.add_argument("--y", type=float, default=360.0, help="fixed y for x-sweep")
    ap.add_argument("--x0", type=float, default=0.0)
    ap.add_argument("--x1", type=float, default=720.0)
    ap.add_argument("--n", type=int, default=13, help="number of sweep points")
    args = ap.parse_args()

    ckpt = Path(args.ckpt).resolve()
    ds_root = Path(args.ds).resolve()

    policy = load_policy(ckpt)
    ds = LeRobotDataset(repo_id="local/xarm_throws", root=str(ds_root))
    x = ds[0]
    state = x["observation.state"].float()
    env0  = x["observation.environment_state"].float().clone()

    # --- A) Target-conditioning sanity (must be >0) ---
    envA = env0.clone(); envB = env0.clone()
    envA[args.slot_x]=0.0;   envA[args.slot_y]=0.0
    envB[args.slot_x]=500.0; envB[args.slot_y]=720.0
    aA = infer_a0(policy, state, envA)
    aB = infer_a0(policy, state, envB)
    d_target = mad(aA,aB)

    # --- B) Smoothness sweep ---
    xs = np.linspace(args.x0, args.x1, args.n, dtype=np.float32)
    ys = float(args.y)
    outs = []
    for xv in xs:
        e = env0.clone()
        e[args.slot_x] = float(xv)
        e[args.slot_y] = ys
        outs.append(infer_a0(policy, state, e))
    outs = np.stack(outs, axis=0)

    step_changes = [mad(outs[i], outs[i+1]) for i in range(len(xs)-1)]
    mean_step = float(np.mean(step_changes))
    max_step  = float(np.max(step_changes))

    # "snap" heuristic: if most steps are ~0 and a few are huge, it's nearest-neighbor-ish
    frac_tiny = float(np.mean(np.array(step_changes) < (0.2 * mean_step + 1e-9)))

    # --- C) Midpoint test: is action(mid) between actions(left/right)? ---
    # Pick 3 points: left, mid, right
    xL, xM, xR = float(xs[len(xs)//2 - 1]), float(xs[len(xs)//2]), float(xs[len(xs)//2 + 1])
    def get(xv):
        e = env0.clone()
        e[args.slot_x] = float(xv)
        e[args.slot_y] = ys
        return infer_a0(policy, state, e)
    aL, aM, aR = get(xL), get(xM), get(xR)
    # check if aM is closer to average of neighbors than to either neighbor alone
    aAvg = 0.5*(aL+aR)
    d_mid_avg = mad(aM, aAvg)
    d_mid_L   = mad(aM, aL)
    d_mid_R   = mad(aM, aR)

    print("CKPT:", ckpt)
    print("DS  :", ds_root)
    print(f"target slot: ({args.slot_x},{args.slot_y})")
    print()
    print("A) Target-conditioning Δ (should be > 1e-4):", d_target)
    if d_target <= 1e-6:
        print("[FAIL] model ignores target (or ckpt too early / wrong ckpt)")
    elif d_target <= 1e-4:
        print("[MAYBE] very weak conditioning; train longer")
    else:
        print("[OK] target-conditioning present")

    print()
    print("B) Smoothness sweep x in [%.1f..%.1f], y=%.1f, n=%d" % (args.x0, args.x1, ys, args.n))
    print("   mean abs step change:", mean_step)
    print("   max  abs step change:", max_step)
    print("   frac tiny steps (snap-heuristic):", frac_tiny)
    if frac_tiny > 0.6 and max_step > 5*mean_step:
        print("[WARN] looks snappy (nearest-neighbor-ish). Not fatal early, but watch it.")
    else:
        print("[OK] looks reasonably smooth")

    print()
    print("C) Midpoint test (xL, xM, xR) =", (xL, xM, xR))
    print("   d(mid, avg(L,R)):", d_mid_avg)
    print("   d(mid, L)       :", d_mid_L)
    print("   d(mid, R)       :", d_mid_R)
    if d_mid_avg < min(d_mid_L, d_mid_R):
        print("[OK] midpoint behaves interpolative (closer to avg of neighbors)")
    else:
        print("[WARN] midpoint not clearly interpolative yet (could improve with training)")

if __name__ == "__main__":
    main()
