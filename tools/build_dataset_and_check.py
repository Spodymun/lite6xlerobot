#!/usr/bin/env python3
"""
tools/build_dataset_and_check.py

Build a single train-ready LeRobot dataset:

Pipeline:
1) combine runs -> temp dataset
2) fix task_index from labels
3) rebuild meta/tasks.parquet (with target_x_mm/target_y_mm)
4) patch env_state[4:6] = observation.task (IN PLACE on temp)
5) patch meta/info.json counts
6) integrity + diversity checks
7) move temp -> --out (single final output)

Usage:
python3 tools/build_dataset_and_check.py --out records/throws/fs_TRAIN_READY

Optional:
  --runs_root records/throws
  --glob "fs_*_job_wurf_*"
  --expect_tasks_min 20
  --slot_x 4 --slot_y 5
  --keep_tmp   (debug)
  --do_delta_check --ckpt_dir <.../checkpoints/last>
"""

from __future__ import annotations

import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------
# Helpers
# ---------------------------
def run(cmd, env=None):
    print("[RUN]", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True, env=env)

def patch_info_json(root: Path):
    info_path = root / "meta" / "info.json"
    data_pq = root / "data" / "chunk-000" / "file-000.parquet"
    epi_pq  = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"

    info = json.loads(info_path.read_text(encoding="utf-8"))
    n_frames = pq.read_table(data_pq, columns=["index"]).num_rows
    n_episodes = pq.read_table(epi_pq, columns=["episode_index"]).num_rows

    info["num_frames"] = int(n_frames)
    info["total_frames"] = int(n_frames)
    info["num_episodes"] = int(n_episodes)
    info["total_episodes"] = int(n_episodes)

    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[OK] patched info.json frames={n_frames} episodes={n_episodes}")

def rebuild_tasks_parquet_from_data(root: Path):
    """
    Rebuild meta/tasks.parquet from:
      data.task_index + data.observation.task
    Writes columns: task_index, target_x_mm, target_y_mm
    """
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    tp = meta_dir / "tasks.parquet"
    data_pq = root / "data" / "chunk-000" / "file-000.parquet"

    t = pq.read_table(data_pq, columns=["task_index", "observation.task"])
    ti = np.asarray(t["task_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    obj = np.asarray(t["observation.task"].to_numpy(zero_copy_only=False), dtype=object)

    first = {}
    for i in range(len(ti)):
        k = int(ti[i])
        if k not in first:
            x, y = map(float, obj[i])
            first[k] = (x, y)

    keys = sorted(first.keys())
    df = pd.DataFrame(
        {
            "task_index": keys,
            "target_x_mm": [first[k][0] for k in keys],
            "target_y_mm": [first[k][1] for k in keys],
        }
    )
    # nice stable index (not required, but handy)
    df.index = [f"target_{x:.1f}_{y:.1f}" for x, y in zip(df["target_x_mm"], df["target_y_mm"])]

    df.to_parquet(tp)
    print(f"[OK] wrote meta/tasks.parquet rows={len(df)} task_index_range={min(keys)}..{max(keys)}")

def validate_dataset_basic(root: Path, expect_tasks_min: int = 1):
    data_pq = root / "data" / "chunk-000" / "file-000.parquet"
    epi_pq  = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    info_p  = root / "meta" / "info.json"
    tasks_p = root / "meta" / "tasks.parquet"

    for p in [data_pq, epi_pq, info_p]:
        if not p.exists():
            raise SystemExit(f"[FAIL] missing required file: {p}")

    cols = [
        "index","episode_index","frame_index","task_index",
        "action","observation.state","observation.task","observation.environment_state"
    ]
    t = pq.read_table(data_pq, columns=cols)

    a0 = t["action"][0].as_py()
    s0 = t["observation.state"][0].as_py()
    k0 = t["observation.task"][0].as_py()
    e0 = t["observation.environment_state"][0].as_py()

    print("[OK] required columns present")
    print("action dim:", len(a0))
    print("state dim :", len(s0))
    print("task dim  :", len(k0))
    print("env dim   :", len(e0))

    if len(a0) != 13:
        raise SystemExit("[FAIL] action is not 13D")
    if len(s0) != 7:
        raise SystemExit("[FAIL] observation.state is not 7D")
    if len(k0) != 2:
        raise SystemExit("[FAIL] observation.task is not 2D")
    if len(e0) != 9:
        raise SystemExit("[FAIL] observation.environment_state is not 9D")

    idx = np.asarray(t["index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    if not (idx.min() == 0 and idx.max() == len(idx)-1):
        raise SystemExit(f"[FAIL] index not contiguous: {idx.min()}..{idx.max()} expected 0..{len(idx)-1}")
    print("[OK] global index is contiguous")

    # task diversity
    ti = np.asarray(t["task_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    obj = np.asarray(t["observation.task"].to_numpy(zero_copy_only=False), dtype=object)

    unique_task_index = len(set(ti.tolist()))
    unique_tasks = len({tuple(map(float, v)) for v in obj})
    print("unique task_index overall:", unique_task_index)
    print("unique observation.task overall:", unique_tasks)

    if unique_task_index < expect_tasks_min:
        raise SystemExit(f"[FAIL] task_index diversity too low: {unique_task_index} < {expect_tasks_min}")
    print("[OK] Task diversity looks correct")

    # per-episode consistency (each episode should have exactly one target)
    epi = np.asarray(t["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    per = {}
    for e, v in zip(epi, obj):
        per.setdefault(int(e), set()).add(tuple(map(float, v)))
    counts = [len(per[e]) for e in sorted(per)]
    print("episodes:", len(counts))
    print("unique tasks per episode: min/median/max =", min(counts), int(np.median(counts)), max(counts))
    if max(counts) != 1:
        raise SystemExit("[FAIL] Some episodes contain multiple targets (expected exactly 1)")
    print("[OK] Each episode has exactly one target")

    # meta/tasks.parquet
    if not tasks_p.exists():
        raise SystemExit("[FAIL] meta/tasks.parquet missing (should exist after rebuild)")
    tasks = pd.read_parquet(tasks_p).reset_index(drop=True)
    need = {"task_index","target_x_mm","target_y_mm"}
    if not need.issubset(set(tasks.columns)):
        raise SystemExit(f"[FAIL] tasks.parquet missing columns. has={list(tasks.columns)} need={sorted(need)}")
    print("[OK] meta/tasks.parquet present and has xy columns")

def patch_task_into_env_inplace(root: Path, slot_x: int = 4, slot_y: int = 5):
    """
    In-place patch:
      observation.environment_state[slot_x] = observation.task[0]
      observation.environment_state[slot_y] = observation.task[1]
    """
    data_pq = root/"data/chunk-000/file-000.parquet"
    t = pq.read_table(data_pq)

    env = np.asarray(t["observation.environment_state"].to_numpy(zero_copy_only=False), dtype=object)
    task = np.asarray(t["observation.task"].to_numpy(zero_copy_only=False), dtype=object)

    env2 = []
    for e, xy in zip(env, task):
        e = np.array(e, dtype=np.float32)
        e[slot_x] = float(xy[0])
        e[slot_y] = float(xy[1])
        env2.append(e.tolist())

    cols = []
    for name in t.column_names:
        if name == "observation.environment_state":
            cols.append(pa.array(env2, type=pa.list_(pa.float32())))
        else:
            cols.append(t[name])

    t2 = pa.table(dict(zip(t.column_names, cols)))
    pq.write_table(t2, data_pq)
    print(f"[OK] patched env_state[{slot_x},{slot_y}] = task (in place)")

def validate_taskinenv_exact(root: Path, slot_x: int = 4, slot_y: int = 5, sample: int = 2000):
    t = pq.read_table(root/"data/chunk-000/file-000.parquet",
                      columns=["observation.task","observation.environment_state"])
    N = t.num_rows
    idx = np.linspace(0, N-1, num=min(sample, N), dtype=int)

    task = np.asarray(t["observation.task"].to_numpy(zero_copy_only=False), dtype=object)[idx]
    env  = np.asarray(t["observation.environment_state"].to_numpy(zero_copy_only=False), dtype=object)[idx]

    task = np.array([[float(a[0]), float(a[1])] for a in task], dtype=np.float32)
    envxy = np.array([[float(e[slot_x]), float(e[slot_y])] for e in env], dtype=np.float32)

    diff = envxy - task
    mae = np.mean(np.abs(diff), axis=0)
    mx  = np.max(np.abs(diff), axis=0)
    print("env-task mean_abs:", mae, "max_abs:", mx)
    if not np.allclose(mae, 0.0) or not np.allclose(mx, 0.0):
        raise SystemExit("[FAIL] env_state does not exactly match task after patch")
    print("[OK] env_state matches task exactly (sampled)")

def delta_target_check_act(ckpt_dir: Path, dataset_root: Path, slot_x: int = 4, slot_y: int = 5):
    """
    Loads ACT checkpoint and checks sensitivity when env[slot_x:slot_y] changes.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import torch

    pm = ckpt_dir/"pretrained_model"
    pm = pm if pm.is_dir() else ckpt_dir

    cfg = PreTrainedConfig.from_pretrained(str(pm))
    policy = ACTPolicy.from_pretrained(str(pm), config=cfg)
    policy.eval()

    ds = LeRobotDataset(repo_id="local/xarm_throws", root=str(dataset_root))
    x = ds[0]
    state = x["observation.state"].float()
    env0  = x["observation.environment_state"].float().clone()

    def add_batch(x):
        if x.dim()==1: return x.unsqueeze(0)
        if x.dim()==2: return x.unsqueeze(0)
        return x

    def infer(env_vec):
        batch = {
            "observation.state": add_batch(state),
            "observation.environment_state": add_batch(env_vec),
        }
        with torch.no_grad():
            out = policy.model(batch)
            act = out[0] if isinstance(out, tuple) else out
        a0 = act[0,0].detach().cpu().numpy() if act.dim()==3 else act[0].detach().cpu().numpy()
        return a0

    envA = env0.clone()
    envB = env0.clone()
    envA[slot_x] = 0.0;   envA[slot_y] = 0.0
    envB[slot_x] = 500.0; envB[slot_y] = 720.0

    aA = infer(envA)
    aB = infer(envB)
    d = float(np.mean(np.abs(aA-aB)))
    print("Δ when TARGET changes via env slots:", d)
    if d > 1e-4:
        print("[PASS] clear target-conditioning signal")
    elif d > 1e-6:
        print("[MAYBE] weak but nonzero — train longer (e.g. 500 steps)")
    else:
        print("[FAIL] still not target-conditioned")


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="FINAL train-ready dataset output root (single folder).")
    ap.add_argument("--runs_root", default="records/throws")
    ap.add_argument("--glob", default="fs_*_job_wurf_*")
    ap.add_argument("--expect_tasks_min", type=int, default=10)

    ap.add_argument("--slot_x", type=int, default=4)
    ap.add_argument("--slot_y", type=int, default=5)

    ap.add_argument("--keep_tmp", action="store_true", help="Keep temp build folder for debugging.")
    ap.add_argument("--do_delta_check", action="store_true")
    ap.add_argument("--ckpt_dir", default=None)

    args = ap.parse_args()
    repo_root = Path(".").resolve()

    out = (repo_root / args.out).resolve()
    if out.exists():
        raise SystemExit(f"[FAIL] --out already exists: {out}\n(choose a new path or delete it manually)")

    # temp build path next to the final output
    tmp = out.parent / (out.name + "_tmp_build_" + time.strftime("%Y%m%d_%H%M%S"))
    if tmp.exists():
        shutil.rmtree(tmp)

    tools_dir = repo_root / "tools"
    combine_py = tools_dir / "combine_runs_to_dataset.py"
    fix_task_py = tools_dir / "fix_task_index_from_labels.py"

    for p in [combine_py, fix_task_py]:
        if not p.exists():
            raise SystemExit(f"[FAIL] missing required tool script: {p}")

    try:
        # 1) combine into tmp
        run([
            sys.executable, str(combine_py),
            "--out", str(tmp),
            "--runs_root", str(repo_root / args.runs_root),
            "--glob", args.glob,
        ])

        # 2) fix task_index based on labels
        run([
            sys.executable, str(fix_task_py),
            "--combined_root", str(tmp),
            "--runs_root", str(repo_root / args.runs_root),
            "--glob", args.glob,
        ])

        # 3) rebuild tasks.parquet and patch info.json
        rebuild_tasks_parquet_from_data(tmp)
        patch_info_json(tmp)

        print("\n================================================================================")
        print("1) COMBINED DATASET CHECKS (before env patch)")
        print("================================================================================")
        validate_dataset_basic(tmp, expect_tasks_min=args.expect_tasks_min)

        # 4) patch env[slot_x,slot_y] = task (in place)
        print("\n================================================================================")
        print("2) PATCH TASK INTO ENV (in place)")
        print("================================================================================")
        patch_task_into_env_inplace(tmp, slot_x=args.slot_x, slot_y=args.slot_y)

        # 5) re-write derived meta
        rebuild_tasks_parquet_from_data(tmp)
        patch_info_json(tmp)

        print("\n================================================================================")
        print("3) FINAL TRAIN-READY CHECKS (after env patch)")
        print("================================================================================")
        validate_dataset_basic(tmp, expect_tasks_min=args.expect_tasks_min)
        validate_taskinenv_exact(tmp, slot_x=args.slot_x, slot_y=args.slot_y)

        # optional delta check
        if args.do_delta_check:
            if not args.ckpt_dir:
                raise SystemExit("[FAIL] --do_delta_check requires --ckpt_dir <.../checkpoints/last>")
            ckpt_dir = Path(args.ckpt_dir).resolve()
            print("\n================================================================================")
            print("4) ΔTARGET CHECK (ACT)")
            print("================================================================================")
            delta_target_check_act(ckpt_dir, tmp, slot_x=args.slot_x, slot_y=args.slot_y)

        # 6) atomically move tmp -> out
        tmp.rename(out)
        print("\n[DONE] Train-ready dataset written to:", out)

    finally:
        # cleanup tmp if it still exists
        if tmp.exists() and not args.keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    print("\nTrain command (CPU-only):")
    print('  export PYTHONPATH="$PWD:${PYTHONPATH}"')
    print('  export CUDA_VISIBLE_DEVICES=""')
    print(f'  lerobot-train --config_path configs/act_env_smoke.yaml --dataset.root "{out}" --steps 200000 --log_freq 50 --save_checkpoint true --save_freq 5000')


if __name__ == "__main__":
    main()
