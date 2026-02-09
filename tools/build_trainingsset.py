#!/usr/bin/env python3
"""
tools/build_all_throws_super.py

Example:
python3 tools/build_all_throws_super.py \
  --out records/throws/fs_2026-02-10_ALL_THROWS_COMBINED \
  --runs_root records/throws \
  --glob "fs_*_job_wurf_*" \
  --patch_targets_from_wuerfe \
  --wuerfe_dir wuerfe
"""

import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


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
    Ensure meta/tasks.parquet contains:
      task_index, target_x_mm, target_y_mm (and a stable string index)
    Source of truth: data.observation.task grouped by data.task_index
    """
    tp = root / "meta" / "tasks.parquet"
    bak = root / "meta" / "tasks.parquet.bak_auto"
    if tp.exists() and not bak.exists():
        tp.rename(bak)
        print("[OK] backup old tasks.parquet ->", bak)

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
    df.index = [f"target_{x:.1f}_{y:.1f}" for x, y in zip(df["target_x_mm"], df["target_y_mm"])]
    df.to_parquet(tp)
    print(f"[OK] wrote tasks.parquet rows={len(df)} task_index_range={min(keys)}..{max(keys)}")


def validate_dataset(root: Path, expect_tasks_min: int = 1):
    data_pq = root / "data" / "chunk-000" / "file-000.parquet"
    epi_pq  = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    tasks_pq = root / "meta" / "tasks.parquet"

    # Basic existence
    for p in [data_pq, epi_pq, tasks_pq, root/"meta/info.json"]:
        if not p.exists():
            raise SystemExit(f"[FAIL] missing required file: {p}")

    # Index continuity & uniqueness
    data = pq.read_table(data_pq, columns=["index","episode_index","frame_index","task_index","observation.task"])
    epi  = pq.read_table(epi_pq,  columns=["episode_index","length","dataset_from_index","dataset_to_index","task_index"])

    idx = np.asarray(data["index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    if not (idx.min() == 0 and idx.max() == len(idx)-1):
        raise SystemExit(f"[FAIL] index range not contiguous: {idx.min()}..{idx.max()} vs expected 0..{len(idx)-1}")
    if len(set(idx.tolist())) != len(idx):
        raise SystemExit("[FAIL] index has duplicates")

    ep_idx = np.asarray(data["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    if len(set(ep_idx.tolist())) != epi.num_rows:
        raise SystemExit(f"[FAIL] unique episodes in data != meta episodes: {len(set(ep_idx.tolist()))} vs {epi.num_rows}")

    # frame_index contiguous for first 30 episodes
    fi = np.asarray(data["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    for e in range(min(epi.num_rows, 30)):
        m = (ep_idx == e)
        if not m.any():
            raise SystemExit(f"[FAIL] episode {e} missing in data")
        f = np.sort(fi[m])
        if not (f[0] == 0 and np.all(np.diff(f) == 1)):
            raise SystemExit(f"[FAIL] frame_index not contiguous in episode {e} (min={f[0]} max={f[-1]})")

    # tasks.parquet must contain xy
    tasks = pd.read_parquet(tasks_pq).reset_index().rename(columns={"index":"task_name"})
    need = {"task_index","target_x_mm","target_y_mm"}
    if not need.issubset(set(tasks.columns)):
        raise SystemExit(f"[FAIL] tasks.parquet missing columns. has={list(tasks.columns)} need={sorted(need)}")

    # observation.task must match tasks lookup for random sample
    tasks = tasks.sort_values("task_index").reset_index(drop=True)
    ti = np.asarray(data["task_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    obj = np.asarray(data["observation.task"].to_numpy(zero_copy_only=False), dtype=object)

    rng = np.random.default_rng(0)
    idxs = rng.integers(0, len(ti), size=min(20000, len(ti)))
    for k in idxs:
        tix = int(ti[k])
        x,y = map(float, obj[k])
        tx = float(tasks.loc[tix, "target_x_mm"])
        ty = float(tasks.loc[tix, "target_y_mm"])
        if abs(x-tx) > 1e-3 or abs(y-ty) > 1e-3:
            raise SystemExit(f"[FAIL] tasks lookup mismatch at row {int(k)} task_index={tix}: obs=({x},{y}) tasks=({tx},{ty})")

    # task diversity sanity
    unique_tasks = len({tuple(map(float, v)) for v in obj[: min(5000, len(obj))]})
    unique_task_index = len(set(ti.tolist()))
    if unique_task_index < expect_tasks_min:
        raise SystemExit(f"[FAIL] task_index diversity too low: {unique_task_index}")
    print(f"[OK] validate: frames={len(idx)} episodes={epi.num_rows} task_index_unique={unique_task_index} obs_task_unique(first5k)={unique_tasks}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="combined dataset root output path")
    ap.add_argument("--runs_root", default="records/throws")
    ap.add_argument("--glob", default="fs_*_job_wurf_*")
    ap.add_argument("--patch_targets_from_wuerfe", action="store_true")
    ap.add_argument("--wuerfe_dir", default="wuerfe")
    ap.add_argument("--expect_tasks_min", type=int, default=10)

    args = ap.parse_args()
    repo_root = Path(".").resolve()

    out = (repo_root / args.out).resolve()
    if out.exists():
        raise SystemExit(f"[FAIL] out already exists: {out}")

    tools_dir = repo_root / "tools"
    combine_py = tools_dir / "combine_runs_to_dataset.py"
    fix_task_py = tools_dir / "fix_task_index_from_labels.py"
    patch_targets_py = tools_dir / "patch_targets_from_wuerfe.py"

    for p in [combine_py, fix_task_py]:
        if not p.exists():
            raise SystemExit(f"[FAIL] missing required tool script: {p}")

    # 1) combine
    run([
        sys.executable, str(combine_py),
        "--out", str(out),
        "--runs_root", str(repo_root / args.runs_root),
        "--glob", args.glob,
    ])

    # 2) optional: patch throw_label.json targets from wuerfe/wurf_*.json
    if args.patch_targets_from_wuerfe:
        if not patch_targets_py.exists():
            raise SystemExit(f"[FAIL] requested --patch_targets_from_wuerfe but missing: {patch_targets_py}")
        run([
            sys.executable, str(patch_targets_py),
            "--wuerfe_dir", str(repo_root / args.wuerfe_dir),
            "--records_dir", str(repo_root / args.runs_root),
            "--backup",
        ])

    # 3) fix task_index everywhere based on labels into the combined root
    run([
        sys.executable, str(fix_task_py),
        "--combined_root", str(out),
        "--runs_root", str(repo_root / args.runs_root),
        "--glob", args.glob,
    ])

    # 4) always rebuild tasks.parquet with xy columns (prevents your “task_name/task_index only” bug)
    rebuild_tasks_parquet_from_data(out)

    # 5) patch info.json counts
    patch_info_json(out)

    # 6) validate
    validate_dataset(out, expect_tasks_min=args.expect_tasks_min)

    print("\n[DONE] dataset ready:", out)


if __name__ == "__main__":
    main()
