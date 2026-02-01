#!/usr/bin/env python3
"""
tools/patch_actions.py (PARAMETER POLICY)

Makes a LeRobot dataset trainable for a *parameter policy*:

  (observation.state + observation.task[target_xy]) -> action(params)

Where:
  observation.task = target_xy_mm (constant per frame, from job.json)
  action = [pos1(6), pos2(6), release_progress(1)]  -> 13 floats, constant per frame

No action_raw. No rate limiting. No servo-style next-state actions.

Usage:
  python3 tools/patch_actions.py \
    --parquet /path/to/dataset/data/chunk-000/file-000.parquet \
    --job     /path/to/wuerfe/wurf_1.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# -----------------------------
# Arrow helpers
# -----------------------------

def to_array(a):
    """ChunkedArray -> Array"""
    if isinstance(a, pa.ChunkedArray):
        return pa.concat_arrays(list(a.chunks))
    return a


def upsert_column(table: pa.Table, name: str, col: pa.Array) -> pa.Table:
    """Replace column if exists, else append."""
    if name in table.column_names:
        idx = table.column_names.index(name)
        return table.set_column(idx, name, col)
    return table.append_column(name, col)


def make_task_array(n_rows: int, x: float, y: float) -> pa.FixedSizeListArray:
    """FixedSizeList<float32>[2] per row: [x, y]"""
    values = pa.array([x, y] * n_rows, type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(values, list_size=2)


def make_action_params_array(n_rows: int, params: np.ndarray) -> pa.FixedSizeListArray:
    """
    params: (13,) float32
    returns FixedSizeList<float32>[13] repeated for n_rows
    """
    params = np.asarray(params, dtype=np.float32).reshape(13,)
    flat = np.tile(params, n_rows).astype(np.float32)
    flat_arr = pa.array(flat.tolist(), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat_arr, list_size=13)


# -----------------------------
# Job loader
# -----------------------------

def load_job(job_path: Path) -> dict:
    with job_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_job_fields(job: dict, job_path: Path):
    needed = ["target_xy_mm", "pos1", "pos2", "release_progress"]
    for k in needed:
        if k not in job:
            raise RuntimeError(f"{job_path} missing '{k}'. Required: {needed}")

    if not (isinstance(job["pos1"], list) and len(job["pos1"]) == 6):
        raise RuntimeError(f"{job_path}: 'pos1' must be list[6] (rad).")
    if not (isinstance(job["pos2"], list) and len(job["pos2"]) == 6):
        raise RuntimeError(f"{job_path}: 'pos2' must be list[6] (rad).")

    rp = float(job["release_progress"])
    if not (0.0 <= rp <= 1.0):
        raise RuntimeError(f"{job_path}: release_progress must be in [0,1]. Got: {rp}")


# -----------------------------
# info.json patch (and enforce feature order)
# -----------------------------

def patch_info_json(parquet_path: Path, *, want_feature_order: list[str]) -> None:
    """
    Ensures meta/info.json contains:
      - features['observation.task'] shape [2]
      - features['action'] shape [13]
    And reorders features keys to exactly want_feature_order (so replay schema matches).
    """
    dataset_root = parquet_path.parents[2]  # .../data/chunk-000/file-000.parquet -> dataset_root
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"meta/info.json not found: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    feats = info.get("features")
    if not isinstance(feats, dict):
        feats = {}
        info["features"] = feats

    # Ensure observation.task
    feats.setdefault("observation.task", {
        "dtype": "float32",
        "names": ["target_x_mm", "target_y_mm"],
        "shape": [2],
    })

    # Ensure action = 13 params
    feats["action"] = {
        "dtype": "float32",
        "names": [
            "pos1_j1","pos1_j2","pos1_j3","pos1_j4","pos1_j5","pos1_j6",
            "pos2_j1","pos2_j2","pos2_j3","pos2_j4","pos2_j5","pos2_j6",
            "release_progress",
        ],
        "shape": [13],
    }

    # Reorder features keys to match parquet columns exactly
    new_feats = OrderedDict()
    for k in want_feature_order:
        if k not in feats:
            raise RuntimeError(f"info.json features missing '{k}' but parquet has it. Fix your dataset.")
        new_feats[k] = feats[k]
    info["features"] = new_feats

    # backup once
    bak = info_path.with_suffix(info_path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(info_path, bak)
        print(f"[ok] info.json backup written: {bak}")

    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[ok] patched info.json: {info_path}")


def reorder_table_to_feature_order(table: pa.Table, feature_order: list[str]) -> pa.Table:
    missing = [c for c in feature_order if c not in table.column_names]
    if missing:
        raise RuntimeError(f"Parquet missing columns required by features order: {missing}")

    # Keep EXACT order and drop any extra columns not in features
    extra = [c for c in table.column_names if c not in feature_order]
    if extra:
        # If you ever add debug columns, they MUST also be declared in info.json features.
        # Otherwise lerobot_replay will fail schema casting.
        print(f"[warn] dropping extra columns not in features: {extra}")

    return table.select(feature_order)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help=".../data/chunk-000/file-000.parquet")
    ap.add_argument("--job", required=True, help=".../wuerfe/wurf_N.json (must contain target_xy_mm,pos1,pos2,release_progress)")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    parquet_path = Path(args.parquet).expanduser().resolve()
    job_path = Path(args.job).expanduser().resolve()

    if not parquet_path.exists():
        raise SystemExit(f"Parquet not found: {parquet_path}")
    if not job_path.exists():
        raise SystemExit(f"Job not found: {job_path}")

    if not args.no_backup:
        bak = parquet_path.with_suffix(parquet_path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(parquet_path, bak)
            print(f"[ok] parquet backup written: {bak}")

    job = load_job(job_path)
    require_job_fields(job, job_path)

    tx, ty = job["target_xy_mm"]
    pos1 = np.array(job["pos1"], dtype=np.float32)
    pos2 = np.array(job["pos2"], dtype=np.float32)
    rp = np.array([float(job["release_progress"])], dtype=np.float32)

    params = np.concatenate([pos1, pos2, rp], axis=0)  # (13,)

    table = pq.read_table(parquet_path)

    # write observation.task + action
    task_arr = make_task_array(table.num_rows, float(tx), float(ty))
    action_arr = make_action_params_array(table.num_rows, params)

    table = upsert_column(table, "observation.task", task_arr)
    table = upsert_column(table, "action", action_arr)

    # preserve parquet metadata if present
    if table.schema.metadata:
        table = table.cast(table.schema.with_metadata(table.schema.metadata))

    # enforce consistent (replay-safe) order:
    # we keep existing base columns in their current order, then ensure task+action are placed as in features.
    # simplest: define feature order explicitly as the exact final table order you want.
    base_cols = [c for c in table.column_names if c not in ("observation.task",)]
    # put observation.task after task_index if it exists, else at end
    if "task_index" in base_cols:
        idx = base_cols.index("task_index") + 1
        feature_order = base_cols[:idx] + ["observation.task"] + base_cols[idx:]
    else:
        feature_order = base_cols + ["observation.task"]

    # ensure action is first (LeRobot default), if it exists in feature_order move to front
    if "action" in feature_order:
        feature_order.remove("action")
    feature_order = ["action"] + [c for c in feature_order if c != "action"]

    # reorder parquet to match features order
    table = reorder_table_to_feature_order(table, feature_order)

    # patch info.json to match
    patch_info_json(parquet_path, want_feature_order=feature_order)

    pq.write_table(table, parquet_path)

    print("[ok] patched dataset for PARAMETER policy")
    print(f"     parquet       : {parquet_path}")
    print(f"     job           : {job_path}")
    print(f"     target_xy_mm  : [{float(tx)}, {float(ty)}]")
    print(f"     action params : pos1+pos2+rp (13D)")
    print(f"     rows          : {table.num_rows}")
    print(f"     columns       : {table.column_names}")


if __name__ == "__main__":
    main()
