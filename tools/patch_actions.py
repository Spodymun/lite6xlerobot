#!/usr/bin/env python3
"""
tools/patch_actions.py (PARAMETER POLICY, REPRODUCIBLE, THROW_LABEL-FIRST)

Makes a LeRobot dataset trainable for a *parameter policy*:

  input:  observation.environment_state = concat(observation.state, observation.task)
          where:
            observation.state = 7 floats  (6 joints + gripper)
            observation.task  = 2 floats  (target_x_mm, target_y_mm)
          => environment_state = 9 floats

  output: action params = [pos1(6), pos2(6), release_progress(1)] -> 13 floats

Sources of truth:
  - Default: <dataset_root>/meta/throw_label.json
  - Optional: --label <path/to/throw_label.json>
  - Legacy fallback (optional): --job <path/to/wurf_N.json>

Also patches:
  - meta/info.json  (features + stable order)
  - meta/stats.json (stats consistent with current parquet schema)
And enforces a stable, replay-safe column order.

Usage:
  # patch an entire run (recommended)
  python3 tools/patch_actions.py --dataset-root records/throws/<RUN>

  # explicit label file
  python3 tools/patch_actions.py --dataset-root records/throws/<RUN> --label records/throws/<RUN>/meta/throw_label.json

  # patch single parquet (rare)
  python3 tools/patch_actions.py --parquet records/throws/<RUN>/data/chunk-000/file-000.parquet --label records/throws/<RUN>/meta/throw_label.json

Notes:
- Creates .bak backups once per file unless --no-backup.
- Idempotent: safe to run multiple times; use --force to rewrite aggressively.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Optional, Tuple

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


def fixed_list(values: np.ndarray, list_size: int) -> pa.FixedSizeListArray:
    """
    values: (N, list_size) float32
    """
    values = np.asarray(values, dtype=np.float32)
    if not (values.ndim == 2 and values.shape[1] == list_size):
        raise RuntimeError(f"fixed_list expected (N,{list_size}), got {values.shape}")
    flat = pa.array(values.reshape(-1).tolist(), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, list_size=list_size)


def make_task_array(n_rows: int, x: float, y: float) -> pa.FixedSizeListArray:
    """FixedSizeList<float32>[2] per row: [x, y]"""
    v = np.tile(np.array([[x, y]], dtype=np.float32), (n_rows, 1))
    return fixed_list(v, 2)


def make_action_params_array(n_rows: int, params13: np.ndarray) -> pa.FixedSizeListArray:
    """FixedSizeList<float32>[13] repeated for n_rows"""
    params13 = np.asarray(params13, dtype=np.float32).reshape(13,)
    v = np.tile(params13[None, :], (n_rows, 1))
    return fixed_list(v, 13)


def compute_env_state(table: pa.Table) -> pa.FixedSizeListArray:
    """
    observation.environment_state = concat(observation.state (7), observation.task (2)) -> 9
    Expects both columns to exist and be list-like.
    """
    if "observation.state" not in table.column_names:
        raise RuntimeError("Missing column: observation.state")
    if "observation.task" not in table.column_names:
        raise RuntimeError("Missing column: observation.task")

    s = to_array(table["observation.state"]).to_numpy(zero_copy_only=False)
    t = to_array(table["observation.task"]).to_numpy(zero_copy_only=False)

    s = np.asarray(s, dtype=object)  # each row is list/ndarray
    t = np.asarray(t, dtype=object)

    out = np.zeros((table.num_rows, 9), dtype=np.float32)
    for i in range(table.num_rows):
        sv = np.asarray(s[i], dtype=np.float32).reshape(-1)
        tv = np.asarray(t[i], dtype=np.float32).reshape(-1)
        if sv.shape[0] != 7:
            raise RuntimeError(f"row {i}: observation.state expected 7, got {sv.shape[0]}")
        if tv.shape[0] != 2:
            raise RuntimeError(f"row {i}: observation.task expected 2, got {tv.shape[0]}")
        out[i, :7] = sv
        out[i, 7:] = tv
    return fixed_list(out, 9)


# -----------------------------
# Input sources (label/job)
# -----------------------------

def _safe_joint6(x) -> Optional[np.ndarray]:
    if not isinstance(x, (list, tuple)) or len(x) != 6:
        return None
    try:
        return np.array([float(v) for v in x], dtype=np.float32)
    except Exception:
        return None


def load_throw_label(label_path: Path) -> dict:
    with label_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_label_fields(label: dict, label_path: Path) -> Tuple[float, float, np.ndarray, np.ndarray, float]:
    # label keys we need:
    # target_xy_mm: [x,y]
    # pos1: [6]
    # pos2: [6]
    # release_progress: float in [0,1]
    if "target_xy_mm" not in label or label["target_xy_mm"] is None:
        raise RuntimeError(f"{label_path}: missing target_xy_mm")
    if not (isinstance(label["target_xy_mm"], list) and len(label["target_xy_mm"]) == 2):
        raise RuntimeError(f"{label_path}: target_xy_mm must be list[2]")

    pos1 = _safe_joint6(label.get("pos1"))
    pos2 = _safe_joint6(label.get("pos2"))
    if pos1 is None or pos2 is None:
        raise RuntimeError(f"{label_path}: pos1/pos2 must be list[6]")

    rp = label.get("release_progress", None)
    if rp is None:
        raise RuntimeError(f"{label_path}: missing release_progress")
    rp = float(rp)
    if not (0.0 <= rp <= 1.0):
        raise RuntimeError(f"{label_path}: release_progress must be in [0,1], got {rp}")

    tx, ty = float(label["target_xy_mm"][0]), float(label["target_xy_mm"][1])
    return tx, ty, pos1, pos2, rp


def load_job(job_path: Path) -> dict:
    with job_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_job_fields(job: dict, job_path: Path) -> Tuple[float, float, np.ndarray, np.ndarray, float]:
    needed = ["target_xy_mm", "pos1", "pos2", "release_progress"]
    for k in needed:
        if k not in job:
            raise RuntimeError(f"{job_path} missing '{k}'. Required: {needed}")

    if not (isinstance(job["target_xy_mm"], list) and len(job["target_xy_mm"]) == 2):
        raise RuntimeError(f"{job_path}: 'target_xy_mm' must be list[2]")

    pos1 = _safe_joint6(job.get("pos1"))
    pos2 = _safe_joint6(job.get("pos2"))
    if pos1 is None or pos2 is None:
        raise RuntimeError(f"{job_path}: pos1/pos2 must be list[6] (rad).")

    rp = float(job["release_progress"])
    if not (0.0 <= rp <= 1.0):
        raise RuntimeError(f"{job_path}: release_progress must be in [0,1]. Got: {rp}")

    tx, ty = float(job["target_xy_mm"][0]), float(job["target_xy_mm"][1])
    return tx, ty, pos1, pos2, rp


# -----------------------------
# meta/info.json patch
# -----------------------------

def patch_info_json(dataset_root: Path, *, feature_order: list[str]) -> None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"meta/info.json not found: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    feats = info.get("features")
    if not isinstance(feats, dict):
        feats = {}
        info["features"] = feats

    feats["action"] = {
        "dtype": "float32",
        "names": [
            "pos1_j1","pos1_j2","pos1_j3","pos1_j4","pos1_j5","pos1_j6",
            "pos2_j1","pos2_j2","pos2_j3","pos2_j4","pos2_j5","pos2_j6",
            "release_progress",
        ],
        "shape": [13],
    }

    feats["observation.task"] = {
        "dtype": "float32",
        "names": ["target_x_mm", "target_y_mm"],
        "shape": [2],
    }

    feats["observation.environment_state"] = {
        "dtype": "float32",
        "names": [
            "joint1.pos","joint2.pos","joint3.pos","joint4.pos","joint5.pos","joint6.pos","gripper.pos",
            "target_x_mm","target_y_mm",
        ],
        "shape": [9],
    }

    new_feats = OrderedDict()
    for k in feature_order:
        if k not in feats:
            raise RuntimeError(f"info.json missing feature '{k}' (but parquet has it).")
        new_feats[k] = feats[k]
    info["features"] = new_feats

    bak = info_path.with_suffix(info_path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(info_path, bak)
        print(f"[ok] info.json backup: {bak}")

    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[ok] patched info.json: {info_path}")


# -----------------------------
# meta/stats.json patch
# -----------------------------

def _stats_for_matrix(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    N, D = x.shape
    if not np.isfinite(x).all():
        raise RuntimeError("Non-finite values in dataset (NaN/Inf). Fix before training.")

    qs = np.quantile(x, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0, method="linear")
    return {
        "min":  np.min(x, axis=0).tolist(),
        "max":  np.max(x, axis=0).tolist(),
        "mean": np.mean(x, axis=0).tolist(),
        "std":  np.std(x, axis=0, ddof=0).tolist(),
        "count": [int(N)] * D,
        "q01": qs[0].tolist(),
        "q10": qs[1].tolist(),
        "q50": qs[2].tolist(),
        "q90": qs[3].tolist(),
        "q99": qs[4].tolist(),
    }


def patch_stats_json(dataset_root: Path, table: pa.Table, feature_order: list[str]) -> None:
    stats_path = dataset_root / "meta" / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"meta/stats.json not found: {stats_path}")

    stats = OrderedDict()
    for name in feature_order:
        col = to_array(table[name])

        if pa.types.is_integer(col.type) or pa.types.is_floating(col.type):
            x = np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float64)
            stats[name] = _stats_for_matrix(x)
            continue

        if pa.types.is_fixed_size_list(col.type):
            obj = np.asarray(col.to_numpy(zero_copy_only=False), dtype=object)
            D = col.type.list_size
            x = np.zeros((len(obj), D), dtype=np.float64)
            for i in range(len(obj)):
                v = np.asarray(obj[i], dtype=np.float64).reshape(-1)
                if v.shape[0] != D:
                    raise RuntimeError(f"{name}: expected {D}, got {v.shape[0]} at row {i}")
                x[i] = v
            stats[name] = _stats_for_matrix(x)
            continue

        raise RuntimeError(f"Unsupported column type for stats: {name} -> {col.type}")

    bak = stats_path.with_suffix(stats_path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(stats_path, bak)
        print(f"[ok] stats.json backup: {bak}")

    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[ok] patched stats.json: {stats_path}")


# -----------------------------
# Column order
# -----------------------------

def enforce_feature_order(table: pa.Table, feature_order: list[str]) -> pa.Table:
    missing = [c for c in feature_order if c not in table.column_names]
    if missing:
        raise RuntimeError(f"Parquet missing required columns: {missing}")

    extra = [c for c in table.column_names if c not in feature_order]
    if extra:
        print(f"[warn] dropping extra columns not in features: {extra}")

    return table.select(feature_order)


def default_feature_order(table: pa.Table) -> list[str]:
    """
    - action first
    - observation.task after task_index (if present) else near end
    - observation.environment_state right after observation.task
    """
    cols = list(table.column_names)

    if "action" in cols:
        cols.remove("action")
    cols = ["action"] + cols

    for name in ["observation.task", "observation.environment_state"]:
        if name in cols:
            cols.remove(name)

    insert_at = cols.index("task_index") + 1 if "task_index" in cols else len(cols)
    cols[insert_at:insert_at] = ["observation.task", "observation.environment_state"]
    return cols


# -----------------------------
# Parquet discovery
# -----------------------------

def iter_parquets(dataset_root: Path) -> Iterable[Path]:
    data_dir = dataset_root / "data"
    return sorted(data_dir.glob("chunk-*/file-*.parquet"))


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--parquet", help=".../<run>/data/chunk-000/file-000.parquet")
    g.add_argument("--dataset-root", help=".../<run> (patches all data/chunk-*/file-*.parquet)")

    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--label", help="Path to throw_label.json (defaults to <dataset_root>/meta/throw_label.json)")
    src.add_argument("--job", help="LEGACY: path to wurf_N.json (fallback only)")

    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--force", action="store_true", help="Rewrite even if it looks already patched.")
    args = ap.parse_args()

    # Resolve dataset_root / parquet list
    if args.parquet:
        parquets = [Path(args.parquet).expanduser().resolve()]
        if not parquets[0].exists():
            raise SystemExit(f"Parquet not found: {parquets[0]}")
        dataset_root = parquets[0].parents[2]
    else:
        dataset_root = Path(args.dataset_root).expanduser().resolve()
        if not dataset_root.exists():
            raise SystemExit(f"Dataset root not found: {dataset_root}")
        parquets = list(iter_parquets(dataset_root))
        if not parquets:
            raise SystemExit(f"No parquet files found under: {dataset_root/'data'}")

    # Load params from label/job
    label_path = None
    if args.label:
        label_path = Path(args.label).expanduser().resolve()
    elif args.job:
        label_path = None
    else:
        # default: dataset-local label
        label_path = (dataset_root / "meta" / "throw_label.json")

    if label_path is not None:
        if not label_path.exists():
            raise SystemExit(f"throw_label.json not found: {label_path}")
        label = load_throw_label(label_path)
        tx, ty, pos1, pos2, rp = require_label_fields(label, label_path)
        source_str = f"label={label_path}"
    else:
        # legacy job fallback
        job_path = Path(args.job).expanduser().resolve()  # type: ignore[arg-type]
        if not job_path.exists():
            raise SystemExit(f"Job not found: {job_path}")
        job = load_job(job_path)
        tx, ty, pos1, pos2, rp = require_job_fields(job, job_path)
        source_str = f"job={job_path}"

    params13 = np.concatenate([pos1, pos2, np.array([rp], dtype=np.float32)], axis=0).astype(np.float32)

    # Patch each parquet
    last_table = None
    for parquet_path in parquets:
        if not args.no_backup:
            bak = parquet_path.with_suffix(parquet_path.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(parquet_path, bak)
                print(f"[ok] parquet backup: {bak}")

        table = pq.read_table(parquet_path)

        already = (
            ("observation.task" in table.column_names) and
            ("observation.environment_state" in table.column_names) and
            ("action" in table.column_names)
        )
        if already and not args.force:
            # we still recompute env_state to guarantee consistency
            pass

        task_arr = make_task_array(table.num_rows, tx, ty)
        action_arr = make_action_params_array(table.num_rows, params13)

        table = upsert_column(table, "observation.task", task_arr)
        table = upsert_column(table, "action", action_arr)

        env_state_arr = compute_env_state(table)
        table = upsert_column(table, "observation.environment_state", env_state_arr)

        # preserve schema metadata if present
        if table.schema.metadata:
            table = table.cast(table.schema.with_metadata(table.schema.metadata))

        feature_order = default_feature_order(table)
        table = enforce_feature_order(table, feature_order)

        pq.write_table(table, parquet_path)
        print(f"[ok] patched parquet: {parquet_path} (rows={table.num_rows})")

        last_table = table

    assert last_table is not None

    # Patch meta files once (based on the final schema)
    feature_order = list(last_table.column_names)
    patch_info_json(dataset_root, feature_order=feature_order)
    patch_stats_json(dataset_root, last_table, feature_order)

    print("[ok] dataset patched for PARAMETER policy")
    print(f"     dataset_root   : {dataset_root}")
    print(f"     parquets       : {len(parquets)}")
    print(f"     source         : {source_str}")
    print(f"     target_xy_mm   : [{tx}, {ty}]")
    print(f"     action params  : pos1+pos2+rp (13D)")
    print(f"     feature_order  : {feature_order}")


if __name__ == "__main__":
    main()
