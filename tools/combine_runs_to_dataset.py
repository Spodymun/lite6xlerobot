#!/usr/bin/env python3
"""
Combine many single-episode throw runs (each already patched by tools/patch_actions.py)
into one multi-episode LeRobot dataset.

Assumptions (matches your pipeline):
- Each run dir contains:
    data/chunk-000/file-000.parquet
    meta/episodes/chunk-000/file-000.parquet
    meta/info.json
    meta/stats.json
    meta/throw_label.json  (source of truth for target/action; already patched into parquet)
- Each run has 1 episode (episode_index 0) and some number of frames.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def read_table(path: Path) -> pa.Table:
    return pq.read_table(path)


def write_table(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def to_array(a):
    if isinstance(a, pa.ChunkedArray):
        return pa.concat_arrays(list(a.chunks))
    return a


def upsert_column(table: pa.Table, name: str, col: pa.Array) -> pa.Table:
    if name in table.column_names:
        idx = table.column_names.index(name)
        return table.set_column(idx, name, col)
    return table.append_column(name, col)


def enforce_feature_order(table: pa.Table, feature_order: list[str]) -> pa.Table:
    missing = [c for c in feature_order if c not in table.column_names]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")
    # drop extras (rare)
    extra = [c for c in table.column_names if c not in feature_order]
    if extra:
        table = table.drop(extra)
    return table.select(feature_order)


def _stats_for_matrix(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    N, D = x.shape
    if not np.isfinite(x).all():
        raise RuntimeError("Non-finite values in dataset (NaN/Inf).")
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


def compute_stats(table: pa.Table, feature_order: list[str]) -> OrderedDict:
    stats = OrderedDict()
    for name in feature_order:
        col = to_array(table[name])
        # scalar numeric
        if pa.types.is_integer(col.type) or pa.types.is_floating(col.type):
            x = np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float64)
            stats[name] = _stats_for_matrix(x)
            continue
        # fixed size list (action, env_state, task, state)
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
        # allow strings etc in future, but skip for now
        raise RuntimeError(f"Unsupported column type for stats: {name} -> {col.type}")
    return stats


def patch_info_json(out_root: Path, feature_order: list[str], template_info: dict) -> None:
    info = dict(template_info)  # shallow copy
    feats = info.get("features") or {}
    if not isinstance(feats, dict):
        feats = {}
    # Ensure these are present (your patch_actions.py expects these)
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

    # Reorder exactly like parquet columns
    new_feats = OrderedDict()
    for k in feature_order:
        if k not in feats:
            # keep whatever was in the template (e.g., timestamp, frame_index, etc.)
            # but if truly missing, add a minimal placeholder
            new_feats[k] = feats.get(k, {"dtype": "unknown", "shape": []})
        else:
            new_feats[k] = feats[k]
    info["features"] = new_feats

    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    (out_root / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="records/throws", help="Parent folder containing fs_*/ runs")
    ap.add_argument("--out", required=True, help="Output dataset root (e.g. records/throws/fs_2026-02-06_ALL_THROWS_FIXED)")
    ap.add_argument("--glob", default="fs_*_job_wurf_*", help="Which run folders to include")
    ap.add_argument("--only_success", action="store_true", help="Only include runs where meta/throw_label.json has success=true")
    ap.add_argument("--require_patched", action="store_true", help="Fail if a run parquet has no observation.task/action/env_state")
    args = ap.parse_args()

    repo_root = Path.cwd().resolve()
    runs_root = (repo_root / args.runs_root).resolve()
    out_root = (repo_root / args.out).resolve()

    run_dirs = sorted([p for p in runs_root.glob(args.glob) if p.is_dir()])
    if not run_dirs:
        raise SystemExit(f"No run dirs matched under {runs_root} with glob '{args.glob}'")

    if out_root.exists():
        raise SystemExit(f"Output already exists: {out_root} (delete it first)")

    data_tables = []
    episode_tables = []

    ep_offset = 0
    global_index_offset = 0

    template_info = None
    feature_order = None

    kept = 0
    skipped = 0

    for rd in run_dirs:
        label_path = rd / "meta" / "throw_label.json"
        if not label_path.exists():
            skipped += 1
            continue

        label = json.loads(label_path.read_text(encoding="utf-8"))
        if args.only_success and not bool(label.get("success", False)):
            skipped += 1
            continue

        data_path = rd / "data" / "chunk-000" / "file-000.parquet"
        epi_path  = rd / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        info_path = rd / "meta" / "info.json"

        if not data_path.exists() or not epi_path.exists() or not info_path.exists():
            skipped += 1
            continue

        dt = read_table(data_path)
        et = read_table(epi_path)

        # Basic sanity: ensure patched columns exist
        required_cols = ["action", "observation.task", "observation.environment_state", "observation.state"]
        if args.require_patched and any(c not in dt.column_names for c in required_cols):
            raise SystemExit(f"Run not patched (missing cols) : {rd}")

        # Set template/feature order from first run
        if template_info is None:
            template_info = json.loads(info_path.read_text(encoding="utf-8"))
        if feature_order is None:
            # Use column order from this parquet (your patch_actions enforces stable order)
            feature_order = list(dt.column_names)

        # Reorder/align columns across runs
        dt = enforce_feature_order(dt, feature_order)

        # Re-index episodes
        # data: episode_index, frame_index, index
        n_rows = dt.num_rows

        epi_idx = to_array(dt["episode_index"]).to_numpy(zero_copy_only=False).astype(np.int64)
        # Expect all zeros for single-episode runs
        epi_idx = epi_idx + ep_offset
        dt = upsert_column(dt, "episode_index", pa.array(epi_idx, type=pa.int64()))

        # global index (if exists)
        if "index" in dt.column_names:
            idx = to_array(dt["index"]).to_numpy(zero_copy_only=False).astype(np.int64)
            # some runs start at 0..N-1; make unique globally
            idx = np.arange(global_index_offset, global_index_offset + n_rows, dtype=np.int64)
            dt = upsert_column(dt, "index", pa.array(idx, type=pa.int64()))
        global_index_offset += n_rows

        # frame_index: keep per-episode 0..T-1
        if "frame_index" in dt.column_names:
            fi = to_array(dt["frame_index"]).to_numpy(zero_copy_only=False).astype(np.int64)
            # normalize to start at 0 in case
            fi = fi - fi.min()
            dt = upsert_column(dt, "frame_index", pa.array(fi, type=pa.int64()))

        # task_index: if you want, can group by unique target later; for now keep 0
        if "task_index" in dt.column_names:
            dt = upsert_column(dt, "task_index", pa.array(np.zeros(n_rows, dtype=np.int64), type=pa.int64()))

        # Episodes table: bump episode_index and fix ranges
        ep_pdf = et.to_pandas()

        # bump episode_index
        ep_pdf["episode_index"] = ep_pdf["episode_index"].astype(np.int64) + ep_offset

        # fix task_index
        if "task_index" in ep_pdf.columns:
            ep_pdf["task_index"] = 0

        # fix dataset_from/to and episode_start/end indices using our global "index"
        # We set index contiguous. Derive from dt row span.
        ep_start = global_index_offset - n_rows
        ep_end = global_index_offset - 1

        if "dataset_from_index" in ep_pdf.columns:
            ep_pdf["dataset_from_index"] = ep_start
        if "dataset_to_index" in ep_pdf.columns:
            ep_pdf["dataset_to_index"] = ep_end
        if "episode_start_index" in ep_pdf.columns:
            ep_pdf["episode_start_index"] = ep_start
        if "episode_end_index" in ep_pdf.columns:
            ep_pdf["episode_end_index"] = ep_end

        # length should match rows in this run
        if "length" in ep_pdf.columns:
            ep_pdf["length"] = int(n_rows)

        et = pa.Table.from_pandas(ep_pdf, preserve_index=False)

        data_tables.append(dt)
        episode_tables.append(et)

        ep_offset += int(ep_pdf.shape[0])  # usually 1
        kept += 1

    if kept == 0:
        raise SystemExit("No runs kept (check glob / only_success / presence of meta/throw_label.json)")

    assert feature_order is not None and template_info is not None

    # Concatenate
    data_all = pa.concat_tables(data_tables, promote=True)
    epi_all  = pa.concat_tables(episode_tables, promote=True)

    # Write out dataset
    write_table(out_root / "data" / "chunk-000" / "file-000.parquet", data_all)
    write_table(out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet", epi_all)

    # tasks.parquet: minimal 1-row mapping (task_index=0)
    # (Training uses observation.task directly; task_index is metadata.)
    import pandas as pd
    tasks_df = pd.DataFrame({"task_index": [0]}, index=["throw"])
    tasks_df.to_parquet(out_root / "meta" / "tasks.parquet")

    # Patch info.json (features order must match parquet)
    patch_info_json(out_root, feature_order, template_info)

    # stats.json
    stats = compute_stats(data_all, feature_order)
    (out_root / "meta" / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Also write a tiny combine report
    report = {
        "runs_root": str(runs_root),
        "glob": args.glob,
        "kept": kept,
        "skipped": skipped,
        "total_frames": int(data_all.num_rows),
        "total_episodes": int(ep_offset),
        "feature_order": feature_order,
    }
    (out_root / "meta" / "combine_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[OK] combined dataset written to:", out_root)
    print("     kept runs:", kept, " skipped:", skipped)
    print("     frames:", data_all.num_rows, " episodes:", ep_offset)


if __name__ == "__main__":
    main()
