#!/usr/bin/env python3
"""
patch_actions.py

Patches a LeRobot parquet so it becomes trainable for "target_xy -> throw".

It writes:
  1) action[t] = observation.state[t+1]   (default)
  2) observation.task = target_xy_mm      (constant per frame, taken from --job)

Usage:
  python3 patch_actions.py --parquet /path/to/file-000.parquet --job /path/to/wurf_1.json
  python3 patch_actions.py --parquet ... --job ... --mode delta
  python3 patch_actions.py --parquet ... --job ... --task-col observation.task
"""

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def to_array(a):
    """Convert ChunkedArray -> Array, keep Array as-is."""
    if isinstance(a, pa.ChunkedArray):
        return pa.concat_arrays(list(a.chunks))
    return a


def find_state_column(table: pa.Table, prefer: str | None = None):
    """
    Find observation.state column.
    Supports:
      - flat column 'observation.state'
      - struct column 'observation' with field 'state'
    """
    names = set(table.column_names)

    if prefer:
        if prefer in names:
            return prefer, to_array(table[prefer])
        if "." in prefer:
            root, field = prefer.split(".", 1)
            if root in names and pa.types.is_struct(table[root].type):
                return prefer, to_array(table[root].field(field))

    if "observation.state" in names:
        return "observation.state", to_array(table["observation.state"])

    if "observation" in names and pa.types.is_struct(table["observation"].type):
        obs = table["observation"]
        if obs.type.get_field_index("state") != -1:
            return "observation.state", to_array(obs.field("state"))

    raise RuntimeError(
        "Could not find observation.state column. "
        f"Available columns: {table.column_names}"
    )


def ensure_list_float(state: pa.Array) -> pa.Array:
    """Ensure state is a list of floats."""
    t = state.type
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        inner = t.value_type
        if pa.types.is_floating(inner):
            return state
        if pa.types.is_integer(inner):
            return pc.cast(state, pa.list_(pa.float32()))
    raise RuntimeError(f"Unsupported state column type: {t}")


def upsert_column(table: pa.Table, name: str, col: pa.Array) -> pa.Table:
    """Replace column if exists, else append."""
    if name in table.column_names:
        idx = table.column_names.index(name)
        return table.set_column(idx, name, col)
    return table.append_column(name, col)


def make_task_array(n_rows: int, x: float, y: float) -> pa.Array:
    """
    FixedSizeList<float32>[2] per row: [x, y]
    """
    values = pa.array([x, y] * n_rows, type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(values, list_size=2)


# ------------------------------------------------------------
# Action builders
# ------------------------------------------------------------

def make_action_next_state(state: pa.Array) -> pa.Array:
    """
    action[t] = state[t+1]
    last action is repeated
    """
    n = len(state)
    if n < 2:
        raise RuntimeError("Need at least 2 rows to create actions.")

    a0 = state.slice(1, n - 1)      # state[1:]
    last = a0.slice(n - 2, 1)       # repeat last valid action
    return pa.concat_arrays([a0, last])


def make_action_delta(state: pa.Array) -> pa.Array:
    """
    action[t] = state[t+1] - state[t]
    """
    n = len(state)
    if n < 2:
        raise RuntimeError("Need at least 2 rows to create actions.")

    s0 = state.slice(0, n - 1)
    s1 = state.slice(1, n - 1)
    delta = pc.subtract(s1, s0)

    last = delta.slice(n - 2, 1)
    return pa.concat_arrays([delta, last])


# ------------------------------------------------------------
# Job loader
# ------------------------------------------------------------

def load_target_xy(job_path: Path) -> tuple[float, float]:
    with job_path.open("r") as f:
        job = json.load(f)
    if "target_xy_mm" not in job:
        raise RuntimeError(
            f"{job_path} has no 'target_xy_mm'. "
            "Add e.g. \"target_xy_mm\": [360.0, 410.7]"
        )
    x, y = job["target_xy_mm"]
    return float(x), float(y)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True,
                    help="Path to LeRobot data parquet (file-000.parquet)")
    ap.add_argument("--job", required=True,
                    help="Path to your job json (must contain target_xy_mm)")
    ap.add_argument("--mode", choices=["next_state", "delta"],
                    default="next_state",
                    help="Action definition (default: next_state)")
    ap.add_argument("--state-col", default=None,
                    help="Override state column (e.g. observation.state)")
    ap.add_argument("--action-col", default="action",
                    help="Action column name to write (default: action)")
    ap.add_argument("--task-col", default="observation.task",
                    help="Task/target column name (default: observation.task)")
    ap.add_argument("--no-backup", action="store_true",
                    help="Do not create .bak backup")
    args = ap.parse_args()

    p = Path(args.parquet)
    job_path = Path(args.job)

    if not p.exists():
        raise SystemExit(f"File not found: {p}")
    if not job_path.exists():
        raise SystemExit(f"Job not found: {job_path}")

    if not args.no_backup:
        bak = p.with_suffix(p.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(p, bak)
            print(f"[ok] backup written: {bak}")

    # Load target from job
    tx, ty = load_target_xy(job_path)

    # Load parquet
    table = pq.read_table(p)

    # Build action from state
    state_name, state = find_state_column(table, args.state_col)
    state = ensure_list_float(state)

    if args.mode == "next_state":
        action = make_action_next_state(state)
    else:
        action = make_action_delta(state)

    if len(action) != table.num_rows:
        raise RuntimeError("Action length mismatch.")

    # Build task column
    task = make_task_array(table.num_rows, tx, ty)

    # Upsert
    table2 = upsert_column(table, args.action_col, action)
    table2 = upsert_column(table2, args.task_col, task)

    # Preserve metadata
    if table.schema.metadata:
        table2 = table2.cast(table2.schema.with_metadata(table.schema.metadata))

    pq.write_table(table2, p)

    print("[ok] patched dataset")
    print(f"     parquet     : {p}")
    print(f"     job         : {job_path}")
    print(f"     target_xy_mm: [{tx}, {ty}]")
    print(f"     state source: {state_name}")
    print(f"     mode        : {args.mode}")
    print(f"     action col  : {args.action_col}")
    print(f"     task col    : {args.task_col}")
    print(f"     rows        : {table.num_rows}")


if __name__ == "__main__":
    main()
