#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# --------------------------
# helpers
# --------------------------
def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())

def save_json(p: Path, obj: Dict[str, Any]) -> None:
    p.write_text(json.dumps(obj, indent=2, sort_keys=True))

def parse_xy(v) -> Optional[Tuple[float, float]]:
    # Accept [x,y], (x,y), {"x":..,"y":..}, {"xy":[x,y]}
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return float(v[0]), float(v[1])
    if isinstance(v, dict):
        if "xy" in v and isinstance(v["xy"], (list, tuple)) and len(v["xy"]) >= 2:
            return float(v["xy"][0]), float(v["xy"][1])
        if "x" in v and "y" in v:
            return float(v["x"]), float(v["y"])
    return None

def pick_xy_from_wurf(w: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    Try several common keys. You can add more if your wurf_*.json uses different names.
    Priority: corrected/final > target/cup > raw
    """
    candidates = [
        "target_xy_mm",
        "target_xy",
        "cup_xy_mm",
        "cup_mm",
        "cup_xy",
        "landing_xy_mm",
        "landing_xy",
        "xy_mm",
        "xy",
        "target",
        "cup",
    ]
    for k in candidates:
        if k in w:
            xy = parse_xy(w[k])
            if xy is not None:
                return xy

    # fallback: search nested
    for k in ["meta", "label", "labels", "result", "job"]:
        if k in w and isinstance(w[k], dict):
            xy = pick_xy_from_wurf(w[k])
            if xy is not None:
                return xy

    return None

def extract_wurf_id_from_folder(name: str) -> Optional[int]:
    # expects ..._job_wurf_123
    m = re.search(r"_job_wurf_(\d+)$", name)
    if not m:
        return None
    return int(m.group(1))

# --------------------------
# main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Overwrite target_xy_mm in records/throws/* using wuerfe/wurf_*.json")
    ap.add_argument("--wuerfe_dir", type=str, default="wuerfe", help="Directory with wurf_*.json")
    ap.add_argument("--records_dir", type=str, default="records/throws", help="Directory with fs_*_job_wurf_* folders")
    ap.add_argument("--dry_run", action="store_true", help="Do not write, only print what would change")
    ap.add_argument("--backup", action="store_true", help="Write .bak copies before modifying")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N episodes (0=all)")
    args = ap.parse_args()

    wuerfe_dir = Path(args.wuerfe_dir)
    records_dir = Path(args.records_dir)

    if not wuerfe_dir.exists():
        raise SystemExit(f"Missing wuerfe_dir: {wuerfe_dir}")
    if not records_dir.exists():
        raise SystemExit(f"Missing records_dir: {records_dir}")

    # Build map: id -> corrected xy
    wurf_map: Dict[int, Tuple[float, float]] = {}
    for p in sorted(wuerfe_dir.glob("wurf_*.json")):
        m = re.match(r"wurf_(\d+)\.json$", p.name)
        if not m:
            continue
        wid = int(m.group(1))
        w = load_json(p)
        xy = pick_xy_from_wurf(w)
        if xy is None:
            print(f"[WARN] {p} has no usable xy key (skipping)")
            continue
        wurf_map[wid] = xy

    if not wurf_map:
        raise SystemExit("No usable wurf_*.json found (no xy keys detected).")

    # Iterate records folders
    folders = sorted([d for d in records_dir.iterdir() if d.is_dir() and d.name.endswith(tuple(str(i) for i in range(10)))])
    # Better filter explicitly:
    folders = sorted([d for d in records_dir.iterdir() if d.is_dir() and "_job_wurf_" in d.name])

    changed = 0
    skipped = 0
    missing = 0

    for i, ep in enumerate(folders):
        if args.limit and i >= args.limit:
            break

        wid = extract_wurf_id_from_folder(ep.name)
        if wid is None:
            skipped += 1
            continue

        if wid not in wurf_map:
            print(f"[MISS] no source wuerfe/wurf_{wid}.json for {ep.name}")
            missing += 1
            continue

        new_x, new_y = wurf_map[wid]

        label_path = ep / "meta" / "throw_label.json"
        if not label_path.exists():
            print(f"[SKIP] no {label_path}")
            skipped += 1
            continue

        j = load_json(label_path)

        # figure out where the target lives
        old_xy = None
        if "target_xy_mm" in j:
            old_xy = parse_xy(j["target_xy_mm"])
        elif "target_xy" in j:
            old_xy = parse_xy(j["target_xy"])
        else:
            # create it if not present
            old_xy = None

        # keep raw backup in-json
        if old_xy is not None and "target_xy_mm_raw" not in j:
            j["target_xy_mm_raw"] = [old_xy[0], old_xy[1]]

        # write corrected
        j["target_xy_mm"] = [float(new_x), float(new_y)]

        # Optional: also patch task/observation target if your pipeline uses it
        # (leave as-is unless you know you need it)

        if args.dry_run:
            print(f"[DRY] {ep.name}: {old_xy} -> ({new_x:.1f},{new_y:.1f})")
            changed += 1
            continue

        if args.backup:
            bak = label_path.with_suffix(".json.bak")
            if not bak.exists():
                bak.write_text(label_path.read_text())

        save_json(label_path, j)
        print(f"[OK] {ep.name}: set target_xy_mm=({new_x:.1f},{new_y:.1f})")
        changed += 1

    print("\n--- summary ---")
    print("episodes:", len(folders))
    print("changed:", changed)
    print("missing source:", missing)
    print("skipped:", skipped)

if __name__ == "__main__":
    main()
