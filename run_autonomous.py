#!/usr/bin/env python3
"""
run_autonomous.py
-----------------
Autonomer Ablauf:

1) Ball finden (BALL_MM)
2) Ball picken
3) Cup finden (CUP_MM) -> Commit
4) Werfen
5) Danach warten, bis der Ball wieder ruhig liegt:
   - y_mm > ball_return_y
   - gleiche Koordinaten (± eps mm) N-mal hintereinander

Erst dann beginnt der nächste Zyklus.
"""

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Any


# =========================
# Dataclasses
# =========================
@dataclass
class BallDet:
    x_mm: float
    y_mm: float
    conf: float


@dataclass
class CupDet:
    x_mm: float
    y_mm: float
    score: float
    yolo_conf: float


# =========================
# Parsing
# =========================
def parse_ball_line(line: str) -> Optional[BallDet]:
    line = line.strip()
    if not line.startswith("BALL_MM "):
        return None
    p = line.split()
    if len(p) < 4:
        return None
    try:
        return BallDet(float(p[1]), float(p[2]), float(p[3]))
    except ValueError:
        return None


def parse_cup_line(line: str) -> Optional[CupDet]:
    line = line.strip()
    if not line.startswith("CUP_MM "):
        return None
    p = line.split()
    if len(p) < 5:
        return None
    try:
        return CupDet(float(p[1]), float(p[2]), float(p[3]), float(p[4]))
    except ValueError:
        return None


# =========================
# Utils
# =========================
def mm_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# =========================
# Subprocess helper
# =========================
def run_and_capture_first_match(
    cmd: List[str],
    *,
    timeout_s: float,
    parse_fn: Callable[[str], Optional[Any]],
    verbose: bool = False,
    cwd: Optional[str] = None,
) -> Tuple[Optional[Any], int]:
    start = time.time()

    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        assert p.stdout is not None
        for line in p.stdout:
            if verbose:
                print(line, end="")
            obj = parse_fn(line)
            if obj is not None:
                try:
                    p.terminate()
                except Exception:
                    pass
                return obj, p.returncode or 0

            if time.time() - start > timeout_s:
                try:
                    p.kill()
                except Exception:
                    pass
                return None, 124

        return None, p.wait(timeout=1.0)

    except Exception:
        try:
            p.kill()
        except Exception:
            pass
        raise


# =========================
# Robot hooks (PLATZHALTER)
# =========================
def pick_ball(x_mm: float, y_mm: float) -> bool:
    print(f"[ROBOT] PICK ball at ({x_mm:.1f}, {y_mm:.1f})")
    return True


def throw_to(x_mm: float, y_mm: float) -> bool:
    print(f"[ROBOT] THROW to cup at ({x_mm:.1f}, {y_mm:.1f})")
    return True


# =========================
# Logging
# =========================
def ensure_csv(path: Path):
    if path.exists():
        return
    with path.open("w", newline="") as f:
        csv.writer(f).writerow([
            "timestamp",
            "ball_x", "ball_y", "ball_conf",
            "cup_x", "cup_y", "cup_score", "cup_yolo",
        ])


def log_row(path: Path, row: List):
    with path.open("a", newline="") as f:
        csv.writer(f).writerow(row)


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--H", type=str, default="H.npy")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-throws", type=int, default=0)
    ap.add_argument("--log", type=str, default="autonomy_log.csv")
    ap.add_argument("--verbose-detectors", action="store_true")

    # Ball return logic
    ap.add_argument("--ball-return-y", type=float, default=500.0)
    ap.add_argument("--ball-return-min-conf", type=float, default=0.4)
    ap.add_argument("--ball-stable-n", type=int, default=5)
    ap.add_argument("--ball-stable-eps-mm", type=float, default=8.0)
    ap.add_argument("--ball-return-poll", type=float, default=0.6)
    ap.add_argument("--ball-return-timeout", type=float, default=120.0)

    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    workdir = base / "koordinaten"

    ball_script = workdir / "ball_finder.py"
    cup_script = workdir / "cups_yolo.py"
    H_path = workdir / args.H

    if not ball_script.exists():
        raise RuntimeError("ball_finder.py not found")
    if not cup_script.exists():
        raise RuntimeError("cups_yolo.py not found")
    if not H_path.exists():
        raise RuntimeError("H.npy not found")

    log_path = Path(args.log)
    ensure_csv(log_path)

    ball_cmd = [
        sys.executable, str(ball_script),
        "--cam", str(args.cam),
        "--H", str(H_path),
        "--device", args.device,
        "--once",
    ]

    cup_cmd = [
        sys.executable, str(cup_script),
        "--cam", str(args.cam),
        "--H", str(H_path),
        "--device", args.device,
        "--once",
        "--show_red",
        "--yolo_conf", "0.10",
        "--roi_pad", "0.05",
        "--ann_in", "1",
        "--ann_out", "6",
        "--red_ring_min", "0.06",
        "--black_min", "0.10",
        "--black_dark_thresh", "125",
        "--inner_r_frac", "0.42",
        "--white_inner_min", "0.22",
        "--white_v_min", "135",
        "--red_inner_max", "0.16",
        "--h_param2", "8",
        "--h_minDist", "12",
    ]

    print("[RUN] Autonomous mode started")

    throws = 0
    while True:
        if args.max_throws > 0 and throws >= args.max_throws:
            print("[RUN] Done")
            break

        # ---- BALL ----
        ball, rc = run_and_capture_first_match(
            ball_cmd,
            timeout_s=15.0,
            parse_fn=parse_ball_line,
            verbose=args.verbose_detectors,
            cwd=str(workdir),
        )
        if ball is None:
            print("[WARN] Ball not found")
            continue

        print(f"[OK] BALL ({ball.x_mm:.1f},{ball.y_mm:.1f})")

        if not pick_ball(ball.x_mm, ball.y_mm):
            continue

        # ---- CUP ----
        cup, rc = run_and_capture_first_match(
            cup_cmd,
            timeout_s=15.0,
            parse_fn=parse_cup_line,
            verbose=args.verbose_detectors,
            cwd=str(workdir),
        )
        if cup is None:
            print("[WARN] Cup not found")
            continue

        print(f"[OK] CUP ({cup.x_mm:.1f},{cup.y_mm:.1f})")

        throw_to(cup.x_mm, cup.y_mm)

        log_row(log_path, [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            ball.x_mm, ball.y_mm, ball.conf,
            cup.x_mm, cup.y_mm, cup.score, cup.yolo_conf,
        ])

        throws += 1

        # ---- WAIT FOR BALL RETURN (STABLE) ----
        print("[WAIT] Waiting for stable ball return...")
        t0 = time.time()
        stable = 0
        last_xy = None

        while True:
            if time.time() - t0 > args.ball_return_timeout:
                print("[WAIT] Timeout, continue anyway")
                break

            bd, rc = run_and_capture_first_match(
                ball_cmd,
                timeout_s=10.0,
                parse_fn=parse_ball_line,
                verbose=False,
                cwd=str(workdir),
            )

            if bd is None:
                time.sleep(args.ball_return_poll)
                continue

            if bd.conf < args.ball_return_min_conf or bd.y_mm <= args.ball_return_y:
                stable = 0
                last_xy = None
                time.sleep(args.ball_return_poll)
                continue

            cur = (bd.x_mm, bd.y_mm)

            if last_xy is None or mm_dist(cur, last_xy) > args.ball_stable_eps_mm:
                stable = 1
                last_xy = cur
            else:
                stable += 1

            print(f"[WAIT] stable {stable}/{args.ball_stable_n} at ({bd.x_mm:.1f},{bd.y_mm:.1f})")

            if stable >= args.ball_stable_n:
                print("[WAIT] Ball is stable")
                break

            time.sleep(args.ball_return_poll)


if __name__ == "__main__":
    main()