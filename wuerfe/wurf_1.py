#!/usr/bin/env python3
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from wurf import WurfConfig, run_wurf


def main():
    cfg = WurfConfig(
        # HOME aus deinem bisherigen wurf_1.py
        home=[-1.570796, 0.717330, 1.192060, -0.101229, 0.488692, -1.570796],

        # POS1 / POS2 wie bisher
        pos1=[-1.640, -0.920, 2.983, 0.042, 0.178, -0.113],
        pos2=[-1.517,  1.476, 2.913, -0.002, 0.079, -0.113],

        release_at=0.35,
    )

    run_wurf(cfg)

if __name__ == "__main__":
    main()
