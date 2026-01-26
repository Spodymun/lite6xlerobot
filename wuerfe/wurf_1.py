#!/usr/bin/env python3
from wurf import WurfConfig, run_wurf

def main():
    cfg = WurfConfig(
        pos1 = [-1.820000, -1.293970, 2.922131, 0.000000, 0.230383, 0.000000],
        pos2 = [-1.820000, 1.371829, 2.921681, 0.000000, 0.230383, 0.000000],
        release_at = 0.67,
    )
    run_wurf(cfg)

if __name__ == "__main__":
    main()
