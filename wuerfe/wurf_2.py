#!/usr/bin/env python3
from wurf import WurfConfig, run_wurf

def main():
    cfg = WurfConfig(
        home=[-1.570796, 0.717330, 1.192060, -0.101229, 0.488692, -1.570796],

        pos1=[-1.640, -0.920, 2.983, 0.042, 0.178, -0.113],
        pos2=[-1.450,  1.300, 2.850, 0.000, 0.100, -0.113],

        release_at=0.30,
    )

    run_wurf(cfg)

if __name__ == "__main__":
    main()
