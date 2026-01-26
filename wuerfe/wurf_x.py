#!/usr/bin/env python3
from wurf import WurfConfig, run_wurf

def main():
    cfg = WurfConfig(
        pos1=[-1.640, -0.920, 2.983, 0.042, 0.178, -0.113],
        pos2=[-1.517,  1.476, 2.913, -0.002, 0.079, -0.113],
        release_at=0.35,
    )

    run_wurf(cfg)

if __name__ == "__main__":
    main()
