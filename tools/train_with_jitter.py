#!/usr/bin/env python3
import os
import sys
import subprocess

# Default-Jitter: ±3mm uniform
os.environ.setdefault("LEROBOT_TASK_JITTER_MM", "3")
os.environ.setdefault("LEROBOT_TASK_JITTER_MODE", "uniform")
os.environ.setdefault("LEROBOT_TASK_JITTER_SEED", "0")

import sitecustomize  # Patch aktivieren

cmd = ["lerobot-train"] + sys.argv[1:]
print("[RUN]", " ".join(cmd))
subprocess.run(cmd, check=True)