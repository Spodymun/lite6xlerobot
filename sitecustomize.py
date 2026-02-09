# sitecustomize.py
# Task-Jitter ±3 mm für LeRobotDataset
# Torch-kompatibel (kein rand_like(generator=...))

import os
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---- Config ----
JITTER_MM = float(os.getenv("LEROBOT_TASK_JITTER_MM", "3"))
MODE = os.getenv("LEROBOT_TASK_JITTER_MODE", "uniform").lower()  # uniform | normal
SEED = os.getenv("LEROBOT_TASK_JITTER_SEED")

if SEED is not None:
    torch.manual_seed(int(SEED))

# ----------------
_orig_getitem = LeRobotDataset.__getitem__

def _noise_like(t: torch.Tensor) -> torch.Tensor:
    if MODE == "normal":
        # Gaussian: sigma = JITTER_MM
        return torch.randn(t.shape, device=t.device, dtype=t.dtype) * JITTER_MM
    # Uniform: [-JITTER_MM, +JITTER_MM]
    return (torch.rand(t.shape, device=t.device, dtype=t.dtype) * 2.0 - 1.0) * JITTER_MM

def _patched_getitem(self, idx):
    x = _orig_getitem(self, idx)

    noise = None

    # observation.task jitter
    if "observation.task" in x:
        t = x["observation.task"].clone()
        noise = _noise_like(t)
        x["observation.task"] = t + noise

    # observation.environment_state: letzte 2 Werte sind target_x/y
    if "observation.environment_state" in x:
        env = x["observation.environment_state"].clone()
        if noise is None:
            noise = _noise_like(env[-2:])
        env[-2:] = env[-2:] + noise
        x["observation.environment_state"] = env

    # task-Key konsistent halten
    if "task" in x and "observation.task" in x:
        x["task"] = x["observation.task"].clone()

    return x

LeRobotDataset.__getitem__ = _patched_getitem
print(f"[sitecustomize] Task-Jitter aktiv: ±{JITTER_MM}mm mode={MODE} seed={SEED}")
