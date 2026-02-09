#!/usr/bin/env python3
import argparse, json, subprocess, sys, os
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.policies import PreTrainedConfig

from xarm.wrapper import XArmAPI


def add_batch(x: torch.Tensor) -> torch.Tensor:
    """Ensure batch dimension exists."""
    if x.dim() == 1:
        return x.unsqueeze(0)   # (D)->(B,D)
    if x.dim() == 2:
        return x.unsqueeze(0)   # (T,D)->(B,T,D)
    return x


def lerp(a, b, t: float):
    """Linear interpolation between two same-length lists."""
    return [(1.0 - t) * av + t * bv for av, bv in zip(a, b)]


def fk_tcp_xyz_mm(robot_ip: str, joints_rad):
    """
    Returns [x,y,z] in mm for given joint angles (rad) using robot FK.
    Requires robot connection.
    """
    arm = XArmAPI(robot_ip, is_radian=True)
    arm.connect()
    try:
        try:
            code, pose = arm.get_forward_kinematics(joints_rad, is_radian=True)
        except TypeError:
            code, pose = arm.get_forward_kinematics(joints_rad)

        if code != 0 or pose is None:
            raise RuntimeError(f"FK failed: code={code}, pose={pose}")

        xyz = [float(pose[0]), float(pose[1]), float(pose[2])]
        return xyz
    finally:
        try:
            arm.disconnect()
        except Exception:
            pass


def resolve_repo_root_and_paths(args):
    repo_root = Path(__file__).resolve().parents[1]
    root = (repo_root / args.root).resolve()

    ckpt_in = (repo_root / args.ckpt).resolve() if not Path(args.ckpt).is_absolute() else Path(args.ckpt).resolve()

    # Allow passing:
    #  - .../checkpoints/010000
    #  - .../checkpoints/010000/pretrained_model
    #  - .../checkpoints/last
    #  - .../checkpoints/last/pretrained_model
    if (ckpt_in / "pretrained_model").is_dir():
        ckpt_dir = ckpt_in / "pretrained_model"
    else:
        ckpt_dir = ckpt_in

    return repo_root, root, ckpt_dir


def load_policy_local(ckpt_dir: Path):
    """
    Load ACTPolicy from a local directory that contains config.json and model.safetensors.
    Avoids HF repo-id validation by providing config explicitly.
    """
    cfg_path = ckpt_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in: {ckpt_dir}")

    # Read config locally (no HF)
    cfg = PreTrainedConfig.from_pretrained(str(ckpt_dir))

    # Provide config explicitly to skip hf_hub_download path in policy loader
    policy = ACTPolicy.from_pretrained(str(ckpt_dir), config=cfg)
    policy.eval()
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="records/throws/fs_2026-02-08_ALL_THROWS_FIXED")
    ap.add_argument("--ckpt", required=True,
                    help="Path to checkpoint dir or pretrained_model dir. "
                         "Examples: outputs/train/.../checkpoints/010000 or .../010000/pretrained_model")
    ap.add_argument("--target", nargs=2, type=float, required=True, metavar=("X", "Y"))
    ap.add_argument("--seed_index", type=int, default=0, help="Dataset frame as base for state/env_state")
    ap.add_argument("--job_out", default="/tmp/wurf_job.json")
    ap.add_argument("--result_out", default="/tmp/throw_result.json")
    ap.add_argument("--robot_ip", default="10.77.77.200")
    ap.add_argument("--dry_run", action="store_true", help="Nur Job schreiben, nicht werfen")

    # New flags:
    ap.add_argument("--fk", action="store_true",
                    help="Berechne FK (requires robot connection). "
                         "Default: OFF in dry_run; ON only if you set --fk.")
    ap.add_argument("--no_fk", action="store_true",
                    help="Nie FK berechnen (auch wenn --fk gesetzt wäre).")

    ap.add_argument("--release_override", type=float, default=None,
                    help="Override predicted release_at (0..1). If set, recompute release_joints accordingly.")
    ap.add_argument("--release_bias", type=float, default=0.0, help="Add to predicted release_at (clamped)")
    args = ap.parse_args()

    # Be strict about HF usage in offline runs (optional but helpful)
    # If you want HF network access, unset these.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    repo_root, root, ckpt_dir = resolve_repo_root_and_paths(args)

    # Dataset (repo_id string is only an identifier here; root is what matters locally)
    ds = LeRobotDataset(repo_id="local/xarm_throws_fixed", root=str(root))

    # Load policy from local dir
    policy = load_policy_local(ckpt_dir)

    x = ds[args.seed_index]
    target_xy = torch.tensor([args.target[0], args.target[1]], dtype=torch.float32)

    state = x["observation.state"].float()

    # env_state: concat(state(7), task(2)) -> 9D
    if "observation.environment_state" in x and x["observation.environment_state"].numel() == 9 and state.numel() == 7:
        env_state = torch.cat([state, target_xy], dim=0)
    else:
        env_state = x.get("observation.environment_state", None)

    batch = {
        "observation.state": add_batch(state),
        "observation.task": add_batch(target_xy),
        "task": add_batch(target_xy),  # some configs use "task"
    }
    if env_state is not None:
        batch["observation.environment_state"] = add_batch(env_state.float())

    with torch.no_grad():
        out = policy.model(batch)
        actions_hat = out[0] if isinstance(out, tuple) else out
        pred = actions_hat[0, 0].cpu() if actions_hat.dim() == 3 else actions_hat[0].cpu()

    if pred.numel() != 13:
        raise RuntimeError(f"Expected 13D action, got {pred.numel()}")

    pos1 = [float(v) for v in pred[0:6].tolist()]
    pos2 = [float(v) for v in pred[6:12].tolist()]
    release_at = float(pred[12].item())

    if args.release_override is not None:
        release_at = float(args.release_override)

    release_at = max(0.05, min(0.95, release_at + float(args.release_bias)))

    # release joints (linear in joint-space)
    release_joints = lerp(pos1, pos2, release_at)

    # FK decision:
    do_fk = args.fk and (not args.no_fk) and (not args.dry_run)
    # If you really want FK even in dry_run, you can allow it by removing "(not args.dry_run)" above.
    # But default behavior keeps dry_run robot-free.

    release_xyz = None
    if do_fk:
        release_xyz = fk_tcp_xyz_mm(args.robot_ip, release_joints)

    job = {
        "pos1": pos1,
        "pos2": pos2,
        "release_at": release_at,
        "release_joints": release_joints,
        "release_xyz": release_xyz,
        "target_xy_mm": [float(args.target[0]), float(args.target[1])],
    }

    job_path = Path(args.job_out)
    job_path.write_text(json.dumps(job, indent=2))

    print("[OK] wrote job:", str(job_path))
    print("ckpt_dir:", str(ckpt_dir))
    print("target:", args.target)
    print("pos1:", pos1)
    print("pos2:", pos2)
    print("release_at:", release_at)
    print("release_joints:", release_joints)
    if release_xyz is None:
        print("release_xyz(mm): <skipped>")
    else:
        print("release_xyz(mm):", release_xyz)

    if args.dry_run:
        return

    throw_script = repo_root / "wuerfe" / "throw_from_job.py"
    cmd = [sys.executable, str(throw_script), "--job", str(job_path), "--result", args.result_out]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[OK] result:", args.result_out)


if __name__ == "__main__":
    main()
