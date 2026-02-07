#!/usr/bin/env python3
import argparse, json, subprocess, sys
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy

from xarm.wrapper import XArmAPI


def add_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1: return x.unsqueeze(0)   # (D)->(B,D)
    if x.dim() == 2: return x.unsqueeze(0)   # (T,D)->(B,T,D)
    return x


def lerp(a, b, t: float):
    return [(1.0 - t) * av + t * bv for av, bv in zip(a, b)]


def fk_tcp_xyz_mm(robot_ip: str, joints_rad):
    """
    Returns [x,y,z] in mm for given joint angles (rad) using robot FK.
    """
    arm = XArmAPI(robot_ip, is_radian=True)
    arm.connect()

    try:
        # Many SDK builds expose get_forward_kinematics
        # Try a couple of calling conventions.
        try:
            code, pose = arm.get_forward_kinematics(joints_rad, is_radian=True)
        except TypeError:
            code, pose = arm.get_forward_kinematics(joints_rad)

        if code != 0 or pose is None:
            raise RuntimeError(f"FK failed: code={code}, pose={pose}")

        # pose is usually [x,y,z,roll,pitch,yaw] with xyz in mm
        xyz = [float(pose[0]), float(pose[1]), float(pose[2])]
        return xyz

    finally:
        try:
            arm.disconnect()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="records/throws/fs_2026-02-06_ALL_THROWS")
    ap.add_argument("--ckpt", default="outputs/train/2026-02-06/23-17-27_gym_manipulator_act/checkpoints/last/pretrained_model")
    ap.add_argument("--target", nargs=2, type=float, required=True, metavar=("X","Y"))
    ap.add_argument("--seed_index", type=int, default=0, help="Welcher Dataset-Frame als Basis für state/env_state genutzt wird")
    ap.add_argument("--job_out", default="/tmp/wurf_job.json")
    ap.add_argument("--result_out", default="/tmp/throw_result.json")
    ap.add_argument("--robot_ip", default="10.77.77.200")
    ap.add_argument("--dry_run", action="store_true", help="Nur Job schreiben, nicht werfen")
    ap.add_argument("--release_override", type=float, default=None, help="Override predicted release_at (0..1). If set, recompute release_xyz accordingly.")
    ap.add_argument("--release_bias", type=float, default=0.0, help="Add to predicted release_at (clamped)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    root = (repo_root / args.root).resolve()
    ckpt = (repo_root / args.ckpt).resolve()

    ds = LeRobotDataset(repo_id="local/xarm_throws", root=str(root))
    policy = ACTPolicy.from_pretrained(str(ckpt))
    policy.eval()

    x = ds[args.seed_index]
    target_xy = torch.tensor([args.target[0], args.target[1]], dtype=torch.float32)

    state = x["observation.state"].float()

    # env_state bei euch sehr wahrscheinlich: concat(state(7), task(2)) -> 9D
    if "observation.environment_state" in x and x["observation.environment_state"].numel() == 9 and state.numel() == 7:
        env_state = torch.cat([state, target_xy], dim=0)
    else:
        # fallback: nehme env_state aus dataset (falls es etwas anderes ist)
        env_state = x.get("observation.environment_state", None)

    batch = {
        "observation.state": add_batch(state),
        "observation.task": add_batch(target_xy),
    }
    if env_state is not None:
        batch["observation.environment_state"] = add_batch(env_state.float())

    # manche configs haben auch ein "task" key
    batch["task"] = add_batch(target_xy)


    with torch.no_grad():
        out = policy.model(batch)
        actions_hat = out[0] if isinstance(out, tuple) else out
        pred = actions_hat[0,0].cpu() if actions_hat.dim() == 3 else actions_hat[0].cpu()

    if pred.numel() != 13:
        raise RuntimeError(f"Expected 13D action, got {pred.numel()}")

    pos1 = [float(v) for v in pred[0:6].tolist()]
    pos2 = [float(v) for v in pred[6:12].tolist()]
    release_at = float(pred[12].item())
    if args.release_override is not None:
        release_at = float(args.release_override)
    release_at = max(0.05, min(0.95, release_at))
    release_at = max(0.05, min(0.95, release_at + float(args.release_bias)))

    # release joints (linear in joint-space)
    release_joints = lerp(pos1, pos2, release_at)

    # FK -> TCP xyz in mm
    release_xyz = fk_tcp_xyz_mm(args.robot_ip, release_joints)

    job = {
        "pos1": pos1,
        "pos2": pos2,
        "release_at": release_at,
        "release_xyz": release_xyz,
    }

    job_path = Path(args.job_out)
    job_path.write_text(json.dumps(job, indent=2))
    print("[OK] wrote job:", str(job_path))
    print("target:", args.target)
    print("pos1:", pos1)
    print("pos2:", pos2)
    print("release_at:", release_at)
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
