"""
lite6_throw_test_safer_v1.py  (vereinfachte Sequenz)

Ziel:
- Release controller-seitig per set_tgpio_digital_with_xyz (Tool DO position trigger, one-shot, radius Pflicht)
- Minimaler Ablauf: home -> wind_up -> release, und bei release bleiben
- Kalibrierung: wind_up -> 30% Zwischenpunkt -> TCP lesen -> zurück wind_up
"""

from __future__ import annotations
import argparse, json, logging, sys, time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---- Lite6 ranges (User Manual / Dev Manual) ----
def deg2rad(x: float) -> float:
    return x * 3.141592653589793 / 180.0

def rad2deg(x: float) -> float:
    return x * 180.0 / 3.141592653589793

LITE6_JOINT_LIMITS_DEG: Dict[int, Tuple[float, float]] = {
    1: (-360.0, 360.0),
    2: (-150.0, 150.0),
    3: (-3.5, 300.0),
    4: (-360.0, 360.0),
    5: (-124.0, 124.0),
    6: (-360.0, 360.0),
}
LITE6_MAX_JOINT_SPEED_DEG_S = 180.0
LITE6_MAX_JOINT_ACCEL_DEG_S2 = 1145.0

# ---- API code handling (UFactory) ----
RETRYABLE_API_CODES = {3, 7, 8}
WAIT_TIMEOUT_CODE = 100
BUSY_API_CODES = {-2, 9}
SOFT_BLOCK_API_CODES = {1, 2}
QUEUE_WARN_CODES = {11, 15}


class ReleaseMethod(str, Enum):
    POSITION = "position"
    TIME = "time"
    IMMEDIATE = "immediate"


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        if hasattr(record, "data") and record.data:
            payload["data"] = record.data
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)

def log_data(logger: logging.Logger, level: int, msg: str, **data: Any) -> None:
    rec = logger.makeRecord(logger.name, level, fn="", lno=0, msg=msg, args=(), exc_info=None)
    rec.data = data if data else None
    logger.handle(rec)

def setup_logger(log_dir: Optional[Path]) -> logging.Logger:
    logger = logging.getLogger("lite6_throw_test")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            log_dir / f"lite6_throw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(JsonLineFormatter())
        logger.addHandler(fh)

    return logger


@dataclass
class MotionConfig:
    # Joint targets (radian)
    home: List[float] = field(default_factory=lambda: [-1.570796, 0.717557, 1.192005, 0.0, 0.474205, -1.570796])      # INIT
    wind_up: List[float] = field(default_factory=lambda: [-1.850000, -1.293970, 2.922131, 0.0, 0.230383, 0.0])          # POS1
    release: List[float] = field(default_factory=lambda: [-1.850000,  1.371829, 2.921681, 0.0, 0.230383, 0.0])          # POS2

    # Behalten (nicht mehr verwendet in der Bewegung, aber lassen wir drin für minimale Änderungen)
    safe_lift: List[float] = field(default_factory=lambda: [-1.570796, 0.717557, 1.192005, 0.0, 0.474205, -1.570796])
    follow_through: List[float] = field(default_factory=lambda: [-1.850000,  1.371829, 2.921681, 0.0, 0.230383, 0.0])

    slow_speed_deg_s: float = 25.0
    throw_speed_deg_s: float = 120.0
    accel_deg_s2: float = 350.0

    collision_sens_throw: int = 4
    collision_sens_restore: int = 3

    # Release
    release_method: ReleaseMethod = ReleaseMethod.POSITION
    release_progress: float = 0.30          # <-- nutzen wir jetzt als Kalibrier-Fraction (30%)
    xyz_tolerance_mm: float = 60.0

    # Tool DO levels for OPEN
    open_to0: int = 1
    open_to1: int = 0

    poll_hz: float = 25.0
    move_timeout_s: float = 10.0
    settle_s: float = 0.2

    api_retries: int = 3
    api_backoff_s: float = 0.08
    auto_clear: bool = False

    enable_reduced_mode: bool = False
    enable_fense_mode: bool = False
    reduced_tcp_boundary: Optional[List[float]] = None

    z_min_mm: Optional[float] = None

    tcp_load_kg: Optional[float] = None
    tcp_cog_mm: Optional[List[float]] = None

    tgpio_reset_when_stop: Optional[bool] = True


@dataclass
class ThrowResult:
    idx: int
    ok: bool
    release_method: str
    api_failures: int
    controller_error: int
    controller_warn: int
    notes: str = ""
    release_xyz: Optional[List[float]] = None
    tcp_poses: Optional[Dict[str, List[float]]] = None
    tgpio_out_after: Optional[Any] = None
    duration_ms: float = 0.0


class Lite6Robot:
    def __init__(self, ip: str, cfg: MotionConfig, logger: logging.Logger, simulator: bool = False):
        self.ip = ip
        self.cfg = cfg
        self.log = logger
        self.simulator = simulator
        self.arm = None
        self.api_failures = 0

    def connect(self) -> None:
        from xarm.wrapper import XArmAPI
        self.arm = XArmAPI(self.ip, is_radian=True, check_joint_limit=False)
        log_data(self.log, logging.INFO, "Connected", ip=self.ip)

        self._call("motion_enable", self.arm.motion_enable, True)
        self._call("set_mode", self.arm.set_mode, 0)
        self._call("set_state", self.arm.set_state, 0)

        if not self.simulator:
            self._try_call("set_self_collision_detection", 1)
        else:
            self._try_call("set_self_collision_detection", 0)

        self._try_call("set_collision_rebound", False)

        if self.cfg.tcp_load_kg is not None and self.cfg.tcp_cog_mm is not None:
            self._call("set_tcp_load", self.arm.set_tcp_load, self.cfg.tcp_load_kg, self.cfg.tcp_cog_mm)
        else:
            log_data(
                self.log, logging.WARNING,
                "TCP load not set. For stability/false collision reduction: set_tcp_load(weight,[x,y,z]).",
            )

        if self.cfg.tgpio_reset_when_stop is not None:
            self._try_call("config_tgpio_reset_when_stop", bool(self.cfg.tgpio_reset_when_stop))
            self._try_call("config_gpio_reset_when_stop", bool(self.cfg.tgpio_reset_when_stop))

        if self.cfg.enable_reduced_mode:
            self._enable_reduced_mode_guardrails()

        self.ensure_ready("post_connect")

    def disconnect(self) -> None:
        if not self.arm:
            return
        try:
            self._try_call("set_collision_sensitivity", self.cfg.collision_sens_restore)
            self.arm.disconnect()
        finally:
            log_data(self.log, logging.INFO, "Disconnected")

    def _try_call(self, method_name: str, *args: Any) -> None:
        if not self.arm:
            return
        fn = getattr(self.arm, method_name, None)
        if not callable(fn):
            log_data(self.log, logging.DEBUG, "API not available", method=method_name)
            return
        try:
            code = int(fn(*args))
            if code != 0:
                log_data(self.log, logging.WARNING, "API call non-zero", method=method_name, api_code=code)
        except Exception as e:
            log_data(self.log, logging.WARNING, "API call exception", method=method_name, err=str(e))

    def _enable_reduced_mode_guardrails(self) -> None:
        boundary = self.cfg.reduced_tcp_boundary
        if boundary is None:
            log_data(self.log, logging.WARNING, "Reduced mode enabled but no boundary provided; skipping boundary config")
        else:
            self._try_call("set_reduced_tcp_boundary", boundary)
        if self.cfg.enable_fense_mode:
            self._try_call("set_fense_mode", True)
            self._try_call("set_fence_mode", True)
        self._try_call("set_reduced_mode", True)

        log_data(self.log, logging.INFO, "Reduced mode guardrails applied",
                 boundary=boundary, fense=self.cfg.enable_fense_mode)

    def _parse_err_warn(self, ew: Any) -> Tuple[int, int]:
        if ew is None:
            return (0, 0)
        if isinstance(ew, (list, tuple)) and len(ew) >= 2:
            try:
                return (int(ew[0]), int(ew[1]))
            except Exception:
                return (0, 0)
        return (0, 0)

    def get_err_warn(self) -> Tuple[int, int]:
        code, ew = self.arm.get_err_warn_code()
        if code != 0:
            log_data(self.log, logging.WARNING, "get_err_warn_code failed", api_code=code)
            return (0, 0)
        return self._parse_err_warn(ew)

    def ensure_ready(self, stage: str) -> None:
        err, warn = self.get_err_warn()
        if err != 0 or warn != 0:
            log_data(self.log, logging.WARNING, "Controller has err/warn", stage=stage, err=err, warn=warn)
            if not self.cfg.auto_clear:
                raise RuntimeError(f"Controller not ready (fail fast). err={err} warn={warn}")

            self._call("clean_warn", self.arm.clean_warn)
            self._call("clean_error", self.arm.clean_error)
            self._call("motion_enable", self.arm.motion_enable, True)
            self._call("set_state", self.arm.set_state, 0)
            time.sleep(0.1)

            err2, warn2 = self.get_err_warn()
            if err2 != 0 or warn2 != 0:
                raise RuntimeError(f"Controller still err/warn after auto_clear: err={err2} warn={warn2}")

    def _call(self, name: str, fn: Callable[..., int], *args: Any, **kwargs: Any) -> int:
        attempt = 0
        backoff = self.cfg.api_backoff_s
        while True:
            attempt += 1
            code = int(fn(*args, **kwargs))
            if code == 0:
                return 0

            self.api_failures += 1
            err, warn = self.get_err_warn()
            log_data(self.log, logging.WARNING, "API call failed", name=name, api_code=code, attempt=attempt, err=err, warn=warn)

            if warn in QUEUE_WARN_CODES:
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 0.8)

            if code in RETRYABLE_API_CODES and attempt <= self.cfg.api_retries:
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 0.8)
                continue

            if code in BUSY_API_CODES and attempt <= self.cfg.api_retries:
                self._try_call("set_state", 0)
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 0.8)
                continue

            if code in SOFT_BLOCK_API_CODES and self.cfg.auto_clear and attempt <= self.cfg.api_retries:
                self._try_call("clean_warn")
                self._try_call("clean_error")
                self._try_call("motion_enable", True)
                self._try_call("set_state", 0)
                time.sleep(backoff)
                continue

            if code == WAIT_TIMEOUT_CODE:
                is_moving = bool(self.arm.get_is_moving())
                if is_moving:
                    log_data(self.log, logging.WARNING, "Wait timeout but still moving; extending wait", name=name)
                    time.sleep(0.5)
                    if not bool(self.arm.get_is_moving()):
                        return 0
                else:
                    return 0

            return code

    def validate_joint_limits(self, q: List[float]) -> None:
        for i in range(6):
            lo, hi = LITE6_JOINT_LIMITS_DEG[i + 1]
            v = rad2deg(q[i])
            if v < lo or v > hi:
                raise ValueError(f"Joint {i+1} out of range: {v:.2f}° not in [{lo},{hi}]")

    def set_collision_sensitivity(self, sens: int) -> None:
        if sens < 0 or sens > 5:
            raise ValueError("collision sensitivity must be 0..5")
        self._call("set_collision_sensitivity", self.arm.set_collision_sensitivity, sens)

    def move_j(self, target: List[float], speed_deg_s: float, accel_deg_s2: float, wait: bool, label: str, radius: Optional[float] = None) -> None:
        self.validate_joint_limits(target)
        if speed_deg_s <= 0 or speed_deg_s > LITE6_MAX_JOINT_SPEED_DEG_S:
            raise ValueError(f"speed_deg_s invalid: {speed_deg_s} (max {LITE6_MAX_JOINT_SPEED_DEG_S})")
        if accel_deg_s2 <= 0 or accel_deg_s2 > LITE6_MAX_JOINT_ACCEL_DEG_S2:
            raise ValueError(f"accel_deg_s2 invalid: {accel_deg_s2} (max {LITE6_MAX_JOINT_ACCEL_DEG_S2})")

        speed_rad = deg2rad(speed_deg_s)
        mvacc_rad = deg2rad(accel_deg_s2)

        kwargs = dict(
            angle=target,
            speed=speed_rad,
            mvacc=mvacc_rad,
            is_radian=True,
            wait=wait,
            timeout=self.cfg.move_timeout_s,
        )
        if radius is not None:
            kwargs["radius"] = radius

        code = self._call(f"set_servo_angle({label})", self.arm.set_servo_angle, **kwargs)
        if code != 0:
            raise RuntimeError(f"Move failed ({label}) api_code={code}")

    def wait_until_stopped(self, timeout_s: float) -> None:
        dt = 1.0 / max(self.cfg.poll_hz, 1.0)
        t0 = time.perf_counter()
        while True:
            if not bool(self.arm.get_is_moving()):
                return
            if (time.perf_counter() - t0) > timeout_s:
                raise TimeoutError("Motion did not stop within timeout")
            time.sleep(dt)

    def get_tcp_pose(self) -> List[float]:
        # Für XYZ ist es egal ob is_radian True/False; wir lassen es "lesbar" (deg)
        code, pose = self.arm.get_position(is_radian=False)
        if code != 0:
            log_data(self.log, logging.WARNING, "get_position failed", api_code=code)
            return [0.0] * 6
        return list(pose[:6])

    def get_tgpio_out(self) -> Any:
        try:
            code, out = self.arm.get_tgpio_output_digital()
            if code != 0:
                return {"api_code": code, "out": None}
            return out
        except Exception as e:
            return {"exc": str(e)}

    def close_gripper(self) -> None:
        self._call("close_lite6_gripper", self.arm.close_lite6_gripper, True)

    def open_gripper_immediate(self) -> None:
        self._call("open_lite6_gripper(sync=False)", self.arm.open_lite6_gripper, False)

    def _interp_joint(self, q1: List[float], q2: List[float], t: float) -> List[float]:
        t = max(0.0, min(1.0, float(t)))
        return [q1[i] + t * (q2[i] - q1[i]) for i in range(6)]

    def calibrate_release_xyz(self) -> Tuple[List[float], Dict[str, List[float]]]:
        """
        NEU (vereinfacht, aber TCP-Logik gleich):
        wind_up (slow) -> Zwischenpunkt bei release_progress (slow, STOP) -> TCP lesen -> back to wind_up.
        """
        tcp_poses: Dict[str, List[float]] = {}

        # optional: erst home, damit reproduzierbar
        self.move_j(self.cfg.home, self.cfg.slow_speed_deg_s, self.cfg.accel_deg_s2, True, "home(cal)")
        tcp_poses["home"] = self.get_tcp_pose()

        # nach POS1
        self.move_j(self.cfg.wind_up, self.cfg.slow_speed_deg_s, self.cfg.accel_deg_s2, True, "wind_up(cal)")
        tcp_poses["wind_up"] = self.get_tcp_pose()

        # 30%-Zwischenpunkt in Joint-Space (POS1->POS2)
        mid_q = self._interp_joint(self.cfg.wind_up, self.cfg.release, self.cfg.release_progress)
        self.move_j(mid_q, self.cfg.slow_speed_deg_s, self.cfg.accel_deg_s2, True, f"mid_{self.cfg.release_progress:.2f}(cal)")
        tcp_poses["mid"] = self.get_tcp_pose()

        xyz = [float(tcp_poses["mid"][0]), float(tcp_poses["mid"][1]), float(tcp_poses["mid"][2])]
        if all(abs(v) < 0.001 for v in xyz):
            raise RuntimeError("Calibration returned zero position; check robot state/connection")

        # zurück nach POS1 als Startpunkt
        self.move_j(self.cfg.wind_up, self.cfg.slow_speed_deg_s, self.cfg.accel_deg_s2, True, "wind_up(back)")
        time.sleep(self.cfg.settle_s)

        return xyz, tcp_poses

    def enforce_z_min(self, tcp_poses: Dict[str, List[float]]) -> None:
        if self.cfg.z_min_mm is None:
            return
        zmin = float(self.cfg.z_min_mm)
        bad = []
        for k, pose in tcp_poses.items():
            z = float(pose[2])
            if z < zmin:
                bad.append((k, z))
        if bad:
            raise RuntimeError(f"Preflight failed: TCP z below z_min_mm={zmin}. Offenders: {bad}")

    def arm_position_trigger_open(self, xyz: List[float]) -> None:
        r = float(self.cfg.xyz_tolerance_mm)
        if r <= 0:
            raise ValueError("xyz_tolerance_mm must be > 0 (otherwise trigger will not work)")

        log_data(self.log, logging.DEBUG, "Arm trigger",
                 note="If STOP clears end IO, triggers/outputs may reset; see Special IO register 146")

        c0 = self._call("set_tgpio_digital_with_xyz(to0)", self.arm.set_tgpio_digital_with_xyz, 0, self.cfg.open_to0, xyz, r)
        c1 = self._call("set_tgpio_digital_with_xyz(to1)", self.arm.set_tgpio_digital_with_xyz, 1, self.cfg.open_to1, xyz, r)
        if c0 != 0 or c1 != 0:
            raise RuntimeError(f"Position trigger setup failed c0={c0} c1={c1}")

    def do_throw(self, idx: int, release_xyz: List[float]) -> ThrowResult:
        t0 = time.perf_counter()
        notes = ""

        self.ensure_ready(f"pre_throw_{idx}")

        # Minimaler Ablauf: home -> wind_up -> release (und dort bleiben)
        self.set_collision_sensitivity(self.cfg.collision_sens_restore)
        self.move_j(self.cfg.home, self.cfg.slow_speed_deg_s, self.cfg.accel_deg_s2, True, "home(pre)")

        self.close_gripper()
        self.move_j(self.cfg.wind_up, self.cfg.slow_speed_deg_s, self.cfg.accel_deg_s2, True, "wind_up")
        time.sleep(self.cfg.settle_s)

        self.set_collision_sensitivity(self.cfg.collision_sens_throw)

        tcp_poses: Dict[str, List[float]] = {"wind_up": self.get_tcp_pose()}

        if self.cfg.release_method == ReleaseMethod.POSITION:
            self.arm_position_trigger_open(release_xyz)

            # EIN Segment: wind_up -> release (non-blocking). Kein follow_through, kein blend.
            self.move_j(self.cfg.release, self.cfg.throw_speed_deg_s, self.cfg.accel_deg_s2, False, "to_release", radius=None)
            self.wait_until_stopped(timeout_s=self.cfg.move_timeout_s)

        elif self.cfg.release_method == ReleaseMethod.TIME:
            self.move_j(self.cfg.release, self.cfg.throw_speed_deg_s, self.cfg.accel_deg_s2, False, "throw_direct")
            delay = max(0.0, min(self.cfg.move_timeout_s * self.cfg.release_progress, self.cfg.move_timeout_s * 0.8))
            time.sleep(delay)
            self.open_gripper_immediate()
            self.wait_until_stopped(timeout_s=self.cfg.move_timeout_s)

        else:  # IMMEDIATE
            self.move_j(self.cfg.release, self.cfg.throw_speed_deg_s, self.cfg.accel_deg_s2, False, "throw_direct")
            self.open_gripper_immediate()
            self.wait_until_stopped(timeout_s=self.cfg.move_timeout_s)

        tcp_poses["after_throw"] = self.get_tcp_pose()

        # Restore sens (bleibt bei POS2; kein Rückweg nach home)
        self.set_collision_sensitivity(self.cfg.collision_sens_restore)

        err, warn = self.get_err_warn()
        if warn in QUEUE_WARN_CODES:
            notes += f"warn_queue={warn} "
        if err != 0:
            notes += f"err={err} "

        dur_ms = (time.perf_counter() - t0) * 1000.0
        ok = (err == 0)

        return ThrowResult(
            idx=idx,
            ok=ok,
            release_method=self.cfg.release_method.value,
            api_failures=self.api_failures,
            controller_error=err,
            controller_warn=warn,
            notes=notes.strip(),
            release_xyz=release_xyz,
            tcp_poses=tcp_poses,
            tgpio_out_after=self.get_tgpio_out(),
            duration_ms=dur_ms,
        )


def parse_xyz(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected 'x,y,z'")
    return [float(parts[0]), float(parts[1]), float(parts[2])]

def parse_boundary6(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 6:
        raise ValueError("Expected 6 values: x_max,x_min,y_max,y_min,z_max,z_min")
    return [float(x) for x in parts]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--num-throws", type=int, default=3)
    ap.add_argument("--log-dir", type=str, default=None)

    ap.add_argument("--release-method", choices=[m.value for m in ReleaseMethod], default="position")
    ap.add_argument("--release-progress", type=float, default=0.30)  # 30% Kalibrierpunkt
    ap.add_argument("--xyz-tolerance-mm", type=float, default=60.0)

    ap.add_argument("--throw-speed", type=float, default=120.0)
    ap.add_argument("--accel", type=float, default=350.0)

    ap.add_argument("--collision-sens-throw", type=int, default=4)
    ap.add_argument("--collision-sens-restore", type=int, default=3)

    ap.add_argument("--tcp-load-kg", type=float, default=None)
    ap.add_argument("--tcp-cog-mm", type=str, default=None, help="x,y,z in mm")

    ap.add_argument("--enable-reduced-mode", action="store_true")
    ap.add_argument("--enable-fense-mode", action="store_true")
    ap.add_argument("--reduced-boundary", type=str, default=None, help="x_max,x_min,y_max,y_min,z_max,z_min")
    ap.add_argument("--z-min-mm", type=float, default=None, help="Abort if any key pose TCP z < this")

    ap.add_argument("--auto-clear", action="store_true")
    ap.add_argument("--save-results", type=str, default=None)
    ap.add_argument("--simulator", action="store_true", help="Enable simulator mode (disables self-collision detection)")
    args = ap.parse_args()

    logger = setup_logger(Path(args.log_dir) if args.log_dir else None)

    cfg = MotionConfig(
        release_method=ReleaseMethod(args.release_method),
        release_progress=float(args.release_progress),
        xyz_tolerance_mm=float(args.xyz_tolerance_mm),
        throw_speed_deg_s=float(args.throw_speed),
        accel_deg_s2=float(args.accel),
        collision_sens_throw=int(args.collision_sens_throw),
        collision_sens_restore=int(args.collision_sens_restore),
        auto_clear=bool(args.auto_clear),
        enable_reduced_mode=bool(args.enable_reduced_mode),
        enable_fense_mode=bool(args.enable_fense_mode),
        reduced_tcp_boundary=parse_boundary6(args.reduced_boundary) if args.reduced_boundary else None,
        z_min_mm=float(args.z_min_mm) if args.z_min_mm is not None else None,
        tcp_load_kg=float(args.tcp_load_kg) if args.tcp_load_kg is not None else None,
        tcp_cog_mm=parse_xyz(args.tcp_cog_mm) if args.tcp_cog_mm else None,
    )

    log_data(logger, logging.INFO, "Session start", cfg=asdict(cfg))

    robot = Lite6Robot(ip=args.ip, cfg=cfg, logger=logger, simulator=args.simulator)
    results: List[ThrowResult] = []

    try:
        robot.connect()

        # Preflight: kalibriere Release-XYZ (30% Punkt zwischen wind_up und release)
        release_xyz, tcp_poses = robot.calibrate_release_xyz()
        log_data(logger, logging.INFO, "Calibrated TCP poses", poses=tcp_poses, release_xyz=release_xyz)

        robot.enforce_z_min(tcp_poses)

        for i in range(int(args.num_throws)):
            r = robot.do_throw(i, release_xyz)
            results.append(r)
            log_data(logger, logging.INFO, "Throw done", result=asdict(r))

            if not r.ok:
                log_data(logger, logging.ERROR, "Throw failed; consider: verify mounting direction + tcp load; reduce speed; review collision sensitivity")
                break

            time.sleep(0.4)

    finally:
        robot.disconnect()

    ok_count = sum(1 for r in results if r.ok)
    log_data(logger, logging.INFO, "Summary", total=len(results), ok=ok_count, fail=len(results)-ok_count, api_failures=robot.api_failures)

    if args.save_results:
        out = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "cfg": asdict(cfg),
            "summary": {"total": len(results), "ok": ok_count, "fail": len(results)-ok_count, "api_failures": robot.api_failures},
            "results": [asdict(r) for r in results],
        }
        Path(args.save_results).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_results, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        log_data(logger, logging.INFO, "Wrote results", path=args.save_results)

    return 0 if ok_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
