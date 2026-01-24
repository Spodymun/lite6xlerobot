#!/usr/bin/env python3
from __future__ import annotations
import time
import threading
from dataclasses import dataclass
from typing import List

import numpy as np
from xarm.wrapper import XArmAPI


# ------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------

@dataclass
class WurfConfig:
    # Robot
    robot_ip: str = "10.77.77.200"

    # Posen (rad)
    home: List[float] = None
    pos1: List[float] = None
    pos2: List[float] = None

    # Geschwindigkeiten / Beschleunigungen
    home_speed: float = 0.5
    home_acc: float = 1.0
    pos1_speed: float = 1.0
    pos1_acc: float = 1.0
    throw_speed: float = 3.0
    throw_acc: float = 6.0

    # Release-Logik
    release_at: float = 0.35          # Fortschritt 0..1
    progress_timeout: float = 2.0     # s
    poll_dt: float = 0.005             # s

    # Greifer
    close_gripper_at_start: bool = False  # <- WICHTIG: Ball ist schon gegriffen


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def joint_progress(cur, start, target) -> float:
    cur = np.array(cur[:6], dtype=float)
    start = np.array(start[:6], dtype=float)
    target = np.array(target[:6], dtype=float)

    total = np.linalg.norm(target - start)
    if total < 1e-6:
        return 1.0

    done = np.linalg.norm(cur - start)
    return float(np.clip(done / total, 0.0, 1.0))


def init_arm(ip: str) -> XArmAPI:
    arm = XArmAPI(ip, is_radian=True)
    arm.connect()
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.3)
    return arm


# ------------------------------------------------------------
# Hauptlogik
# ------------------------------------------------------------

def run_wurf(cfg: WurfConfig):
    assert cfg.home and cfg.pos1 and cfg.pos2, \
        "home / pos1 / pos2 müssen gesetzt sein"

    arm = init_arm(cfg.robot_ip)

    try:
        # ----------------------------------------------------
        # HOME
        # ----------------------------------------------------
        arm.set_servo_angle(
            angle=cfg.home,
            speed=cfg.home_speed,
            mvacc=cfg.home_acc,
            is_radian=True,
            wait=True,
        )

        if cfg.close_gripper_at_start:
            arm.close_lite6_gripper(sync=True)
            time.sleep(0.2)

        # ----------------------------------------------------
        # POS1 (Init)
        # ----------------------------------------------------
        arm.set_servo_angle(
            angle=cfg.pos1,
            speed=cfg.pos1_speed,
            mvacc=cfg.pos1_acc,
            is_radian=True,
            wait=True,
        )

        if cfg.close_gripper_at_start:
            arm.close_lite6_gripper(sync=True)
            time.sleep(0.2)

        # ----------------------------------------------------
        # THROW: POS1 -> POS2
        # ----------------------------------------------------
        arm.set_servo_angle(
            angle=cfg.pos2,
            speed=cfg.throw_speed,
            mvacc=cfg.throw_acc,
            is_radian=True,
            wait=False,
        )

        # >>> KRITISCH: Start-Barriere <<<
        time.sleep(0.05)

        # Startwinkel NACH Bewegungsbeginn erfassen
        code, start_angles = arm.get_servo_angle(is_radian=True)
        if code != 0:
            start_angles = cfg.pos1
        start_angles = start_angles[:6]

        t0 = time.time()

        def release_thread():
            opened = False
            while (time.time() - t0) < cfg.progress_timeout and not opened:
                c, cur = arm.get_servo_angle(is_radian=True)
                if c == 0:
                    p = joint_progress(cur, start_angles, cfg.pos2)
                    if p >= cfg.release_at:
                        arm.open_lite6_gripper(sync=False)
                        opened = True
                        break
                time.sleep(cfg.poll_dt)

            if not opened:
                arm.open_lite6_gripper(sync=False)

        threading.Thread(target=release_thread, daemon=True).start()

        # ausreichend Zeit für den Wurf
        time.sleep(2.0)

        # ----------------------------------------------------
        # ENDE (kein Rückfahren!)
        # ----------------------------------------------------

    finally:
        arm.disconnect()
