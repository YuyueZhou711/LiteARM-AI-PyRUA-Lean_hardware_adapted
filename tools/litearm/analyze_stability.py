#!/usr/bin/env python3
"""Summarise carry/hover stability samples recorded by validate_block_grasp.py.

For each JSONL: the true block pose, the block pose relative to the right TCP,
the TCP speed/acceleration, the measured opening and (with --contacts) the
finger contact forces.  Hover phases show whether the grasp slips while the arm
is still; carry phases show the speed/acceleration at the first true-z drop.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


def _phase_samples(records, phase):
    return [r for r in records if r.get("event") == "sample" and r.get("phase") == phase]


def _rel_drift_key(a, b, key):
    p0 = np.asarray(a[key]["right"]["pos"], float)
    p1 = np.asarray(b[key]["right"]["pos"], float)
    r0 = np.asarray(a[key]["right"]["rot"], float)
    r1 = np.asarray(b[key]["right"]["rot"], float)
    trace = float(np.trace(r0.T @ r1))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))
    return float(np.linalg.norm(p1 - p0)), angle


def _rel_drift(a, b):
    return _rel_drift_key(a, b, "block_rel")


def _vec_norms(values):
    if not values:
        return None
    arr = np.asarray(values, float)
    return float(np.linalg.norm(arr.reshape(-1, 3), axis=1).max())


def _contact_summary(sample):
    out = {}
    for key in (
        "finger_contact_forces_n",
        "finger_net_contact_forces_n",
        "friction_force_sum_n",
    ):
        value = sample.get(key)
        if value is None:
            continue
        arr = np.asarray(value, float)
        if arr.ndim >= 1 and arr.shape[-1] == 3:
            out[key] = float(np.linalg.norm(arr.reshape(-1, 3), axis=1).max())
        else:
            out[key] = float(np.abs(arr).max())
    contacts = sample.get("contacts") or []
    if contacts:
        points = [p for c in contacts for p in c.get("points", [])]
        out["contacts"] = len(points)
        if points:
            out["min_separation_m"] = min(float(p.get("separation", 0.0)) for p in points)
            out["max_impulse"] = max(
                float(np.linalg.norm(p.get("impulse", [0, 0, 0]))) for p in points
            )
    fc = sample.get("finger_contact_forces_n")
    if fc is not None:
        arr = np.asarray(fc, float)
        if arr.ndim >= 1 and arr.shape[-1] == 3:
            out["per_finger_n"] = [round(float(v), 3) for v in np.linalg.norm(arr.reshape(-1, 3), axis=1)]
    return out


def _summarise(path):
    records = [json.loads(line) for line in Path(path).open()]
    setup = next((r for r in records if r.get("event") == "setup"), {})
    close = next(
        (r["result"] for r in records if r.get("event") == "tool" and r.get("phase") == "close"),
        {},
    )
    verdict = next((r for r in records if r.get("event") == "verdict"), {})
    print(f"\n===== {Path(path).name}")
    print(
        "setup",
        {
            k: setup.get(k)
            for k in (
                "label",
                "yaw",
                "joint_step_rad",
                "carry_joint_step_rad",
                "carry_slip_drop",
                "carry_route",
                "grasp_z",
                "physics_device",
            )
        },
    )
    print("close", {k: close.get(k) for k in ("holding", "opening", "width_m", "steps_used")})
    print("verdict", {k: verdict.get(k) for k in ("success", "mode", "initial", "final", "hover_s")})
    phases = []
    for ph in ("close", "lift", "hover", "carry_0", "carry_1", "carry_2"):
        ps = _phase_samples(records, ph)
        if ph == "hover":
            ps = [r for r in ps if r.get("phase") == "hover"]
        if ps:
            phases.append((ph, ps))
    for ph, ps in phases:
        first, last = ps[0], ps[-1]
        drift_p, drift_r = _rel_drift(first, last)
        zs = [r["block_xyz"][2] for r in ps]
        opens = [r["opening"].get("right_arm") for r in ps if r.get("opening", {}).get("right_arm") is not None]
        speeds = [
            float(np.linalg.norm(r["tcp_vel"]["right"]))
            for r in ps
            if r.get("tcp_vel", {}).get("right")
        ]
        accels = [
            float(np.linalg.norm(r["tcp_acc"]["right"]))
            for r in ps
            if r.get("tcp_acc", {}).get("right")
        ]
        c_first = _contact_summary(first)
        c_last = _contact_summary(last)
        grip_p, grip_r = _rel_drift_key(first, last, "block_rel_grip")
        print(
            f"{ph:8s} n={len(ps):3d} "
            f"z=[{min(zs):.4f},{max(zs):.4f}] "
            f"rel_dpos={drift_p*1000:.2f}mm rel_drot={drift_r:.2f}deg "
            f"grip_dpos={grip_p*1000:.2f}mm grip_drot={grip_r:.2f}deg "
            f"open=[{min(opens):.3f},{max(opens):.3f}] "
            f"vmax={max(speeds, default=0):.3f}m/s amax={max(accels, default=0):.1f}m/s2 "
            f"contacts[{c_first.get('contacts', 0)}] "
            f"fc={c_first.get('finger_contact_forces_n', 0):.2f}->{c_last.get('finger_contact_forces_n', 0):.2f} "
            f"per_finger={c_first.get('per_finger_n')}->{c_last.get('per_finger_n')} "
            f"gap_mid={first.get('finger_gap_m')}/{first.get('finger_mid_m')}"
            f"->{last.get('finger_gap_m')}/{last.get('finger_mid_m')} "
            f"targets={first.get('grip_targets')}"
        )
        # First sample with a true z drop > 5 mm below the phase's first z:
        drop = next((r for r in ps if r["block_xyz"][2] < zs[0] - 0.005), None)
        if drop is not None:
            tcp = drop.get("tcp_xyz", {}).get("right")
            rel = drop.get("block_rel", {}).get("right", {})
            print(
                "  first_z_drop step=",
                drop.get("step"),
                "z=",
                round(drop["block_xyz"][2], 4),
                "tcp=",
                [round(v, 4) for v in tcp] if tcp else None,
                "rel=",
                [round(v, 4) for v in rel.get("pos", [])],
                "open=",
                drop.get("opening", {}).get("right_arm"),
                "speed=",
                round(float(np.linalg.norm(drop.get("tcp_vel", {}).get("right", [0, 0, 0]))), 3),
                "acc=",
                round(float(np.linalg.norm(drop.get("tcp_acc", {}).get("right", [0, 0, 0]))), 1),
            )


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        _summarise(arg)
