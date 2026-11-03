#!/usr/bin/env python3
"""Local diagnostic: real EvalEnv physics + production LiteArm tools, no policy server.

Run in RoboDojo conda environment from any directory:
  python tools/litearm/validate_block_grasp.py --headless --enable_cameras --trials 1
Ground truth is used only here, never supplied to a formal policy evaluation.
"""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import sys
import subprocess
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT / "robodojo")
for path in reversed((ROOT / "robodojo", ROOT / "robodojo/XPolicyLab", ROOT / "pyrua-lean/src")):
    sys.path.insert(0, str(path))

# Both repositories expose a namespace package named utils. RoboDojo must win,
# as in scripts/eval_policy.sh; XPolicyLab's load_file lacks object metadata.
from utils.load_file import load_object_metadata

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trials", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--layout", type=int, default=0)
parser.add_argument("--label", default="block_2", help="Yellow block in seed-0 layout")
parser.add_argument("--grasp-z", type=float, default=0.815)
parser.add_argument("--lift-z", type=float, default=0.90)
parser.add_argument("--yaw", type=float, default=0.0)
parser.add_argument("--descend-path", choices=("auto", "line", "joint"), default="line")
parser.add_argument("--carry-route", choices=("none", "second-yellow", "second-white"), default="none",
                    help="After the lift, reproduce the second full-run carry route and stop.")
parser.add_argument("--carry-slip-drop", type=float, default=1.0,
                    help="Carry guard opening drop; 1.0 disables it for observation.")
parser.add_argument("--carry-joint-step", type=float, default=0.025,
                    help="Joint step used only while carrying (finer = slower).")
parser.add_argument("--carry-path", choices=("line", "auto", "joint"), default="line",
                    help="Path type for the carry route (auto swings where line is out of reach).")
parser.add_argument("--carry-smooth", action="store_true",
                    help="Ease the first and last joint steps of each carry move.")
parser.add_argument("--hover-s", type=float, default=0.0,
                    help="After the lift, hold still this long (seconds) and log the drift.")
parser.add_argument("--mimic", action="store_true",
                    help="Apply a temporary q_left = q_right PhysX mimic to the two slider joints.")
parser.add_argument("--place-on", action="store_true",
                    help="After the lift, put the block on --place-support-label with place_on.")
parser.add_argument("--home-after-place", action="store_true",
                    help="After place_on, log the gripper keep values and call home.")
parser.add_argument("--place-support-label", default="block_0")
parser.add_argument("--support-top-offset", type=float, default=0.0146,
                    help="From the support block's root z to its top surface.")
parser.add_argument("--tcp-to-bottom", type=float, default=0.050,
                    help="Grasp point to the held block's bottom for this grasp.")
parser.add_argument("--joint-step", type=float, default=0.025,
                    help="Diagnostic maximum joint waypoint spacing in radians")
parser.add_argument("--rigid-pad", action="store_true", help="In-memory control without compliant contact")
parser.add_argument("--velocity-iterations", type=int, default=None, help="In-memory solver control")
parser.add_argument("--contacts", action="store_true", help="Record PhysX finger contact and friction reports")
parser.add_argument("--force-gripper", action="store_true", help="In-memory control using force slider drives")
parser.add_argument("--output", default=str(ROOT / "tools/litearm/grasp_validation.jsonl"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.trials < 1:
    parser.error("--trials must be at least 1")
if not 0 < args.joint_step <= 0.15:
    parser.error("--joint-step must be in (0, 0.15]")
if args.trials > 1:
    # RoboDojo's scene cache can retain stale actor names across resets. Each
    # repeat gets a fresh Kit process, so every trial starts from the same state.
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    exit_code = 0
    with output.open("w") as combined:
        for trial in range(args.trials):
            child_output = output.with_name(f"{output.stem}.trial_{trial}{output.suffix}")
            print(f"GRASP_DIAG starting independent trial {trial + 1}/{args.trials}", flush=True)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                                     "--trials", "1", "--output", str(child_output)])
            if child_output.exists():
                for line in child_output.read_text().splitlines():
                    record = json.loads(line)
                    record["trial"] = trial
                    combined.write(json.dumps(record) + "\n")
                combined.flush()
            exit_code = max(exit_code, result.returncode if result.returncode >= 0 else 2)
            if result.returncode not in (0, 1):
                break
    sys.exit(exit_code)
launcher = AppLauncher(args)

import numpy as np
from omegaconf import OmegaConf
from env.global_configs import ENV_CONFIG_PATH, ROOT_DIR, BENCHMARK
from task.RoboDojo import task_registry
from utils.load_file import load_yaml
from utils.pipeline_utils import process_config, process_randomization
from src.eval_client import eval_env
from pyrualean.hosts import litearm_tools as tools
from collision_normalization import validate_collision_shapes
from drive_normalization import normalize_gripper_drives
from isaacsim.core.utils.stage import get_current_stage
from pxr import Usd, UsdPhysics, UsdShade, PhysicsSchemaTools, PhysxSchema


def find_joint_prim(stage, robot_path, name):
    """Joint prims are not direct children of the robot root (URDF importer nests
    them), so fall back to a name search under the robot."""
    prim = stage.GetPrimAtPath(f"{robot_path}/{name}")
    if prim:
        return prim
    root = stage.GetPrimAtPath(robot_path)
    if not root:
        return None
    for p in Usd.PrimRange(root):
        if p.GetName() == name:
            return p
    return None


def apply_mimic(stage, robot_path):
    """Make the left slider follow the right one: q_left = 1.0 * q_right + 0.0.

    The URDF has no ``<mimic>``; its two slide axes point opposite ways, so equal
    joint positions already mirror the fingers.  The left joint is the mimic, the
    right is the reference/driven joint; the left's own linear drive is zeroed and
    it is left out of the gripper actuator (see main), so only the right one drives.

    PhysX uses the additive form ``q_target + gearing * q_ref + offset = 0``,
    unlike URDF's ``q_target = multiplier * q_ref + offset``.  The URDF importer
    maps ``multiplier=-1`` to ``gearing=+1`` (see its cobotta mimic test), so a
    URDF multiplier of +1 is gearing -1 here.
    """
    left = find_joint_prim(stage, robot_path, "gripper_slide_joint_left")
    right = find_joint_prim(stage, robot_path, "gripper_slide_joint_right")
    if not (left and right):
        print(f"[mimic] gripper slide joints not found under {robot_path}", flush=True)
        return
    api = PhysxSchema.PhysxMimicJointAPI.Apply(left, UsdPhysics.Tokens.rotX)
    api.GetReferenceJointRel().SetTargets([right.GetPath()])
    api.GetGearingAttr().Set(-1.0)
    api.GetOffsetAttr().Set(0.0)
    drive = UsdPhysics.DriveAPI.Get(left, "linear")
    if drive:
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(0.0)
    print(
        f"[mimic] q({left.GetName()}) + -1.0 * q({right.GetName()}) = 0 "
        f"(i.e. q_left = +1.0 * q_right); left drive zeroed",
        flush=True,
    )

tools.JOINT_STEP_RAD = args.joint_step
tools.CARRY_JOINT_STEP_RAD = args.carry_joint_step

active_env = None


class NoNetworkClient:
    def __init__(self, **kwargs):
        pass

    def call(self, **kwargs):
        return None


def make_config():
    cfg = load_yaml(str(Path(ENV_CONFIG_PATH) / "litearm.yml"))
    cfg.update(task_name="stack_blocks", num_envs=1, device_id=0,
               eval_batch=False, policy_name="local_grasp_diagnostic", seed=args.seed,
               additional_info="no_policy", eval_num=1, physx_monitor_enabled=True)
    sections = {
        key: load_yaml(str(Path(ENV_CONFIG_PATH) / key / (name + ".yml")))
        for key, name in cfg["config"].items()
    }
    sections.update(
        task_env=load_yaml(task_registry.task_config_path(
            str(Path(ROOT_DIR) / "task" / BENCHMARK / "config"), "stack_blocks")),
        eval_cfg=cfg, deploy_cfg={"port": 1, "policy_name": "local_grasp_diagnostic"})
    result = OmegaConf.create(sections)
    result.sim.scene.num_envs = 1
    result = process_randomization(result)
    result, _ = process_config(result, task_name="stack_blocks")
    result.sim.seed = [args.seed]
    result.sim.device = args.device
    # One diagnostic environment needs far fewer contacts than the default
    # multi-environment reserve; leave GPU memory for rendering on an 8-GB card.
    result.sim.physx = dict(gpu_max_rigid_contact_count=1048576, gpu_max_rigid_patch_count=16384)
    result.camera.default_frequency = 25
    return result


def values(value):
    return np.asarray(value).tolist()


class LocalAdapter:
    def __init__(self, env, emit):
        self.raw = env
        self.emit = emit
        self.phase = "init"
        self.contacts = []
        self.contact_view = None
        self.contact_headers_seen = []
        self._last_sample = None
        self._last_vel = {}
        self.primitives = None

    def contact_report(self, headers, data, anchors):
        for header in headers:
            actors = [str(PhysicsSchemaTools.intToSdfPath(a)) for a in (header.actor0, header.actor1)]
            self.contact_headers_seen.append(actors)
            if not any("gripper_slider_link" in a for a in actors) or not any("/rigid/block/" in a for a in actors):
                continue
            points = data[header.contact_data_offset:header.contact_data_offset + header.num_contact_data]
            friction = anchors[header.friction_anchors_offset:header.friction_anchors_offset + header.num_friction_anchors_data]
            self.contacts.append(dict(actors=actors,
                                      points=[dict(normal=list(p.normal), impulse=list(p.impulse),
                                                   separation=p.separation,
                                                   materials=[str(PhysicsSchemaTools.intToSdfPath(m))
                                                              for m in (p.material0, p.material1)]) for p in points],
                                      friction_impulses=[list(f.impulse) for f in friction]))

    def get_obs(self):
        # Read robot state directly to avoid a camera render on every IK substep.
        rm = self.raw.robot_manager
        state = {}
        for robot in rm.robot_list:
            if robot.type != "target":
                continue
            state[f"{robot.arm_name}_joint_state"] = values(rm.get_joint(robot, [0])[0])
            fingers = np.asarray(rm.get_end_effector_real_val(robot, [0])[0])
            lo, hi = robot.gripper_scale
            opening = (float(fingers.mean()) - lo) / (hi - lo)
            if robot.gripper_move["sign"] != 1:
                opening = 1 - opening
            state[rm.process_name(robot.gripper_name)] = [opening]
        return {"state": state}

    def position(self):
        pos, _ = self.raw.reward_manager.func_parser.get_label_pose(args.label)
        return np.asarray(pos[0], dtype=float)

    def time(self):
        return float(self.raw.sim.unwrapped.sim.current_time)

    def snapshot(self):
        rm = self.raw.robot_manager
        fingers = {
            robot.arm_name: values(rm.get_end_effector_real_val(robot, [0])[0])
            for robot in rm.robot_list if robot.type == "target"
        }
        state = self.get_obs()["state"]
        tcp = {arm: tools.kin.tcp(tools.kin.fk(arm, tools._joints(state, arm))).tolist()
               for arm in tools.ARMS}
        opening = {}
        for robot in rm.robot_list:
            if robot.type != "target":
                continue
            actual = np.asarray(rm.get_end_effector_real_val(robot, [0])[0], dtype=float)
            q = float(actual.mean()) if actual.size else 0.0
            lo, hi = robot.gripper_scale
            value = (q - lo) / (hi - lo)
            if robot.gripper_move["sign"] != 1:
                value = 1 - value
            opening[robot.arm_name] = round(value, 4)
        _pos, rot = self.raw.reward_manager.func_parser.get_label_pose(args.label)
        block_quat = np.asarray(rot[0], dtype=float).tolist() if rot is not None else None
        block_pos = np.asarray(_pos[0], dtype=float)
        w, x, y, z = block_quat if block_quat is not None else (1.0, 0.0, 0.0, 0.0)
        block_rot = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        block_rel = {}
        for arm in tools.ARMS:
            pose = tools.kin.fk(arm, tools._joints(state, arm))
            tcp_p = np.asarray(tools.kin.tcp(pose), dtype=float)
            rot_m = np.asarray(pose[:3, :3], dtype=float)
            block_rel[arm] = {
                "pos": values(rot_m.T @ (block_pos - tcp_p)),
                "rot": values(rot_m.T @ block_rot),
            }
        f0, f1 = (fingers.get("right_arm") or [0.0, 0.0])[:2]
        finger_gap = 0.087 - (float(f0) + float(f1))
        finger_diff = float(f1) - float(f0)
        finger_mid = finger_diff / 2.0
        block_rel_grip = {}
        for arm in tools.ARMS:
            pos = list(block_rel[arm]["pos"])
            pos[1] = float(pos[1]) - finger_mid
            block_rel_grip[arm] = {"pos": pos, "rot": block_rel[arm]["rot"]}
        grip_targets = {}
        try:
            prev = self.raw.robot_manager.control_manager.prev_control[0] or {}
            for key, data in prev.items():
                if "ee_joint_state" in key:
                    grip_targets[key] = values(data.get("position"))
        except Exception:  # noqa: BLE001 - diagnostics must not stop for a missing target
            pass
        now_s = self.time()
        tcp_vel = {}
        tcp_acc = {}
        if self._last_sample is not None:
            dt = now_s - self._last_sample["time"]
            if dt > 1e-9:
                for arm in tools.ARMS:
                    v = (np.asarray(tcp[arm]) - np.asarray(self._last_sample["tcp"][arm])) / dt
                    tcp_vel[arm] = values(v)
                    if arm in self._last_vel:
                        tcp_acc[arm] = values((v - np.asarray(self._last_vel[arm])) / dt)
                    self._last_vel[arm] = v
        self._last_sample = {"time": now_s, "tcp": tcp}
        record = dict(event="sample", phase=self.phase, step=self.raw.take_action_cnt[0],
                      sim_time_s=now_s, block_xyz=block_pos.tolist(),
                      block_quat=block_quat, block_rel=block_rel,
                      block_rel_grip=block_rel_grip,
                      finger_gap_m=round(finger_gap, 6),
                      finger_diff_m=round(finger_diff, 6),
                      finger_mid_m=round(finger_mid, 6),
                      fingers=fingers, opening=opening, tcp_xyz=tcp,
                      grip_targets=grip_targets,
                      tcp_vel=tcp_vel, tcp_acc=tcp_acc)
        record["gripper_estimated_force_n"] = {
            robot.arm_name: values(rm.robot_key[i].data.applied_torque[0, robot.gripper_joint_indices].cpu())
            for i, robot in enumerate(rm.robot_list) if robot.type == "target"
        }
        if args.contacts:
            record["contacts"] = self.contacts[-8:]
            if self.contact_view is not None:
                record["finger_contact_forces_n"] = values(self.contact_view.get_contact_force_matrix(dt=self.raw.dt).cpu())
                record["finger_net_contact_forces_n"] = values(self.contact_view.get_net_contact_forces(dt=self.raw.dt).cpu())
                friction = self.contact_view.get_friction_data(dt=self.raw.dt)[0]
                record["friction_force_sum_n"] = values(friction.sum(dim=0).cpu())
            record["contact_actor_examples"] = self.contact_headers_seen[-8:]
        self.emit(record)
        return record

    def step(self, action):
        if self.raw.end_flag[0] or self.raw.take_action_cnt[0] >= self.raw.step_lim:
            raise RuntimeError("Diagnostic exhausted the episode budget; refusing fake hold samples")
        self.contacts = []
        self.contact_headers_seen = []
        self.raw.take_action(action)
        self.snapshot()
        return None, 0, False, {"status": {"step": self.raw.take_action_cnt[0],
                                          "step_limit": self.raw.step_lim}}


def main():
    global active_env
    from env.camera_manager.camera_manager import REAL_MAP
    for camera in REAL_MAP.values():
        camera["resolution"] = (160, 120)
    from env.robot_manager.robot_config import litearm as robot_config
    if args.force_gripper:
        force_spawn_usd = robot_config.sim_utils.spawn_from_usd

        def force_drives(path, cfg, *a, **kw):
            prim = force_spawn_usd(path, cfg, *a, **kw)
            normalize_gripper_drives(get_current_stage(), path)
            return prim

        robot_config.sim_utils.spawn_from_usd = force_drives
    if args.rigid_pad:
        spawn_material = robot_config.sim_utils.spawn_rigid_body_material

        def rigid_material(path, cfg, *a, **kw):
            if path.endswith("/finger_pad_material"):
                cfg = cfg.replace(compliant_contact_stiffness=0.0, compliant_contact_damping=0.0)
            return spawn_material(path, cfg, *a, **kw)

        robot_config.sim_utils.spawn_rigid_body_material = rigid_material
    if args.velocity_iterations is not None or args.contacts:
        solver_get_config = robot_config.get_robot_config

        def solver_config():
            cfg = solver_get_config()
            if args.velocity_iterations is not None:
                cfg.spawn.articulation_props.solver_velocity_iteration_count = args.velocity_iterations
            if args.contacts:
                cfg.spawn.activate_contact_sensors = True
            return cfg

        robot_config.get_robot_config = solver_config
    if args.mimic:
        mimic_get_config = robot_config.get_robot_config

        def mimic_config():
            cfg = mimic_get_config()
            # Only the right slider is driven; the left follows through the
            # PhysX mimic added in mimic_spawn, and its own drive is zeroed.
            cfg.actuators["gripper"].joint_names_expr = ["gripper_slide_joint_right"]
            return cfg

        robot_config.get_robot_config = mimic_config
        mimic_spawn_usd = robot_config.sim_utils.spawn_from_usd

        def mimic_spawn(path, cfg, *a, **kw):
            prim = mimic_spawn_usd(path, cfg, *a, **kw)
            if path.endswith(("robot0", "robot1")):
                apply_mimic(get_current_stage(), path)
            return prim

        robot_config.sim_utils.spawn_from_usd = mimic_spawn
    eval_env.WsModelClient = NoNetworkClient
    env = eval_env.create_eval_env(make_config(), launcher.app)
    active_env = env
    # RoboDojo's reward expressions use NumPy. Keep their diagnostic pose
    # inputs on CPU while PhysX and robot tensors stay on the selected device.
    layout_pose = env.scene_manager.layout_manager.get_instance_pose

    def numpy_pose(*a, **kw):
        return tuple(v.detach().cpu().numpy() if hasattr(v, "detach") else v
                     for v in layout_pose(*a, **kw))

    env.scene_manager.layout_manager.get_instance_pose = numpy_pose
    if args.contacts:
        # IsaacLab disables contact reporting unless a ContactSensor enables it.
        import carb
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
    # Keep diagnostic videos out of official eval_result success/failure folders.
    env._stream_vision = lambda *a, **kw: None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with output.open("w") as stream:
        for trial in range(args.trials):
            def emit(record):
                line = json.dumps(dict(trial=trial, label=args.label, **record), default=str)
                stream.write(line + "\n")
                stream.flush()
                print("GRASP_DIAG " + line, flush=True)

            env.reset(seed=[args.layout])
            if args.contacts:
                carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
            # Match run_eval: an empty reward checklist otherwise returns 1.0
            # and marks the episode successful after the very first action.
            env.run_reward()
            if hasattr(env, "get_score"):
                env.get_score()
            adapter = LocalAdapter(env, emit)
            if args.contacts:
                from omni.physx import get_physx_simulation_interface
                subscription = get_physx_simulation_interface().subscribe_full_contact_report_events(adapter.contact_report)
            stage = get_current_stage()
            counts = {f"robot{i}": validate_collision_shapes(stage, f"/World/envs/env_0/robot{i}")
                      for i in range(2)}
            usd = ROOT / "assets" / "litearm" / "litearm.usd"
            asset = {}
            if usd.is_file():
                import hashlib
                asset = {"asset_usd": str(usd),
                         "asset_sha256": hashlib.sha256(usd.read_bytes()).hexdigest(),
                         "asset_mtime_s": usd.stat().st_mtime}
            emit(dict(event="setup", collision_shapes=counts, grasp_z=args.grasp_z,
                      yaw=args.yaw, descend_path=args.descend_path, joint_step_rad=args.joint_step,
                      rigid_pad=args.rigid_pad, velocity_iterations=args.velocity_iterations,
                      force_gripper=args.force_gripper, physics_device=env.device,
                      carry_route=args.carry_route, carry_slip_drop=args.carry_slip_drop,
                      carry_joint_step_rad=args.carry_joint_step, **asset))
            materials = []
            for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/robot1")):
                if prim.HasAPI(UsdPhysics.CollisionAPI) and "gripper_slider_link" in str(prim.GetPath()):
                    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                    materials.append(dict(collider=str(prim.GetPath()), material=str(mat.GetPath()),
                                          properties={a.GetName(): a.Get() for a in mat.GetPrim().GetAttributes()
                                                      if a.GetName().startswith(("physics:", "physxMaterial:"))}))
            obj = env.scene_manager.layout_manager.get_scene_object(
                env_idx=0, inst_name=env.scene_manager.layout_manager.get_instance_name(0, label=args.label))
            drives = {}
            joints = {}
            for name in ("gripper_slide_joint_left", "gripper_slide_joint_right"):
                prim = find_joint_prim(stage, "/World/envs/env_0/robot1", name)
                if not prim:
                    continue
                joints[name] = str(prim.GetPath())
                drive = UsdPhysics.DriveAPI.Get(prim, "linear")
                if drive:
                    drives[name] = {
                        attr.GetName(): attr.Get()
                        for attr in prim.GetAttributes()
                        if attr.GetName().startswith(("drive:", "physxMimicJoint:"))
                    }
                rels = {
                    rel.GetName(): [str(t) for t in rel.GetTargets()]
                    for rel in prim.GetRelationships()
                    if rel.GetName().startswith("physxMimicJoint:")
                }
                if rels:
                    drives.setdefault(name, {}).update(rels)
            mimic = {}
            for name, prim_path in joints.items():
                prim = stage.GetPrimAtPath(prim_path)
                attrs = {
                    attr.GetName(): attr.Get()
                    for attr in prim.GetAttributes()
                    if attr.GetName().startswith("physxMimicJoint:")
                }
                if attrs:
                    mimic[name] = attrs
            emit(dict(event="physics", finger_materials=materials, block_mass_kg=float(obj.get_mass()),
                      block_prim=obj.prim_path, joint_prims=joints, drives=drives, mimic=mimic))
            if args.contacts:
                from isaacsim.core.simulation_manager import SimulationManager
                adapter.contact_view = SimulationManager.get_physics_sim_view().create_rigid_contact_view(
                    "/World/envs/env_0/robot1/gripper_slider_link*", filter_patterns=[obj.prim_path],
                    max_contact_data_count=256)
            primitives = SimpleNamespace(env=adapter, _check_cancelled=lambda: None)
            adapter.primitives = primitives
            start = {arm: tools._joints(adapter.get_obs()["state"], arm) for arm in tools.ARMS}
            initial = adapter.position().copy()
            x, y, z = initial
            adapter.snapshot()

            def perform(name, fn, **kwargs):
                adapter.phase = name
                result = fn(primitives, **kwargs)
                emit(dict(event="tool", phase=name, result=result))
                return result

            perform("half_open", tools.grip, arm="right", gripper=-1, opening=0.5)
            perform("full_open", tools.grip, arm="right", gripper=-1)
            perform("home", tools.home, start_joints=start, arm="right")
            motion = dict(start_joints=start, arm="right", pitch=90, yaw=args.yaw, roll=0, path="auto")
            approach = perform("approach", tools.move_pose, xyz=[x, y, z + 0.1175], gripper=-1, **motion)
            if not approach.get("reached") or approach.get("blocked"):
                failures += 1
                emit(dict(event="verdict", success=False, reason="approach_failed", initial=initial.tolist()))
                continue
            descend_motion = dict(motion, path=args.descend_path)
            descend = perform("descend", tools.move_pose, xyz=[x, y, args.grasp_z], gripper=-1, **descend_motion)
            if not descend.get("reached") or descend.get("blocked"):
                failures += 1
                emit(dict(event="verdict", success=False, reason="path_failed", initial=initial.tolist()))
                continue
            # Preserve the actual descent orientation, including the selected
            # half-turn. Reusing pitch=90/yaw=-90/roll=0 would rotate a held cube
            # back by 180 degrees because closing disables half-turn selection.
            q_descend = tools._joints(adapter.get_obs()["state"], "right")
            angles = tools.kin.angles(tools.kin.fk("right", q_descend)[:3, :3])
            carry_motion = dict(motion, **dict(zip(("pitch", "yaw", "roll"), angles)))
            emit(dict(event="carry_orientation", pitch=angles[0], yaw=angles[1], roll=angles[2]))
            close = perform("close", tools.grip, arm="right", gripper=1)
            if not close.get("settled"):
                failures += 1
                emit(dict(event="verdict", success=False, reason="gripper_unsettled"))
                continue
            lift = perform("lift", tools.move_pose, xyz=[x, y, args.lift_z], gripper=1, **carry_motion)
            if args.hover_s > 0.0:
                adapter.phase = "hover"
                state = adapter.get_obs()["state"]
                joints = {arm: tools._joints(state, arm) for arm in tools.ARMS}
                hover_start = adapter.time()
                while adapter.time() - hover_start < args.hover_s:
                    state, _ = tools._act(primitives, state, joints, {"right": 0.0})
                final = adapter.position().copy()
                success = bool(final[2] >= initial[2] + 0.03)
                failures += not success
                emit(dict(event="verdict", success=success, mode="hover",
                          hover_s=args.hover_s, initial=initial.tolist(),
                          final=final.tolist()))
                continue
            if args.place_on:
                tools.CARRY_SLIP_DROP = args.carry_slip_drop
                support_pos, _ = adapter.raw.reward_manager.func_parser.get_label_pose(
                    args.place_support_label
                )
                sx, sy = float(support_pos[0][0]), float(support_pos[0][1])
                support_top = float(support_pos[0][2]) + args.support_top_offset
                approach = perform("place_approach", tools.move_pose, xyz=[sx, sy, 0.90],
                                   gripper=0, **dict(carry_motion, path=args.carry_path))
                if not approach.get("reached") or approach.get("blocked") or approach.get("slip"):
                    failures += 1
                    emit(dict(event="verdict", success=False, reason="place_approach_failed",
                              initial=initial.tolist()))
                    continue
                placed = tools.place_on(
                    primitives, start, x=sx, y=sy, support_top=support_top,
                    tcp_to_bottom=args.tcp_to_bottom, arm="right", clearance=0.002,
                    lift_m=0.12, retreat_xy=(0.0, -0.10), path=args.carry_path,
                )
                emit(dict(event="tool", phase="place_on", result=placed))
                if args.home_after_place:
                    state = adapter.get_obs()["state"]
                    emit(dict(
                        event="grip_state", phase="before_home",
                        commands=dict(getattr(primitives, "_grip_commands", {}) or {}),
                        keeps={arm: tools._keep(primitives, state, arm) for arm in tools.ARMS},
                        measured={arm: tools._opening(state, arm) for arm in tools.ARMS},
                    ))
                    home = tools.home(primitives, start, arm="both")
                    emit(dict(event="tool", phase="home_after_place", result=home))
                    after = adapter.get_obs()["state"]
                    emit(dict(
                        event="grip_state", phase="after_home",
                        commands=dict(getattr(primitives, "_grip_commands", {}) or {}),
                        measured={arm: tools._opening(after, arm) for arm in tools.ARMS},
                    ))
                final = adapter.position().copy()
                xy_error = float(np.linalg.norm(np.asarray(final[:2]) - np.asarray([sx, sy])))
                success = bool(
                    placed.get("placed")
                    and placed.get("safe_to_home")
                    and final[2] >= initial[2] + 0.03
                    and xy_error <= 0.02
                )
                failures += not success
                emit(dict(event="verdict", success=success, mode="place_on",
                          initial=initial.tolist(), final=final.tolist(),
                          xy_error=xy_error, support_top=support_top,
                          place_z=placed.get("place_z"),
                          support_label=args.place_support_label))
                continue
            if args.carry_route != "none":
                tools.CARRY_SLIP_DROP = args.carry_slip_drop
                route = {
                    "second-yellow": [[0.2822, -0.1226, 0.90],
                                      [0.2693, -0.1434, 0.90],
                                      [0.2693, -0.1434, 0.855]],
                    "second-white": [[0.3481, -0.1639, 0.86],
                                     [0.3221, -0.1582, 0.86],
                                     [0.2960, -0.1526, 0.86]],
                }[args.carry_route]
                ok = True
                for i, waypoint in enumerate(route):
                    carry = perform(f"carry_{i}", tools.move_pose, xyz=waypoint, gripper=1,
                                    **dict(carry_motion, path=args.carry_path,
                                           smooth=args.carry_smooth))
                    if carry.get("slip") or not carry.get("reached") or carry.get("blocked"):
                        ok = False
                        break
                final = adapter.position().copy()
                success = bool(ok and final[2] >= initial[2] + 0.03)
                failures += not success
                emit(dict(event="verdict", success=success, route=args.carry_route,
                          initial=initial.tolist(), final=final.tolist(),
                          joint_step_rad=args.joint_step,
                          carry_slip_drop=args.carry_slip_drop))
                continue
            adapter.phase = "hold_1s"
            state = adapter.get_obs()["state"]
            joints = {arm: tools._joints(state, arm) for arm in tools.ARMS}
            samples = []
            hold_start = adapter.time()
            while adapter.time() - hold_start < 1.0 - 1e-8:
                state, _ = tools._act(primitives, state, joints, {"right": 0.0})
                samples.append(adapter.position().copy())
            hold_duration = adapter.time() - hold_start
            transport = perform("transport", tools.move_pose, xyz=[x + 0.05, y, args.lift_z], gripper=1, **carry_motion)
            adapter.phase = "transport_hold_1s"
            state = adapter.get_obs()["state"]
            joints = {arm: tools._joints(state, arm) for arm in tools.ARMS}
            carry_samples = []
            carry_hold_start = adapter.time()
            while adapter.time() - carry_hold_start < 1.0 - 1e-8:
                state, _ = tools._act(primitives, state, joints, {"right": 0.0})
                carry_samples.append(adapter.position().copy())
            final = adapter.position()
            held = min(p[2] for p in samples) >= initial[2] + 0.03
            carried = all(p[2] >= initial[2] + 0.03 and p[0] - initial[0] >= 0.04 for p in carry_samples)
            motions_ok = lift.get("reached") and not lift.get("blocked") and transport.get("reached") and not transport.get("blocked")
            success = bool(held and carried and motions_ok)
            failures += not success
            emit(dict(event="verdict", success=success, held_1s=bool(held), carried=bool(carried),
                      initial=initial.tolist(), final=final.tolist(),
                      min_hold_z=float(min(p[2] for p in samples)), hold_duration_s=hold_duration,
                      min_transport_hold_z=float(min(p[2] for p in carry_samples)),
                      transport_hold_duration_s=adapter.time() - carry_hold_start))
    return 1 if failures else 0


exit_code = 2
try:
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    exit_code = main()
except BaseException:
    # Kit teardown can block; print the original error before attempting it.
    traceback.print_exc()
    sys.stderr.flush()
finally:
    if active_env is not None:
        try:
            # A stopped Kit timeline normally enters an interactive render loop.
            # This standalone diagnostic must terminate even on an exception.
            if active_env.sim is not None:
                active_env.sim.unwrapped.sim._disable_app_control_on_stop_handle = True
            active_env.close()
        except BaseException:
            traceback.print_exc()
            sys.stderr.flush()
    print("GRASP_DIAG closing SimulationApp", file=sys.stderr, flush=True)
    # Kit shutdown may terminate Python before the final sys.exit is reached.
    # Give Kit the diagnostic status before releasing its framework.
    launcher.app.app.post_quit(exit_code)
    launcher.app.close()
    faulthandler.cancel_dump_traceback_later()
sys.exit(exit_code)
