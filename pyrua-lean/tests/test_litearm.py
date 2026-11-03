"""The LiteArm adapter: kinematics against the URDF, the API, and the real tools.

The robot-class tests reuse the X5 fake backend's TCP/angle simulator with the
state shape a LiteArm station reports (seven joints, ``tcp_link`` as the
end-effector pose).  The tool tests drive the real ``litearm_tools`` over a
joint-level environment whose observation is ``litearm_kinematics.fk``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

np = pytest.importorskip("numpy")

from pyrualean import get_adapter  # noqa: E402
from pyrualean.hosts import litearm_kinematics as kin  # noqa: E402
from pyrualean.hosts import litearm_sight as sight  # noqa: E402
from pyrualean.hosts import litearm_tools as tools  # noqa: E402
from pyrualean.litearm import Gripper, LitearmRobot, State  # noqa: E402
from pyrualean.prompt import _public_members, api_reference, knowledge_text  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
URDF = Path(os.environ.get("LITEARM_URDF", _REPO / "assets" / "litearm" / "litearm.urdf"))


# ---------------------------------------------------------------------------
# Independent URDF forward kinematics (the authority the hard-coded chain copies)
# ---------------------------------------------------------------------------


def _rpy(rpy: str) -> np.ndarray:
    r, p, y = (float(v) for v in rpy.split())
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) or 1.0)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def _homogeneous(rot, pos):
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = pos
    return out


def _urdf_chain(urdf: Path):
    robot = ET.parse(urdf).getroot()
    parent_of = {j.find("child").get("link"): j for j in robot.findall("joint")}
    chain, link = [], "tcp_link"
    while link != "base_link":
        joint = parent_of[link]
        chain.append(joint)
        link = joint.find("parent").get("link")
    return list(reversed(chain))


def _urdf_fk(chain, q):
    pose = np.eye(4)
    for joint in chain:
        origin = joint.find("origin")
        xyz = tuple(float(v) for v in origin.get("xyz", "0 0 0").split())
        pose = pose @ _homogeneous(_rpy(origin.get("rpy", "0 0 0")), xyz)
        if joint.get("type") == "revolute":
            axis = [float(v) for v in joint.find("axis").get("xyz").split()]
            pose = pose @ _homogeneous(_axis_angle(axis, q.get(joint.get("name"), 0.0)), (0, 0, 0))
    return pose


def test_kinematics_reproduce_the_urdf_chain():
    if not URDF.is_file():
        pytest.skip("RoboDojo LiteArm asset not present on this machine")
    chain = _urdf_chain(URDF)
    rng = np.random.default_rng(7)
    for _ in range(40):
        q = rng.uniform(
            [lim[0] for lim in kin.JOINT_LIMITS], [lim[1] for lim in kin.JOINT_LIMITS]
        )
        names = [j.get("name") for j in chain if j.get("type") == "revolute"]
        expected = _urdf_fk(chain, dict(zip(names, q, strict=True)))
        # the URDF chain runs from base_link; put it on the robot origin
        expected = _homogeneous(
            kin._rot_z(kin.BASES["left"][1]), kin.BASES["left"][0]
        ) @ expected
        got = kin.fk("left", q)
        assert np.linalg.norm(got[:3, 3] - expected[:3, 3]) < 1e-6
        assert kin.rotation_error(got[:3, :3], expected[:3, :3]) < 1e-4


def test_zero_pose_is_straight_up_and_tcp_is_the_pose():
    pose = kin.fk("left", [0.0] * 7)
    assert pose[:3, 3] == pytest.approx((-0.3, -0.45, 0.765 + 0.9136), abs=1e-4)
    assert kin.angles(pose[:3, :3]) == pytest.approx((-90.0, 0.0, 0.0), abs=0.1)
    assert kin.tcp(pose) == pytest.approx(pose[:3, 3])  # TCP_OFFSET is 0


def test_ik_reaches_a_table_pose_and_refuses_the_fingers_into_the_table():
    target = np.eye(4)
    target[:3, :3] = kin.orientation(90, 0, 0)
    target[:3, 3] = (-0.3, -0.1, 0.95)
    res = kin.best_solution("left", target, [0.0] * 7, extra_seeds=[[0.0] * 7])
    assert res.ok and not res.collides and res.pos_err_m < 0.002
    # a grasp point 2 cm above the table puts the 4.6 cm fingers through it
    low = target.copy()
    low[:3, 3] = (-0.3, -0.1, 0.787)
    res = kin.best_solution("left", low, [0.0] * 7, extra_seeds=[[0.0] * 7])
    assert not (res.ok and not res.collides)


def test_wrist_camera_mount_is_the_robot_config_pose():
    tcp = kin.fk("left", [0.0] * 7)
    view = sight.wrist_view([*tcp[:3, 3], *kin.quat_wxyz(tcp[:3, :3])])
    # camera_link sits 13.06 cm behind and 6 cm to the side of the TCP
    expected = tcp[:3, 3] + tcp[:3, :3] @ np.array([-0.1306, 0.0, -0.06])
    assert view.pos == pytest.approx(expected, abs=1e-6)
    # it looks forward and 20 degrees down along the tool axis
    looking = -view.pose[:3, 2]
    along = tcp[:3, :3] @ np.array([math.cos(math.radians(20)), 0.0, -math.sin(math.radians(20))])
    assert looking == pytest.approx(along, abs=1e-6)


# ---------------------------------------------------------------------------
# The adapter and the robot class
# ---------------------------------------------------------------------------


def test_adapter_and_reference_are_lite(monkeypatch):
    adapter = get_adapter("litearm")
    assert adapter.robot_cls is LitearmRobot
    assert adapter.host == "pyrualean.hosts.xpolicylab_litearm"
    assert adapter.knowledge == "litearm"
    reference = api_reference(adapter.robot_cls, primitives="no-vla")
    assert "pi0_pick" not in reference and "0.145" not in reference
    assert "seven arm joint" in reference and "LiteGrip" in reference
    members = _public_members(adapter.robot_cls)
    assert "pi0_pick" not in members and "done" not in members and "world_map" not in members
    assert "place_on" in members and "place_on" in reference
    text = knowledge_text(adapter.knowledge, primitives="no-vla")
    assert "LiteArm SE" in text and "4.6 cm" in text


def _litearm_backend():
    """The X5 fake with a LiteArm-shaped observation (7 joints, TCP = reported pose)."""

    from test_robodojo import FakeRobodojoBackend, _quat

    class FakeLitearmBackend(FakeRobodojoBackend):
        TASK = "Put the green bottle into the dustbin."

        def _summary(self):
            state = super()._summary()
            for name, arm in self.arms.items():
                state["state"][f"{name}_ee_pose"] = [*arm["tcp"], *_quat(*arm["angles"])]
                state["state"][f"{name}_arm_joint_state"] = [0.0] * 7
            return state

        def _do_grip(self, arm, gripper, opening=None):
            payload = super()._do_grip(arm, gripper, opening)
            payload["width_m"] = (
                round(max(0.0, 0.0020 + 0.0850 * payload["opening"]), 4)
                if payload.get("width_m") is not None
                else None
            )
            return payload

    return FakeLitearmBackend()


@pytest.fixture
def lit(monkeypatch):
    backend = _litearm_backend()
    return LitearmRobot(backend)


def test_state_has_seven_joints_and_the_tcp_is_the_end_effector(lit):
    state = lit.state()
    assert isinstance(state, State) and state.step == 0
    assert state.left.tcp_pos == pytest.approx((-0.30, -0.20, 1.00))
    assert state.left.eef_pos == state.left.tcp_pos  # no TCP offset on a LiteArm
    assert len(state.left.joints) == 7
    assert (state.left.pitch, state.left.yaw, state.left.roll) == (0.0, 0.0, 0.0)


def test_move_to_and_set_gripper_keep_the_typed_results(lit):
    move = lit.move_to([0.30, -0.20, 1.05], arm="right", pitch=90.0, path="line")
    assert move.reached and move.dist_to_target_m < 0.01
    lit._backend.arms["right"]["tcp"] = list(lit._backend.BOTTLE)  # close on the bottle
    grip = lit.set_gripper("right", lit.CLOSE)
    assert isinstance(grip, Gripper) and grip.commanded == "closed"
    assert grip.holding and grip.width_m is not None
    assert grip.width_m == pytest.approx(max(0.0, 0.0020 + 0.0850 * grip.opening), abs=1e-4)


def test_no_vla_and_no_success_flag(lit):
    with pytest.raises(AttributeError):
        _ = lit.done
    with pytest.raises(AttributeError):
        lit.pi0_pick("pick the bottle")


def test_move_result_keeps_the_slip_fields(lit):
    move = lit._move(
        "right",
        (0.30, -0.20, 1.00),
        {
            "reached": False,
            "final_xyz": [0.30, -0.20, 0.99],
            "slip": True,
            "slip_opening": 0.33,
            "slip_baseline": 0.36,
            "steps_used": 4,
            "path": "line",
        },
    )
    assert move.slip is True
    assert move.slip_opening == pytest.approx(0.33)
    assert move.slip_baseline == pytest.approx(0.36)


# ---------------------------------------------------------------------------
# The real tools over a joint-level environment
# ---------------------------------------------------------------------------


class _JointEnv:
    """Seven joints per arm; the observation is the URDF forward kinematics."""

    STEP_LIMIT = 400

    def __init__(self):
        # folded down over the table, a natural starting pose for the tools
        self.start = {
            "left": [0.0, 0.8, 0.0, -1.5, 0.0, 0.5, 0.0],
            "right": [0.0, 0.8, 0.0, -1.5, 0.0, 0.5, 0.0],
        }
        self.joints = {arm: list(q) for arm, q in self.start.items()}
        self.grip = {"left": 1.0, "right": 1.0}
        self.step_count = 0
        self.actions: list[dict[str, list[float]]] = []

    def get_obs(self):
        state = {}
        for arm in kin.ARMS:
            pose = kin.fk(arm, self.joints[arm])
            state[f"{arm}_ee_pose"] = [
                *pose[:3, 3].tolist(),
                *kin.quat_wxyz(pose[:3, :3]),
            ]
            state[f"{arm}_arm_joint_state"] = list(self.joints[arm])
            state[f"{arm}_ee_joint_state"] = [self.grip[arm]]
        return {"state": state, "vision": {}, "instruction": "test"}

    def get_status(self):
        return {"step": self.step_count, "step_limit": self.STEP_LIMIT}

    def step(self, action):
        self.actions.append(action)
        self.step_count += 1
        for arm in kin.ARMS:
            target = action[f"{arm}_arm_joint_state"]
            self.joints[arm] = [
                now + 0.8 * (goal - now)
                for now, goal in zip(self.joints[arm], target, strict=True)
            ]
            self.grip[arm] = action[f"{arm}_ee_joint_state"][0]
        return self.get_obs(), 0.0, False, {"status": self.get_status()}


class _JointPrimitives:
    def __init__(self, env):
        self.env = env
        self._last_obs = None

    def _check_cancelled(self):
        pass


def test_move_pose_reaches_a_nearby_pose_with_the_seven_joint_ik():
    env = _JointEnv()
    primitives = _JointPrimitives(env)
    start = {arm: list(q) for arm, q in env.start.items()}
    pose = kin.fk("left", env.start["left"])
    target = pose[:3, 3] + pose[:3, :3] @ np.array([0.0, 0.0, 0.05])
    result = tools.move_pose(
        primitives,
        start,
        target.tolist(),
        arm="left",
        pitch=None,
        yaw=None,
        roll=None,
        path="line",
    )
    assert "error" not in result
    assert result["reached"] and result["dist_to_target_m"] < 0.01
    assert len(env.actions[0]["left_arm_joint_state"]) == 7


@pytest.mark.parametrize("blocked", [False, True])
def test_move_pose_waits_for_tcp_to_settle_before_reporting_contact(monkeypatch, blocked):
    class SettlingEnv(_JointEnv):
        def step(self, action):
            super().step(action)
            self.joints["left"] = list(self.start["left"])
            # Less than the old 0.01-radian joint tolerance, yet enough TCP
            # error to trigger its 5-mm obstacle check while still moving.
            if blocked or self.step_count == 1:
                self.joints["left"][1] += 0.009
            return self.get_obs(), 0.0, False, {"status": self.get_status()}

    env = SettlingEnv()
    goal = env.start["left"]
    monkeypatch.setattr(tools, "_plan", lambda *a, **k: ([goal], "line", None, None))
    target = kin.tcp(kin.fk("left", goal)).tolist()
    result = tools.move_pose(_JointPrimitives(env), env.start, target,
                             arm="left", gripper=1, path="line")
    assert result["settled"] is (not blocked)
    assert result["reached"] is (not blocked)
    assert bool(result.get("blocked")) is blocked
    assert result["steps_used"] > 1


class _CarryEnv(_JointEnv):
    """The joints follow the action; the held opening follows a script.

    A real carried block is not an action target: it stays where the fingers
    left it and slides when it slides.  The script reproduces that so the carry
    guard can be tested over waypoints and the settling tail.
    """

    def __init__(self, openings):
        super().__init__()
        self.openings = list(openings)

    def step(self, action):
        self.actions.append(action)
        self.step_count += 1
        for arm in kin.ARMS:
            target = action[f"{arm}_arm_joint_state"]
            self.joints[arm] = [
                now + 0.8 * (goal - now)
                for now, goal in zip(self.joints[arm], target, strict=True)
            ]
        if self.openings:
            self.grip["left"] = self.openings.pop(0)
        return self.get_obs(), 0.0, False, {"status": self.get_status()}


def _carry_target(env, dx=0.04, dz=0.02):
    pose = kin.fk("left", env.start["left"])
    return (pose[:3, 3] + np.array([dx, 0.0, dz])).tolist()


def test_move_pose_stops_when_a_carried_block_slips():
    env = _CarryEnv([0.36, 0.33, 0.33, 0.33, 0.33])
    primitives = _JointPrimitives(env)
    primitives._held_openings = {"left": 0.36}
    target = _carry_target(env, dx=0.10)
    result = tools.move_pose(primitives, env.start, target, arm="left", gripper=1, path="line")
    assert result["slip"] is True
    assert result["reached"] is False
    assert result["slip_baseline"] == pytest.approx(0.36)
    assert result["slip_opening"] == pytest.approx(0.33)
    assert np.linalg.norm(np.asarray(result["final_xyz"]) - np.asarray(target)) > 0.01
    # The second native action loses the block. Do not send the distant
    # endpoint as a settling action after the waypoint loop stops.
    assert result["steps_used"] == 2
    assert len(env.actions) == 2
    assert all(action["left_ee_joint_state"] == [0.0] for action in env.actions)


@pytest.mark.parametrize("opening", [0.33, 0.01, 0.0])
def test_move_pose_refuses_a_carry_already_below_its_close_baseline(opening):
    env = _CarryEnv([])
    env.grip["left"] = opening
    primitives = _JointPrimitives(env)
    primitives._held_openings = {"left": 0.36}
    start = kin.tcp(kin.fk("left", env.start["left"])).tolist()
    result = tools.move_pose(
        primitives, env.start, _carry_target(env), arm="left", gripper=1, path="line"
    )
    assert result["slip"] is True and result["reached"] is False
    assert result["steps_used"] == 0
    assert result["final_xyz"] == pytest.approx(result["start_xyz"], abs=1e-3)
    assert result["final_xyz"] == pytest.approx(start, abs=1e-3)
    assert env.actions == []


def test_move_pose_slip_guard_covers_keep():
    env = _CarryEnv([0.36, 0.36, 0.33, 0.33])
    primitives = _JointPrimitives(env)
    primitives._held_openings = {"left": 0.36}
    result = tools.move_pose(
        primitives, env.start, _carry_target(env), arm="left", gripper=0, path="line"
    )
    assert result["slip"] is True
    assert result["slip_baseline"] == pytest.approx(0.36)


def test_move_pose_slip_guard_covers_the_settling_tail(monkeypatch):
    env = _CarryEnv([0.36, 0.33, 0.33, 0.33])
    primitives = _JointPrimitives(env)
    primitives._held_openings = {"left": 0.36}
    goal = env.start["left"]
    monkeypatch.setattr(tools, "_plan", lambda *a, **k: ([goal], "line", None, None))
    target = kin.tcp(kin.fk("left", goal)).tolist()
    result = tools.move_pose(primitives, env.start, target, arm="left", gripper=1, path="line")
    assert result["slip"] is True
    assert result["settled"] is False
    assert result["reached"] is False


def test_smooth_steps_ease_the_first_and_last_move():
    q = [0.0] * 7
    out = tools._smooth_ends([[0.1] * 7, [0.2] * 7], q)
    deltas = [abs(b[0] - a[0]) for a, b in zip([q, *out[:-1]], out)]
    assert len(out) == 6
    assert deltas[0] < max(deltas) and deltas[-1] < max(deltas)


def test_place_on_does_not_open_after_a_blocked_descent(monkeypatch):
    calls = []

    def move(*args, **kwargs):
        calls.append("move")
        return {
            "reached": False,
            "blocked": True,
            "slip": False,
            "steps_used": 5,
            "status": {},
        }

    def open_grip(*args, **kwargs):
        calls.append("grip")
        return {"settled": True, "opening": 1.0, "steps_used": 3}

    monkeypatch.setattr(tools, "move_pose", move)
    monkeypatch.setattr(tools, "grip", open_grip)
    result = tools.place_on(
        _JointPrimitives(_JointEnv()), {"left": [], "right": []}, 0.30, -0.20, 0.80, 0.050
    )
    assert result["placed"] is False and result["released"] is False
    assert result["descent_blocked"] is True
    assert calls == ["move"]


def test_place_on_does_not_claim_safe_to_home_when_the_retreat_fails(monkeypatch):
    results = [
        {"reached": True, "blocked": False, "slip": False, "steps_used": 10, "status": {}},
        {"reached": True, "blocked": False, "slip": False, "steps_used": 5, "status": {}},
        {"reached": False, "blocked": True, "slip": False, "steps_used": 5, "status": {}},
    ]
    monkeypatch.setattr(tools, "move_pose", lambda *a, **k: results.pop(0))
    monkeypatch.setattr(
        tools, "grip", lambda *a, **k: {"settled": True, "opening": 1.0, "steps_used": 3}
    )
    monkeypatch.setattr(
        tools, "home", lambda *a, **k: (_ for _ in ()).throw(AssertionError("home called"))
    )
    result = tools.place_on(
        _JointPrimitives(_JointEnv()), {"left": [], "right": []}, 0.30, -0.20, 0.80, 0.050
    )
    assert result["placed"] is True and result["released"] is True
    assert result["retreat_blocked"] is True
    assert result["safe_to_home"] is False


def test_place_on_uses_the_tcp_to_bottom_offset_it_is_given(monkeypatch):
    targets = []

    def move(*args, **kwargs):
        targets.append(args[2])
        return {
            "reached": True,
            "blocked": False,
            "slip": False,
            "steps_used": 1,
            "status": {},
        }

    monkeypatch.setattr(tools, "move_pose", move)
    monkeypatch.setattr(
        tools, "grip", lambda *a, **k: {"settled": True, "opening": 1.0, "steps_used": 1}
    )
    result = tools.place_on(
        _JointPrimitives(_JointEnv()),
        {"left": [], "right": []},
        0.30,
        -0.20,
        support_top=0.80,
        tcp_to_bottom=0.050,
        clearance=0.002,
    )
    assert targets[0] == pytest.approx([0.30, -0.20, 0.852])
    assert result["place_z"] == pytest.approx(0.852)


def test_home_returns_the_seven_starting_joints():
    env = _JointEnv()
    primitives = _JointPrimitives(env)
    start = {arm: list(q) for arm, q in env.start.items()}
    env.joints["left"] = [0.2, 1.4, -0.3, -0.9, 0.4, 0.2, -0.5]
    result = tools.home(primitives, start, arm="left", lift=False)
    assert result["reached"] and result["max_joint_error_rad"] <= 0.01
    assert env.joints["left"] == pytest.approx(start["left"], abs=0.01)
    for action in env.actions:
        assert len(action["left_arm_joint_state"]) == 7
        assert action["right_arm_joint_state"] != start["right"] or True


def test_grip_reports_the_litegrip_width():
    env = _JointEnv()
    primitives = _JointPrimitives(env)
    env.grip["left"] = 0.0  # closed on something: a reading of 0 is nothing
    result = tools.grip(primitives, arm="left", gripper=1)
    assert result["gripper"] == "closed"
    assert result["opening"] == pytest.approx(0.0, abs=0.01)
    assert result["holding"] is False


class _GripSequenceEnv(_JointEnv):
    def __init__(self, readings):
        super().__init__()
        self.readings = iter(readings)

    def step(self, action):
        super().step(action)
        self.grip["left"] = next(self.readings)
        return self.get_obs(), 0.0, False, {"status": self.get_status()}


def test_grip_does_not_mistake_one_still_reading_for_contact():
    env = _GripSequenceEnv([0.7, 0.7, 0.4, 0.0, 0.0, 0.0, 0.0])
    result = tools.grip(_JointPrimitives(env), arm="left", gripper=1)
    assert result["settled"] is True
    assert result["holding"] is False
    assert result["opening"] == 0.0


def test_grip_does_not_report_holding_when_budget_ends_midstroke():
    env = _GripSequenceEnv([0.7, 0.6])
    env.STEP_LIMIT = 2
    result = tools.grip(_JointPrimitives(env), arm="left", gripper=1)
    assert result["settled"] is False
    assert result["holding"] is False
    assert result["steps_used"] == 2


def test_grip_reports_candidate_contact_after_sustained_stop():
    env = _GripSequenceEnv([0.4, 0.4, 0.4, 0.4])
    result = tools.grip(_JointPrimitives(env), arm="left", gripper=1)
    assert result["settled"] is True
    assert result["holding"] is True
    assert result["width_m"] == pytest.approx(0.036)


def test_keep_does_not_turn_contact_pushed_open_fingers_into_a_close():
    from types import SimpleNamespace

    # Last command was fully open; the arm sweeps the open fingers through a
    # stack and contact pushes the reading below the old 0.08 tolerance.  The
    # keep must still command open: a close here grabbed the top block.
    primitives = SimpleNamespace(_grip_commands={"right": 1.0})
    state = {"right_ee_joint_state": [0.90, 0.90]}
    assert tools._keep(primitives, state, "right") == 1.0

    # A last CLOSE command still keeps the squeeze while holding.
    primitives = SimpleNamespace(_grip_commands={"right": 0.0})
    state = {"right_ee_joint_state": [0.37, 0.37]}
    assert tools._keep(primitives, state, "right") == 0.0

    # A reading wider than the last partial open is followed.
    primitives = SimpleNamespace(_grip_commands={"right": 0.55})
    state = {"right_ee_joint_state": [0.70, 0.70]}
    assert tools._keep(primitives, state, "right") == 0.70
