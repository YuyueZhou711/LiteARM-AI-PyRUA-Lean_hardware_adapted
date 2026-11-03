from typing import Callable

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.sim.utils import clone
from pxr import Usd, UsdPhysics

from env.global_configs import ROBOTS_PATH

#: Measured/assumed friction of the soft grip pad, applied to the official finger
#: meshes; no extra pad geometry is added.  0.7 is the predicted value until a
#: pull/slope test replaces it.
PAD_MU = 0.7


@clone
def _spawn_litearm(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the LiteArm USD, then bind a high-friction material to the two finger faces.

    ``UsdFileCfg`` has no physics-material field; the material is created under the robot
    and bound to every collision prim of ``gripper_slider_link1/2`` (the official finger
    meshes; no extra pad geometry).  The bowl's own 0.25
    friction stays untouched; the ``max`` combine makes the pad's value dominate the
    finger-bowl contact.

    ``@clone`` resolves the ``env_*``/``robot*`` prim-path regex before the material
    lookup: the robot spawner calls ``func`` once with the regex, and without the
    decorator ``GetPrimAtPath('/World/envs/env_.*/robot0/...')`` is an invalid SdfPath.
    """
    prim = sim_utils.spawn_from_usd(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs
    )
    stage = sim_utils.get_current_stage()
    material_path = f"{prim_path}/finger_pad_material"
    sim_utils.spawn_rigid_body_material(
        material_path,
        sim_utils.RigidBodyMaterialCfg(
            static_friction=PAD_MU,
            dynamic_friction=PAD_MU,
            restitution=0.0,
            friction_combine_mode="max",
            # take the smaller restitution of the pair: the object materials
            # used to be 0.5 and the contact bounced the block out.
            restitution_combine_mode="min",
            # The real 2.3 mm pad is foam: a rigid flat contact cams a rounded
            # watch out during the lift.  A compliant contact (implicit spring)
            # makes the finger face yield like the foam without adding any
            # geometry; 2e4 N/m reaches the 35 N close force at ~1.7 mm.
            compliant_contact_stiffness=2.0e4,
            compliant_contact_damping=2.0e2,
        ),
    )
    for link_name in ("gripper_slider_link1", "gripper_slider_link2"):
        link = stage.GetPrimAtPath(f"{prim_path}/{link_name}")
        if not link.IsValid():
            continue
        for child in Usd.PrimRange(link):
            if child.HasAPI(UsdPhysics.CollisionAPI):
                sim_utils.bind_physics_material(
                    str(child.GetPath()), material_path, stage=stage
                )
    return prim


@configclass
class LitearmUsdFileCfg(sim_utils.UsdFileCfg):
    """LiteArm USD plus the finger-pad physics material."""

    func: Callable = _spawn_litearm


def get_robot_config():
    return ArticulationCfg(
        spawn=LitearmUsdFileCfg(
            usd_path=f"{ROBOTS_PATH}/litearm/litearm.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                fix_root_link=True,
            ),
        ),
        # Park pose, close to the X5 station's folded start (X5 tcp
        # [0.300, -0.207, 0.922], pitch 0): tcp [±0.300, -0.200, 0.950],
        # pitch 10°, yaw -10°.  The arms lie forward over the table so the
        # wrist cameras see the table/bowls at reset, as they do on the X5
        # station, and the left/right locate calls get a second view.  The
        # URDF zero pose is still straight up; this is the robot's start.
        # IK 0.1 mm / 0.04°, no table or self collision; joint4 is at
        # -2.957 rad, 0.132 rad inside its -3.089 rad limit.
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": -1.204,
                "joint2": -0.6622,
                "joint3": 0.2572,
                "joint4": -2.957,
                "joint5": 2.2126,
                "joint6": 0.437,
                "joint7": 0.651,
                # LiteGrip: 0 is fully open
                "gripper_slide_joint_right": 0.0,
                "gripper_slide_joint_left": 0.0,
            },
            pos=(0.25, -0.25, 0.0),
            rot=(0.707, 0, 0, 0.707),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-7]"],
                # URDF effort limits: 97 / 97 / 27 / 27 / 12.5 / 12.5 / 12.5 N*m
                effort_limit_sim={
                    "joint1": 97.0,
                    "joint2": 97.0,
                    "joint3": 27.0,
                    "joint4": 27.0,
                    "joint5": 12.5,
                    "joint6": 12.5,
                    "joint7": 12.5,
                },
                velocity_limit_sim=5.0,
                stiffness=4400.0,
                damping=40.0,
                armature=0.01,
            ),
            # LiteGrip parallel fingers (prismatic, 0.0425 m per finger).
            # The asset constrains the pair with a PhysX mimic (q_left = q_right,
            # from the URDF mimic tag; gearing=-1 in PhysX's additive form) and
            # zeroes the follower's drive, like the real single motor + rack.
            # Only the right/reference joint is actuated here; including the
            # follower would write gains to it at runtime and fight the mimic.
            # SDK values (litegrip-python): grasp force_n default 20 N
            # (= 2.0 N.m at N_TO_NM = 0.1), rated max 35 N (3.5 N.m);
            # open/close/grasp speed 50 mm/s of JAW travel; MIT kp = 100
            # N.m/rad, kd = 2.0 (jaw-equivalent ~6.6 kN/m and ~130 N.s/m
            # per finger at 75.44 mm/rad).  An earlier 35 N close with two
            # independent drives (17.5 N/finger) was the rated maximum, not
            # the SDK's operational grasp: it squeezed a 0.5 N object at
            # 35 N and ejected it.  10 N/finger = the SDK's default 20 N
            # grasp (Litegrip软件开发手册: grasp(force_n=20), force_n x
            # 0.1 N.m); the rated 35 N (torque_limit_nm 3.5) is the real
            # ``close()``/``grasp(35)``.  The SDK/manual ``close()`` has no
            # force argument and uses torque_limit_nm = 3.5, i.e. 35 N
            # total; set_gripper(CLOSE) is that close.  With the single
            # driven joint and the mimic, the drive effort is calibrated
            # against measured contact: 17.5 N -> 17.6 N total and 35 N ->
            # 44 N total; 28 N measures 35.0 N total (17.5 N/finger), which
            # reproduces the real 35 N close (the rigid mimic shares the
            # load with the position loop).
            # Closing speed 50 mm/s of jaw travel (the SDK default); 100 mm/s
            # made the block slip out during the lift.
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper_slide_joint_right"],
                effort_limit_sim=28.0,
                velocity_limit_sim=0.025,
                stiffness=6600,
                damping=130,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
