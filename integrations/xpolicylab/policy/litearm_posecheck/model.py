"""A wiring/visual check policy for the dual LiteArm station.

It sends one bent pose through the regular evaluation path (so the 7-joint
action keys, the interpolation and the joint targets are exercised) and then
holds zero.  There is no model and no learning here.
"""

import numpy as np
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info

#: joint1..joint7 of the LiteArm, radians. joint4's upper limit is +0.035 rad.
BENT_POSE = [0.0, 0.5, 0.0, -1.0, 0.0, 0.4, 0.0]
#: how many get_action calls return the bent pose (2 actions each)
BENT_CALLS = 2


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]
        if self.action_type != "joint":
            raise ValueError(f"litearm_posecheck sends joint targets, got {self.action_type!r}")
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.num_arms = len(self.robot_action_dim_info["arm_dim"])
        self._calls = 0
        print(
            f"[posecheck] env_cfg={self.env_cfg_type} arms={self.num_arms} "
            f"arm_dim={self.robot_action_dim_info['arm_dim']}"
        )

    def _arm_keys(self):
        if self.num_arms == 1:
            return ["arm_joint_state"], ["ee_joint_state"]
        return (
            ["left_arm_joint_state", "right_arm_joint_state"],
            ["left_ee_joint_state", "right_ee_joint_state"],
        )

    def _action(self):
        arm_keys, ee_keys = self._arm_keys()
        bent = self._calls < BENT_CALLS
        self._calls += 1
        action = {}
        for i, (arm_key, ee_key) in enumerate(zip(arm_keys, ee_keys)):
            dim = self.robot_action_dim_info["arm_dim"][i]
            if bent:
                if dim != len(BENT_POSE):
                    raise ValueError(f"{arm_key} has {dim} joints, expected {len(BENT_POSE)}")
                action[arm_key] = np.asarray(BENT_POSE, dtype=np.float32)
            else:
                action[arm_key] = np.zeros(dim, dtype=np.float32)
            # 1.0 = open (the LiteArm robot_config maps the normalized action to
            # the LiteGrip stroke with sign -1)
            action[ee_key] = np.ones(self.robot_action_dim_info["ee_dim"][i], dtype=np.float32)
        return action

    def update_obs(self, obs):
        pass

    def update_obs_batch(self, obs_list):
        pass

    def get_action(self):
        return [self._action(), self._action()]

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("one environment")

    def reset(self):
        self._calls = 0
