from collections.abc import Sequence

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from env.environment.isaac.direct_rl_env import CustomDirectRLEnv


@configclass
class IsaacRLEnvCfg(DirectRLEnvCfg):
    decimation: int = 3
    episode_length_s: float = 2000 * (1 / 60) * decimation
    observation_space: int = 0
    action_space: int = 0
    state_space: int = 0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
    )
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(),
    )
    # GUI only: IsaacLab's default eye (7.5, 7.5, 7.5) sits outside RoboDojo's room
    # (10x10x3 m, scaled 0.5), so the viewport showed the room from outside.  Headless
    # evaluation never creates a viewport, so this cannot affect evaluation results.
    viewer: ViewerCfg = ViewerCfg(eye=(1.2, -1.3, 1.4), lookat=(0.0, -0.05, 0.75))


class IsaacRLEnv(CustomDirectRLEnv):
    cfg: IsaacRLEnvCfg
    env_spacing: float = 4.0

    def __init__(self, cfg: IsaacRLEnvCfg, render_mode: str | None = None, **kwargs):
        self.func = kwargs.get("func", None)
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        self.func["_setup_scene"](self)
        self.scene.clone_environments(copy_from_source=True)
        self.func["_post_setup_scene"](self)

    def _get_observations(self):
        pass

    def _get_dones(self):
        pass

    def _get_states(self):
        pass

    def _get_rewards(self):
        pass

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        super()._reset_idx(env_ids)
        self.func["_reset_idx"](self, env_ids)

    def _apply_action(self):
        pass

    def _pre_physics_step(self, actions, env_ids):
        pass
