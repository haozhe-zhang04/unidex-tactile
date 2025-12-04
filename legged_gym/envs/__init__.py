from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


from legged_gym.envs.b2.b2z1_pos_force_config import B2Z1PosForceRoughCfg, B2Z1PosForceRoughCfgPPO
from legged_gym.envs.wuji.wuji_pos_force_config import WujiPosForceRoughCfg, WujiPosForceRoughCfgPPO

from legged_gym.envs.wuji.wuji_robot_pos_force import WujiRobot_pos_force

from .base.legged_robot import LeggedRobot
from .b2.legged_robot_b2z1_pos_force import LeggedRobot_b2z1_pos_force

from legged_gym.utils.task_registry import task_registry

# register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
# task_registry.register( "b2z1_pos_force", LeggedRobot_b2z1_pos_force, B2Z1PosForceRoughCfg(), B2Z1PosForceRoughCfgPPO())
task_registry.register( "wuji_pos_force", WujiRobot_pos_force, WujiPosForceRoughCfg(), WujiPosForceRoughCfgPPO())