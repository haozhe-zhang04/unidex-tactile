from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


from legged_gym.envs.b2.b2z1_pos_force_config import B2Z1PosForceRoughCfg, B2Z1PosForceRoughCfgPPO

from .base.legged_robot import LeggedRobot
from .b2.legged_robot_b2z1_pos_force import LeggedRobot_b2z1_pos_force

from legged_gym.utils.task_registry import task_registry


task_registry.register( "b2z1_pos_force", LeggedRobot_b2z1_pos_force, B2Z1PosForceRoughCfg(), B2Z1PosForceRoughCfgPPO())