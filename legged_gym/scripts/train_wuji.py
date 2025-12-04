import os
unitree_rl_gym_path = os.path.abspath(__file__ + "../../../../")
import numpy as np
from datetime import datetime
import sys
sys.path.append(unitree_rl_gym_path)

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import time
from isaacgym import gymapi  # <-- 添加这一行
def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.flat_terrain:
        env_cfg.terrain.height = [0.0, 0.0]
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # visualise the initial observation/frame of the environment (no stepping)
    try:
        # ensure env is in its initial state
        env.reset()
    except Exception:
        pass

    env.render()
    #在这里添加绘制传感器的代码

    while True:
        env.gym.clear_lines(env.viewer)
        env._draw_sensor_axes()
        env._draw_sensor_force()
        # 当前时间步 t（可以是整数或 float）
        t = env.global_steps  # 例如用环境步数作为时间
        num_actions = env.num_actions

        # 生成每个 action 的正弦值
        # 可以给每个 action 一个不同的相位，使动作不完全同步
        phases = torch.linspace(0, 2*torch.pi, num_actions, device=env.device)
        amplitude = 0.3  # 控制动作幅度

        actions = amplitude * torch.sin(t * 0.1 + phases)
        # 给我一个正弦actions


        env.step(actions)
        
        env.gym.step_graphics(env.sim)
        env.gym.draw_viewer(env.viewer, env.sim, True)
        env.gym.sync_frame_time(env.sim)

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    args.headless = False
    train(args)
