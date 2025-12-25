from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from collections import deque
import torch
import torch.nn.functional as F

from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.terrain import Terrain, Terrain_Perlin
from legged_gym.utils.isaacgym_utils import euler_from_quat, sphere2cart, cart2sphere
from legged_gym.envs.wuji.wuji_pos_force_config import WujiRobotCfg

import matplotlib.pyplot as plt
from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin
import numpy as np


# 手指力控制索引
# finger1
INDEX_TIP_FORCE_X = 0
INDEX_TIP_FORCE_Y = 1
INDEX_TIP_FORCE_Z = 2
INDEX_TIP_TORQUE_X = 3
INDEX_TIP_TORQUE_Y = 4
INDEX_TIP_TORQUE_Z = 5
INDEX_TIP_POS_X_CMD = 6
INDEX_TIP_POS_Y_CMD = 7
INDEX_TIP_POS_Z_CMD = 8
INDEX_TIP_ORIENTATION_X_CMD = 9
INDEX_TIP_ORIENTATION_Y_CMD = 10
INDEX_TIP_ORIENTATION_Z_CMD = 11
INDEX_TIP_ORIENTATION_W_CMD = 12


def quat_to_rotation_matrix(quat):
    """
    将四元数转换为旋转矩阵 (GPU优化版本)
    
    Args:
        quat: tensor of shape (..., 4)  四元数 [x, y, z, w]
    Returns:
        R: tensor of shape (..., 3, 3)  旋转矩阵
    """
    # 使用 F.normalize 替代手动归一化
    # p=2 表示 2-范数（欧几里得长度），dim=-1 表示对最后一个维度（xyzw）做归一化
    quat = F.normalize(quat, p=2, dim=-1)
    
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # 计算旋转矩阵元素（避免重复计算）
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    
    # 构建旋转矩阵
    R = torch.stack([
        torch.stack([1 - 2*(yy + zz), 2*(xy - zw), 2*(xz + yw)], dim=-1),
        torch.stack([2*(xy + zw), 1 - 2*(xx + zz), 2*(yz - xw)], dim=-1),
        torch.stack([2*(xz - yw), 2*(yz + xw), 1 - 2*(xx + yy)], dim=-1),
    ], dim=-2)
    assert R.dim() == 4 and R.shape[1:] == (5, 3, 3)
    return R
    
def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

# def quat_conjugate(quat):
#     """
#     计算四元数的共轭
    
#     Args:
#         quat: tensor of shape (..., 4)  四元数 [x, y, z, w]
#     Returns:
#         quat_conj: tensor of shape (..., 4)  共轭四元数 [-x, -y, -z, w]
#     """
#     x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
#     return torch.stack([-x, -y, -z, w], dim=-1)

# def quat_mul(q1, q2):
#     """
#     批量四元数乘法 (Hamilton product)
    
#     Args:
#         q1: tensor of shape (..., 4)  四元数 [x, y, z, w]
#         q2: tensor of shape (..., 4)  四元数 [x, y, z, w]

#     Returns:
#         q: tensor of shape (..., 4)  四元数乘积 q = q1 * q2
#     """
#     assert q1.shape[-1] == 4 and q2.shape[-1] == 4, "输入最后一维必须是4"

#     x1, y1, z1, w1 = q1.unbind(-1)
#     x2, y2, z2, w2 = q2.unbind(-1)

#     x = w1*x2 + x1*w2 + y1*z2 - z1*y2
#     y = w1*y2 - x1*z2 + y1*w2 + z1*x2
#     z = w1*z2 + x1*y2 - y1*x2 + z1*w2
#     w = w1*w2 - x1*x2 - y1*y2 - z1*z2

#     return torch.stack([x, y, z, w], dim=-1)

def mat3x3_to_xyzw(R):
    """
    将旋转矩阵转换为四元数 (GPU优化版本)
    
    Args:
        R: tensor of shape (..., 3, 3)  旋转矩阵，支持批量处理
    Returns:
        quat: tensor of shape (..., 4)  四元数 [x, y, z, w]
    
    使用 Shepperd's method，避免数值不稳定，GPU友好
    """
    # 提取矩阵元素（避免重复索引）
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    
    # 计算迹
    trace = m00 + m11 + m22
    
    # 初始化四元数（自动继承设备、dtype）
    device = R.device
    dtype = R.dtype
    shape = R.shape[:-2]  # 除了最后两个维度 (3, 3)
    
    qx = torch.zeros(shape, device=device, dtype=dtype)
    qy = torch.zeros(shape, device=device, dtype=dtype)
    qz = torch.zeros(shape, device=device, dtype=dtype)
    qw = torch.zeros(shape, device=device, dtype=dtype)
    
    # Case 1: trace > 0 (最常见情况，优先处理)
    cond1 = trace > 0
    if cond1.any():
        s = torch.sqrt(trace[cond1] + 1.0) * 2.0  # s = 4 * qw
        qw[cond1] = s * 0.25
        qx[cond1] = (m21[cond1] - m12[cond1]) / s
        qy[cond1] = (m02[cond1] - m20[cond1]) / s
        qz[cond1] = (m10[cond1] - m01[cond1]) / s
    
    # Case 2: m00 最大
    cond2 = (~cond1) & (m00 >= m11) & (m00 >= m22)
    if cond2.any():
        s = torch.sqrt(1.0 + m00[cond2] - m11[cond2] - m22[cond2]) * 2.0
        qx[cond2] = s * 0.25
        qw[cond2] = (m21[cond2] - m12[cond2]) / s
        qy[cond2] = (m01[cond2] + m10[cond2]) / s
        qz[cond2] = (m02[cond2] + m20[cond2]) / s
    
    # Case 3: m11 最大
    cond3 = (~cond1) & (~cond2) & (m11 >= m22)
    if cond3.any():
        s = torch.sqrt(1.0 + m11[cond3] - m00[cond3] - m22[cond3]) * 2.0
        qy[cond3] = s * 0.25
        qw[cond3] = (m02[cond3] - m20[cond3]) / s
        qx[cond3] = (m01[cond3] + m10[cond3]) / s
        qz[cond3] = (m12[cond3] + m21[cond3]) / s
    
    # Case 4: m22 最大
    cond4 = (~cond1) & (~cond2) & (~cond3)
    if cond4.any():
        s = torch.sqrt(1.0 + m22[cond4] - m00[cond4] - m11[cond4]) * 2.0
        qz[cond4] = s * 0.25
        qw[cond4] = (m10[cond4] - m01[cond4]) / s
        qx[cond4] = (m02[cond4] + m20[cond4]) / s
        qy[cond4] = (m12[cond4] + m21[cond4]) / s
    
    # 堆叠并归一化
    quat = torch.stack([qx, qy, qz, qw], dim=-1)
    # 归一化（添加小量避免除零）
    norm = torch.norm(quat, dim=-1, keepdim=True) + 1e-8
    quat = quat / norm
    
    return quat

# 球形插值四元数
def slerp_xyzw(q0, q1, t, eps=1e-6):
    """
    SLERP for quaternions in xyzw format
    Args:
        q0: tensor (...,4) start quaternion in xyzw
        q1: tensor (...,4) end quaternion in xyzw
        t:  interpolation factor (...,1) or float in [0,1]
        eps: small number to avoid division by zero
    Returns:
        q_interp: tensor (...,4) interpolated quaternion in xyzw
    """
    # convert to wxyz for easier computation
    q0_wxyz = torch.stack([q0[...,3], q0[...,0], q0[...,1], q0[...,2]], dim=-1)
    q1_wxyz = torch.stack([q1[...,3], q1[...,0], q1[...,1], q1[...,2]], dim=-1)

    # normalize
    q0_wxyz = q0_wxyz / q0_wxyz.norm(dim=-1, keepdim=True)
    q1_wxyz = q1_wxyz / q1_wxyz.norm(dim=-1, keepdim=True)

    # dot product
    dot = (q0_wxyz * q1_wxyz).sum(dim=-1, keepdim=True)

    # if dot < 0, invert q1 to take the shortest path
    q1_wxyz = torch.where(dot < 0, -q1_wxyz, q1_wxyz)
    dot = torch.clamp(dot, -1.0, 1.0)  # clamp for numerical stability

    # compute theta
    theta_0 = torch.acos(dot)  # angle between q0 and q1
    sin_theta_0 = torch.sin(theta_0)

    # if angle is small, use lerp
    use_lerp = sin_theta_0 < eps
    # compute coefficients
    coeff0 = torch.where(use_lerp, 1.0 - t, torch.sin((1.0 - t) * theta_0) / sin_theta_0)
    coeff1 = torch.where(use_lerp, t, torch.sin(t * theta_0) / sin_theta_0)

    # interpolate
    q_interp_wxyz = coeff0 * q0_wxyz + coeff1 * q1_wxyz
    q_interp_wxyz = q_interp_wxyz / q_interp_wxyz.norm(dim=-1, keepdim=True)  # normalize

    # convert back to xyzw
    q_interp_xyzw = torch.stack([q_interp_wxyz[...,1], q_interp_wxyz[...,2], q_interp_wxyz[...,3], q_interp_wxyz[...,0]], dim=-1)

    return q_interp_xyzw

# 四元数转换为旋转矩阵前6维
def quat_to_mat6d(quat):

    assert quat.dim() == 3 and quat.shape[1:] == (5, 4)
    rot_matrices = quat_to_rotation_matrix(quat) # shape: (num_envs, num_fingers, 3, 3)
    
    # 2. 提取前两列
    # col1: X轴方向 (num_envs, num_fingers, 3)
    col1 = rot_matrices[:, :, :, 0] 
    # col2: Y轴方向 (num_envs, num_fingers, 3)
    col2 = rot_matrices[:, :, :, 1]
    
    # 3. 拼接成 6D 向量 (num_envs, num_fingers, 6)
    mat6d = torch.cat([col1, col2], dim=-1)
    assert mat6d.dim() == 3 and mat6d.shape[1:] == (5, 6)
    return mat6d

class WujiRobot_pos_force(BaseTask):
    def __init__(self, cfg: WujiRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """ 
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.debug_curve = False

        if self.debug_curve:
            pass

        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.num_pred_obs = self.cfg.env.num_pred_obs
        self.num_single_obs = self.cfg.env.num_single_obs
        self.obs_pred = torch.zeros(self.num_envs, self.num_pred_obs, device=self.device, dtype=torch.float)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        
    def step(self, actions):

        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        if not self.headless:
            self.render()
        
        # actions为 delta action
        for _ in range(self.cfg.control.decimation):
            if self.cfg.control.use_delta_action:
                self.dof_pos_target[:,:] = self.actions[:,:] * self.cfg.control.action_scale + self.dof_pos[:,:]

            else:
                # 绝对位置控制
                self.dof_pos_target[:,:] = self.actions[:,:] * self.cfg.control.action_scale + self.default_dof_pos[:].unsqueeze(0)
 
            # ⭐ 关键：裁剪到关节限制范围内
            self.dof_pos_target[:,:] = torch.clamp(
                self.dof_pos_target[:,:],
                self.dof_pos_limits[:,0],  # lower limits
                self.dof_pos_limits[:,1],   # upper limits
            )      

            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_pos_target))

            if self.global_steps > self.cfg.commands.force_start_step * 24:
                self._push_finger_tip(torch.arange(self.num_envs, device=self.device))  
            

            # DEBUG
            # 实时记录 self.forces[0,1,0:3] 的值
            if self.enable_realtime_plot and _ == 0:  # 只在第一个 decimation step 记录
                forces_xyz = self.forces[0, self.finger_tips_idx[2], 0:3].cpu().clone().numpy()
                self.forces_history.append(forces_xyz)
                self.step_history.append(self.global_steps)

            is_apply_ext = self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces[:,:,:3].contiguous()), None, gymapi.LOCAL_SPACE)
            
            if not is_apply_ext:
                raise Exception("Failed to apply external force to fingers")
            self.gym.simulate(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
        # for _ in range(self.cfg.control.decimation):
        #     self.torques = self._compute_torques(self.actions)
        #     self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        #     if self.cfg.env.test:
        #         elapsed_time = self.gym.get_elapsed_time(self.sim)
        #         sim_time = self.gym.get_sim_time(self.sim)
        #         if sim_time-elapsed_time>0:
        #             time.sleep(sim_time-elapsed_time)
            
        #     if self.device == 'cpu':
        #         self.gym.fetch_results(self.sim, True)

        #     if self.global_steps > self.cfg.commands.force_start_step * 24:
        #         # push gripper
                # self._push_gripper(torch.arange(self.num_envs, device=self.device))      
        #         # push robot base 
        #         # self._push_robot_base(torch.arange(self.num_envs, device=self.device)) 
        #     self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), None, gymapi.GLOBAL_SPACE)
        #     self.gym.simulate(self.sim)
        #     self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        
        # ⚠️ 诊断：检查 clip 前的值，如果经常被 clip 说明 normalization 有问题
        if self.global_steps % 500 == 0:
            obs_max_before_clip = self.obs_buf.abs().max().item()
            obs_mean_before_clip = self.obs_buf.abs().mean().item()
            if obs_max_before_clip > clip_obs * 0.9:  # 如果最大值接近 clip 阈值
                print(f"Warning: obs_buf values near clip limit. Max abs: {obs_max_before_clip:.4f}, Mean abs: {obs_mean_before_clip:.4f}, Clip: {clip_obs}")
        
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        self.obs_pred = torch.clip(self.obs_pred, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.global_steps += 1
        return {'obs': self.obs_buf, 'privileged_obs': self.privileged_obs_buf, 'obs_pred': self.obs_pred}, self.rew_buf, self.reset_buf, self.extras,self.reward_logs


    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1


        # self._post_physics_step_callback()


        # update ee goal
        self.update_curr_ee_goal()


        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_rigid_state[:] = self.rigid_state[:]


    
        # link_pos = self.rigid_state[:, self.finger_tips_idx,:3].clone()
        # link_q = self.rigid_state[:, self.finger_tips_idx,3:7].clone()
        # offset = self.sensors_pos_link[:, :3].view(1, 5, 3)   # (1,1,3)
        # offset = offset.expand_as(link_pos)     
        # # self.sensors_world[:, :, :3] = (link_pos + offset)       

        # q_rel = self.sensors_pos_link[:, 3:7].reshape(1,5, 4)  # (1,1,4)
        # self.sensors_world[:, :, 3:7] = quat_mul(link_q , q_rel)
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
        #     # self._draw_debug_vis()
        #     self.gym.clear_lines(self.viewer)
            self._draw_ee_goal_curr()
            self._draw_ee_force()
            # self._draw_ee()


    def _draw_ee(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.1, 4, 4, None, color=(1, 1, 0))
        # 直接从 rigid_state 读取最新位置
        finger_tip_pos_world = self.rigid_state[:, self.finger_tips_idx, :3].clone().squeeze(1)
        for i in range(self.num_envs):
            for idx in range(len(self.finger_tips_idx)):
                sphere_pose = gymapi.Transform(gymapi.Vec3(finger_tip_pos_world[i, idx, 0], finger_tip_pos_world[i, idx, 1], finger_tip_pos_world[i, idx, 2]), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        # self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # self.reset_buf |= torch.logical_or(torch.abs(self.base_euler_xyz[:,1])>1.0, torch.abs(self.base_euler_xyz[:,0])>0.8)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        # reset robot states
        self._reset_dofs(env_ids)
        # self._reset_root_states(env_ids)

        # self._resample_commands(env_ids)
        self._randomize_dof_props(env_ids)
        self._resample_ee_goal(env_ids, is_init=True)

        # reset buffers
        self.last_last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_rigid_state[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        # self.gait_indices[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.goal_timer[env_ids] = 0.

        # force control
        self.forces[env_ids[:, None], self.finger_tips_idx, :3] = 0.
        self.selected_env_ids_finger_tips_cmd[env_ids] = 0
        self.selected_env_ids_finger_tips_ext[env_ids] = 0
        self.push_end_time_finger_tips_cmd[env_ids] = 0.
        self.force_target_finger_tips_cmd[env_ids, :3] = 0.
        self.force_target_finger_tips_ext[env_ids, :3] = 0.
        self.push_duration_finger_tips_cmd[env_ids] = 0.
        self.current_Fxyz_finger_tips_cmd_local[env_ids, :3] = 0.


        self.commands[env_ids, :,INDEX_TIP_FORCE_X] = 0.0
        self.commands[env_ids, :,INDEX_TIP_FORCE_Y] = 0.0
        self.commands[env_ids, :,INDEX_TIP_FORCE_Z] = 0.0


        # Reset push gripper 
        if self.cfg.commands.push_finger_tips:
            self.forces[env_ids[:, None], self.finger_tips_idx, :3] = 0.
            self.selected_env_ids_finger_tips_cmd[env_ids] = 0
            self.selected_env_ids_finger_tips_ext[env_ids] = 0
            self.push_end_time_finger_tips_cmd[env_ids] = 0.


        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
            

        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0
        
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.reward_logs[name]=rew  
            self.episode_sums[name] += rew  # ⚠️ 关键修复：累加到 episode_sums

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
         # 添加诊断信息
        if self.global_steps % 100 == 0:
            print(f"Reward stats: mean={self.rew_buf.mean():.4f}, std={self.rew_buf.std():.4f}, "
                f"min={self.rew_buf.min():.4f}, max={self.rew_buf.max():.4f}")
        
    def get_body_orientation(self, return_yaw=False):
        r, p, y = euler_from_quat(self.base_quat)
        body_angles = torch.stack([r, p, y], dim=-1)

        if not return_yaw:
            return body_angles[:, :-1]
        else:
            return body_angles
        
    def compute_observations(self):
        """ Computes observations
        """
        num_fingers = len(self.finger_tips_idx)
        base_quat_reshaped = self.base_quat.unsqueeze(1).expand(self.num_envs, num_fingers, 4).reshape(self.num_envs*num_fingers,4)
        base_pos_reshaped = self.base_pos.unsqueeze(1).expand(self.num_envs, num_fingers, 3)
        # finger_tip_pos
        finger_tip_pos_world = self.rigid_state[:,self.finger_tips_idx, :3].clone() # shape: (num_envs, num_fingers, 3)
        finger_tip_pos_base = quat_rotate_inverse(base_quat_reshaped, (finger_tip_pos_world - base_pos_reshaped).reshape(-1,3)).reshape(self.num_envs,num_fingers,3)
        # finger_tip_orn
        finger_tip_orn_quat_world = self.rigid_state[:,self.finger_tips_idx, 3:7].clone()
        finger_tip_orn_quat_base = quat_mul(quat_conjugate(base_quat_reshaped), finger_tip_orn_quat_world.reshape(-1,4)).reshape(self.num_envs,num_fingers,4) # shape: (num_envs, num_fingers, 4)


        finger_tip_orn_6d_base = quat_to_mat6d(finger_tip_orn_quat_base)

        # # ⚠️ 关键修复：裁剪6D表示到合理范围，避免异常值
        # finger_tip_orn_6d_base = torch.clamp(finger_tip_orn_6d_base, min=-2.0, max=2.0)

        # finger_tip_vel
        finger_tip_vel_world = self.rigid_state[:,self.finger_tips_idx, 7:10].clone()
        finger_tip_vel_base = quat_rotate_inverse(base_quat_reshaped, finger_tip_vel_world.reshape(-1,3)).reshape(self.num_envs,num_fingers,3)

        # force+pos
        forces_local = self.forces[:, self.finger_tips_idx, :3].clone() # 手动施加的外力
        forces_base = self.transform_force_finger_tip_local_to_base(forces_local)
        forces_cmd_local = self.current_Fxyz_finger_tips_cmd_local
        forces_offset_local = (forces_local + forces_cmd_local)

        forces_offset_base = self.transform_force_finger_tip_local_to_base(forces_offset_local)

        # 计算考虑力偏移的目标位置 shape: (num_envs, num_fingers, 3)
        curr_finger_tip_goal_cart_base = forces_offset_base / (self.gripper_force_kps) + self.curr_finger_tip_goal_cart

        # error
        finger_tip_pos_error = finger_tip_pos_base - curr_finger_tip_goal_cart_base

        # ⚠️ 关键修复：确保目标姿态四元数归一化
        curr_finger_tip_goal_orn_normalized = self.curr_finger_tip_goal_orn / (torch.norm(self.curr_finger_tip_goal_orn, dim=-1, keepdim=True) + 1e-8)
        curr_finger_tip_goal_orn_6d_base = quat_to_mat6d(curr_finger_tip_goal_orn_normalized) # shape: (num_envs, num_fingers, 6)
        # # ⚠️ 关键修复：裁剪6D表示到合理范围
        # curr_finger_tip_goal_orn_6d_base = torch.clamp(curr_finger_tip_goal_orn_6d_base, min=-2.0, max=2.0)
        finger_tip_orn_6d_error = finger_tip_orn_6d_base - curr_finger_tip_goal_orn_6d_base

        # # sensor_forces
        # sensor_forces_world = quat_apply(self.rigid_state[:,self.finger_tips_idx,3:7].clone(),self.sensors_forces[:,:,:3].clone())
        # sensor_forces_base = quat_rotate_inverse(base_quat_reshaped,sensor_forces_world.reshape(-1,3)).reshape(self.num_envs,num_fingers,3)
        
        # 实时记录 self.sensors_forces[0,1,0:3] 的值
        if self.enable_realtime_plot:
            cmd_forces_xyz = self.current_Fxyz_finger_tips_cmd_local[0, 2, 0:3].cpu().clone().numpy()
            self.cmd_forces_history.append(cmd_forces_xyz)
            
            # 实时更新绘图（每10步更新一次，避免太频繁）
            if len(self.step_history) > 0 and len(self.step_history) % 10 == 0:
                self._update_realtime_plots()
        
        # commands 中的 F_cmd 已经是 base 坐标系，直接使用
        F_cmd_base = self.commands[:,:,INDEX_TIP_FORCE_X:(INDEX_TIP_FORCE_Z+1)]  # (num_envs, num_fingers, 3)
        commands = self.commands.clone()
        # 不需要再转换

        forces_error = forces_base + F_cmd_base

        # normalize pos
        finger_tip_pos_base_normalized = self._normalize_pos(finger_tip_pos_base) # shape: (num_envs, num_fingers, 3)
        curr_finger_tip_goal_cart_base_normalized = self._normalize_pos(curr_finger_tip_goal_cart_base) # (num_envs, num_fingers, 3)
        pos_cmd_normalized = self._normalize_pos(commands[:,:,INDEX_TIP_POS_X_CMD:(INDEX_TIP_POS_Z_CMD+1)]) # (num_envs, num_fingers*3)
        
        self.privileged_obs_buf = torch.cat((

                                self.dof_pos[:,:] * self.cfg.normalization.obs_scales.dof_pos,#20
                                self.dof_vel[:,:]* self.cfg.normalization.obs_scales.dof_vel,#20
                                # self.actions[:,:] * self.cfg.control.action_scale, # 20
                                # 食指位置 归一化
                                finger_tip_pos_base_normalized.reshape(self.num_envs,-1), # num_fingers*3 = 15

                                # 食指旋转 6d
                                finger_tip_orn_6d_base.reshape(self.num_envs,-1), # 5*6=30
                                # 食指速度
                                finger_tip_vel_base.reshape(self.num_envs,-1) * self.cfg.normalization.obs_scales.finger_tip_vel  ,# 5*3=15
              
                                # 目标位置
                                curr_finger_tip_goal_cart_base_normalized.reshape(self.num_envs,-1), # num_fingers*3 = 15
                                
                                # 位置误差
                                finger_tip_pos_error.reshape(self.num_envs,-1) * self.cfg.normalization.obs_scales.pose_error,# 5*3=15
                                # 旋转误差 6d
                                finger_tip_orn_6d_error.reshape(self.num_envs,-1) * self.cfg.normalization.obs_scales.orn_error,# 5*6=30
                                
                                # 力误差
                                forces_error.reshape(self.num_envs,-1) * self.cfg.normalization.obs_scales.force_error,# 5*3=15
                                
                                # 用外力模仿 传感器力
                                forces_base.reshape(self.num_envs,-1) * self.cfg.normalization.obs_scales.sensor_force,# 5*3=15
                                
                                # POS_CMD
                                pos_cmd_normalized.reshape(self.num_envs,-1), # num_fingers*3 = 15
                                # ORIENTATION_CMD
                                commands[:,:,INDEX_TIP_ORIENTATION_X_CMD:INDEX_TIP_ORIENTATION_W_CMD+1].reshape(self.num_envs,-1) * self.cfg.normalization.obs_scales.orientation_cmd, # 5*4=20
                                # FORCE_CMD 
                                commands[:,:,INDEX_TIP_FORCE_X:INDEX_TIP_FORCE_Z+1].reshape(self.num_envs,-1) * self.cfg.normalization.obs_scales.force_cmd,# 5*3=15
                            ),dim=-1) # 52
        obs_pred = torch.cat((
                                finger_tip_vel_base.reshape(self.num_envs,-1) * self.cfg.normalization.obs_scales.finger_tip_vel  ,# 5*3=15
                            ),dim=-1) #15

        obs_buf = torch.cat((   
                                # 关节角度
                                self.dof_pos[:,:] * self.cfg.normalization.obs_scales.dof_pos,#20
                                # # 上一次动作
                                # self.actions[:,:] * self.cfg.control.action_scale, # 20
                                # 食指位置
                                finger_tip_pos_base_normalized.reshape(self.num_envs,-1), # num_fingers*3 = 15
                                # 食指旋转 6d
                                finger_tip_orn_6d_base.reshape(self.num_envs,-1), #5*6=30
                                # 目标位置
                                curr_finger_tip_goal_cart_base_normalized.reshape(self.num_envs,-1), # num_fingers*3 = 15
                                # 位置误差
                                finger_tip_pos_error.reshape(self.num_envs,-1) * self.cfg.normalization.obs_scales.pose_error,# 5*3=15
                                # 旋转误差 6d
                                finger_tip_orn_6d_error.reshape(self.num_envs,-1) * self.cfg.normalization.obs_scales.orn_error,# 5*6=30
                                # 传感器力
                                forces_base.reshape(self.num_envs,-1) * self.cfg.normalization.obs_scales.sensor_force,# 5*3=15
                                # 力误差
                                forces_error.reshape(self.num_envs,-1) * self.cfg.normalization.obs_scales.force_error,# 5*3=15
                                # POS_CMD
                                pos_cmd_normalized.reshape(self.num_envs,-1), # num_fingers*3 = 15
                                # ORIENTATION_CMD
                                commands[:,:,INDEX_TIP_ORIENTATION_X_CMD:INDEX_TIP_ORIENTATION_W_CMD+1].reshape(self.num_envs,-1) * self.cfg.normalization.obs_scales.orientation_cmd, # 5*4=20
                                # FORCE_CMD
                                commands[:,:,INDEX_TIP_FORCE_X:INDEX_TIP_FORCE_Z+1].reshape(self.num_envs,-1) * self.cfg.normalization.obs_scales.force_cmd, # 5*3=15
                            ),dim=-1) #45

        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:  
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()

        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K
        self.obs_pred = obs_pred.clone()
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def get_observations(self):
        return {'obs': self.obs_buf, 'privileged_obs': self.privileged_obs_buf, 'obs_pred': self.obs_pred}
    
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.terrain = Terrain(self.cfg.terrain, )
        self._create_trimesh()
        # self._create_ground_plane()
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _draw_curve(self):
        if self.debug_curve:

            self.ax1.set_ylim(self.default_dof_pos_wo_gripper[0, self.FL_joint_index].item()-1.5, self.default_dof_pos_wo_gripper[0, self.FL_joint_index].item()+1.5)
            self.ax1.set_title(f"FL_joint_{self.FL_joint_index}")
            self.ax1.set_xlabel("time / s")
            self.ax1.legend(loc="upper right")
            self.ax2.set_ylim(self.default_dof_pos_wo_gripper[0, self.FR_joint_index].item()-1.5, self.default_dof_pos_wo_gripper[0, self.FR_joint_index].item()+1.5)
            self.ax2.set_title(f"FL_joint_{self.FR_joint_index}")
            self.ax2.set_xlabel("time / s")
            self.ax2.legend(loc="upper right")

            self.ax3.set_ylim(self.default_dof_pos_wo_gripper[0, self.FL_joint_index].item()-1.5, self.default_dof_pos_wo_gripper[0, self.FL_joint_index].item()+1.5)
            self.ax3.set_title(f"front_ref_motion")
            self.ax3.set_xlabel("time / s")
            self.ax3.legend(loc="upper right")

            self.ax4.set_ylim(self.default_dof_pos_wo_gripper[0, self.RL_joint_index].item()-1.5, self.default_dof_pos_wo_gripper[0, self.RL_joint_index].item()+1.5)
            self.ax4.set_title(f"FL_joint_{self.RL_joint_index}")
            self.ax4.set_xlabel("time / s")
            self.ax4.legend(loc="upper right")
            self.ax5.set_ylim(self.default_dof_pos_wo_gripper[0, self.RR_joint_index].item()-1.5, self.default_dof_pos_wo_gripper[0, self.RR_joint_index].item()+1.5)
            self.ax5.set_title(f"FL_joint_{self.RR_joint_index}")
            self.ax5.set_xlabel("time / s")
            self.ax5.legend(loc="upper right")

            self.ax6.set_ylim(-.5, 1.5)
            self.ax6.set_title(f"ront_ref_motion")
            self.ax6.set_xlabel("time / s")
            self.ax6.legend(loc="upper right")

            self.t_data.append(self.global_steps)
            self.FL_action.append(self.actions[0, self.FL_joint_index].item() * self.cfg.control.action_scale + self.default_dof_pos_wo_gripper[0, self.FR_joint_index].item())
            self.FL_state.append(self.dof_pos[0, self.FL_joint_index].item())
            self.FL_ref.append(self.ref_dof_pos[0, self.FL_joint_index].item())
            self.FR_action.append(self.actions[0, self.FR_joint_index].item() * self.cfg.control.action_scale + self.default_dof_pos_wo_gripper[0, self.FR_joint_index].item())
            self.FR_state.append(self.dof_pos[0, self.FR_joint_index].item())
            self.FR_ref.append(self.ref_dof_pos[0, self.FR_joint_index].item())

            # stand_mask = self._get_gait_phase()
            self.RL_action.append(self.actions[0, self.RL_joint_index].item() * self.cfg.control.action_scale + self.default_dof_pos_wo_gripper[0, self.RR_joint_index].item())
            self.RL_state.append(self.dof_pos[0, self.RL_joint_index].item())
            self.RL_ref.append(self.ref_dof_pos[0, self.RL_joint_index].item())
            self.RR_action.append(self.actions[0, self.RR_joint_index].item() * self.cfg.control.action_scale + self.default_dof_pos_wo_gripper[0, self.RR_joint_index].item())
            self.RR_state.append(self.dof_pos[0, self.RR_joint_index].item())
            self.RR_ref.append(self.ref_dof_pos[0, self.RR_joint_index].item())
            
            if len(self.t_data) > 100:
                self.t_data.pop(0)
                self.FL_action.pop(0)
                self.FL_state.pop(0)
                self.FL_ref.pop(0)
                self.FR_action.pop(0)
                self.FR_state.pop(0)
                self.FR_ref.pop(0)
                self.RL_action.pop(0)
                self.RL_state.pop(0)
                self.RL_ref.pop(0)
                self.RR_action.pop(0)
                self.RR_state.pop(0)
                self.RR_ref.pop(0)
            self.FL_action_line.set_data(self.t_data, self.FL_action)
            self.FL_state_line.set_data(self.t_data, self.FL_state)
            self.FL_ref_line.set_data(self.t_data, self.FL_ref)
            self.FR_action_line.set_data(self.t_data, self.FR_action)
            self.FR_state_line.set_data(self.t_data, self.FR_state)
            self.FR_ref_line.set_data(self.t_data, self.FR_ref)

            self.RL_action_line.set_data(self.t_data, self.RL_action)
            self.RL_state_line.set_data(self.t_data, self.RL_state)
            self.RL_ref_line.set_data(self.t_data, self.RL_ref)
            self.RR_action_line.set_data(self.t_data, self.RR_action)
            self.RR_state_line.set_data(self.t_data, self.RR_state)
            self.RR_ref_line.set_data(self.t_data, self.RR_ref)
            self.ax1.set_xlim(max(0, self.global_steps - 100), self.global_steps + 1)
            self.ax2.set_xlim(max(0, self.global_steps - 100), self.global_steps + 1)
            self.ax3.set_xlim(max(0, self.global_steps - 100), self.global_steps + 1)
            self.ax4.set_xlim(max(0, self.global_steps - 100), self.global_steps + 1)
            self.ax5.set_xlim(max(0, self.global_steps - 100), self.global_steps + 1)
            self.ax6.set_xlim(max(0, self.global_steps - 100), self.global_steps + 1)
            plt.pause(0.001)
            # print(self.commands)
    def _draw_collision_bbox(self):

        center = self.ee_goal_center_offset
        bbox0 = center + self.collision_upper_limits
        bbox1 = center + self.collision_lower_limits
        bboxes = torch.stack([bbox0, bbox1], dim=1)
        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0))

        for i in range(self.num_envs):
            bbox_geom = gymutil.WireframeBBoxGeometry(bboxes[i], None, color=(1, 0, 0))
            quat = self.base_yaw_quat[i]
            r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
            pose0 = gymapi.Transform(gymapi.Vec3(self.root_states[i, 0], self.root_states[i, 1], 0), r=r)
            gymutil.draw_lines(bbox_geom, self.gym, self.viewer, self.envs[i], pose=pose0) 

           
    def _draw_ee_goal_curr(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.001, 4, 4, None, color=(1, 1, 0)) # 黄

        sphere_geom_4 = gymutil.WireframeSphereGeometry(0.001, 4, 4, None, color=(1, 0, 1)) # 紫

        sphere_geom_origin = gymutil.WireframeSphereGeometry(0.001, 8, 8, None, color=(0, 1, 0)) # 绿
        sphere_pose = gymapi.Transform(gymapi.Vec3(0, 0, 0), r=None)
        gymutil.draw_lines(sphere_geom_origin, self.gym, self.viewer, self.envs[0], sphere_pose)

        axes_geom = gymutil.AxesGeometry(scale=0.2)

        # forces_local = self.sensors_forces[:, :, :3]
        forces_local = self.forces[:,self.finger_tips_idx,:3]
        forces_cmd_local = self.current_Fxyz_finger_tips_cmd_local

        forces_offset_local = (forces_local + forces_cmd_local)

        forces_offset_base = self.transform_force_finger_tip_local_to_base(forces_offset_local)

        # 计算考虑力偏移的目标位置 shape: (num_envs, num_fingers, 3)
        curr_finger_tip_goal_cart_base = forces_offset_base / (self.gripper_force_kps) + self.curr_finger_tip_goal_cart
        curr_ee_goal_cart_world_offset = self.transform_pos_base_to_world(curr_finger_tip_goal_cart_base)
        curr_finger_tip_goal_cart_global = self.transform_pos_base_to_world(self.curr_finger_tip_goal_cart)

        base_quat_reshaped = self.base_quat.unsqueeze(1).expand(self.num_envs,len(self.finger_tips_idx), 4) #(num_envs, len(finger_tips_idx), 4)

        # 将目标姿态从base坐标系转换到世界坐标系
        # q_world = q_base_world * q_base
        curr_finger_tip_goal_quat_base = self.curr_finger_tip_goal_orn
        curr_finger_tip_goal_quat_world = quat_mul(base_quat_reshaped, curr_finger_tip_goal_quat_base)

        for i in range(self.num_envs):
            for finger_idx in range(len(self.finger_tips_idx)):
                # 当前的目标位置（x_cmd, y_cmd, z_cmd） 黄
                sphere_pose = gymapi.Transform(gymapi.Vec3(curr_finger_tip_goal_cart_global[i,finger_idx, 0], curr_finger_tip_goal_cart_global[i,finger_idx, 1], curr_finger_tip_goal_cart_global[i,finger_idx, 2]), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

                # x_target, y_target, z_target 紫
                sphere_pose_4 = gymapi.Transform(gymapi.Vec3(curr_ee_goal_cart_world_offset[i,finger_idx, 0].item(), curr_ee_goal_cart_world_offset[i,finger_idx, 1].item(), curr_ee_goal_cart_world_offset[i,finger_idx, 2].item()), r=None)
                gymutil.draw_lines(sphere_geom_4, self.gym, self.viewer, self.envs[i], sphere_pose_4) #紫
                
                # sphere_pose_2 = gymapi.Transform(gymapi.Vec3(ee_pose[i,0, 0].item(), ee_pose[i,0, 1].item(), ee_pose[i,0, 2].item()), r=None)
                # gymutil.draw_lines(sphere_geom_2, self.gym, self.viewer, self.envs[i], sphere_pose_2) #蓝

                # sphere_pose_3 = gymapi.Transform(gymapi.Vec3(upper_arm_pose[i, 0], upper_arm_pose[i, 1], upper_arm_pose[i, 2]), r=None)
                # gymutil.draw_lines(sphere_geom_3, self.gym, self.viewer, self.envs[i], sphere_pose_3) 

                # 绘制目标姿态的朝向（使用世界坐标系下的四元数）
                # gymapi.Quat的参数顺序是 (x, y, z, w)
                goal_quat_world = curr_finger_tip_goal_quat_world[i,finger_idx]
                goal_quat_gymapi = gymapi.Quat(goal_quat_world[0].item(), goal_quat_world[1].item(), 
                                            goal_quat_world[2].item(), goal_quat_world[3].item())
                pose = gymapi.Transform(
                    gymapi.Vec3(curr_finger_tip_goal_cart_global[i,finger_idx, 0], 
                            curr_finger_tip_goal_cart_global[i,finger_idx, 1], 
                            curr_finger_tip_goal_cart_global[i,finger_idx, 2]), 
                    r=goal_quat_gymapi
                )
                gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose)


    def _draw_ee_force(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        sphere_geom_arrow_1 = gymutil.WireframeSphereGeometry(0.001, 16, 16, None, color=(0, 0, 1))
        arrow_color_1 = [0, 0, 1]
        sphere_geom_arrow_2 = gymutil.WireframeSphereGeometry(0.001, 16, 16, None, color=(0, 1, 0))
        arrow_color_2 = [0, 1, 0]
        
        # commands 中的 F_cmd 是 base 坐标系，转换到 world 用于可视化
        F_cmd_base = self.commands[:,:,INDEX_TIP_FORCE_X:INDEX_TIP_FORCE_Z+1].clone()
        F_cmd_gloal = self.transform_force_base_to_world(F_cmd_base)
        F_cmd_norm = torch.norm(F_cmd_gloal,dim=-1,keepdim=True)

        ext_forces = self.forces[:,self.finger_tips_idx,:3].clone()
        finger_tip_quat = self.rigid_state[:, self.finger_tips_idx, 3:7]
        
        # finger_tip local -> world
        ext_forces_global = quat_apply(
            finger_tip_quat,
            ext_forces.reshape(-1, 3)
        ).reshape(self.num_envs, len(self.finger_tips_idx), 3)
        ext_forces_norm = torch.norm(ext_forces,dim=-1,keepdim=True)

        ee_pose = self.rigid_state[:, self.finger_tips_idx, :3]

        for i in range(self.num_envs):
            for finger_idx in range(len(self.finger_tips_idx)):
                # F_cmd 蓝色
                start_pos = ee_pose[i,finger_idx].cpu().numpy()
                arrow_direction = F_cmd_gloal[i,finger_idx].cpu().numpy()
                arrow_length = F_cmd_norm[i,finger_idx].item() * 5
                end_pos = start_pos + arrow_direction * arrow_length

                verts = [start_pos, end_pos]
                colors = [arrow_color_1, arrow_color_1]
                self.gym.add_lines(self.viewer, self.envs[i], len(verts), verts, colors)
                head_pos = end_pos
                head_pose = gymapi.Transform(gymapi.Vec3(head_pos[0], head_pos[1], head_pos[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow_1, self.gym, self.viewer, self.envs[i], head_pose)
                
                # ext_forces 绿色
                start_pos = ee_pose[i,finger_idx].cpu().numpy()
                arrow_direction = ext_forces_global[i,finger_idx].cpu().numpy()
                arrow_length = ext_forces_norm[i,finger_idx].item() * 5
                end_pos = start_pos + arrow_direction * arrow_length
                verts = [start_pos, end_pos]
                colors = [arrow_color_2, arrow_color_2]
                self.gym.add_lines(self.viewer, self.envs[i], len(verts), verts, colors)
                head_pos = end_pos
                head_pose = gymapi.Transform(gymapi.Vec3(head_pos[0], head_pos[1], head_pos[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow_2, self.gym, self.viewer, self.envs[i], head_pose)
    
    def _draw_base_force(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        sphere_geom_arrow_1 = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 0, 1))
        arrow_color_1 = [0, 0, 1]
        sphere_geom_arrow_2 = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0))
        arrow_color_2 = [0, 1, 0]
        base_pose = self.rigid_state[:, self.robot_base_idx, :3]
        forces_global = self.forces[:, self.robot_base_idx, 0:3] / 100
        forces_global_norm = torch.norm(forces_global, dim=-1, keepdim=True)
        target_forces_global = forces_global / (forces_global_norm + 1e-5)
        forces_cmd = self.current_Fxyz_base_cmd / 100
        forces_cmd_global = quat_apply(self.base_yaw_quat, forces_cmd)
        forces_cmd_norm = torch.norm(forces_cmd_global, dim=-1, keepdim=True)
        target_forces_cmd = forces_cmd_global / (forces_cmd_norm + 1e-5)
        for i in range(self.num_envs):

            start_pos = base_pose[i].cpu().numpy() + [0, 0, 0.5]
            arrow_direction = target_forces_global[i].cpu().numpy()
            arrow_length = forces_global_norm[i].item()
            end_pos = start_pos + arrow_direction * arrow_length
            verts = [start_pos, end_pos]
            colors = [arrow_color_1, arrow_color_1]
            self.gym.add_lines(self.viewer, self.envs[i], len(verts), verts, colors)
            head_pos = end_pos
            head_pose = gymapi.Transform(gymapi.Vec3(head_pos[0], head_pos[1], head_pos[2]), r=None)
            gymutil.draw_lines(sphere_geom_arrow_1, self.gym, self.viewer, self.envs[i], head_pose)

            start_pos = base_pose[i].cpu().numpy() + [0, 0, 0.5]
            arrow_direction = target_forces_cmd[i].cpu().numpy()
            arrow_length = forces_cmd_norm[i].item()
            end_pos = start_pos + arrow_direction * arrow_length
            verts = [start_pos, end_pos]
            colors = [arrow_color_2, arrow_color_2]
            self.gym.add_lines(self.viewer, self.envs[i], len(verts), verts, colors)
            head_pos = end_pos
            head_pose = gymapi.Transform(gymapi.Vec3(head_pos[0], head_pos[1], head_pos[2]), r=None)
            gymutil.draw_lines(sphere_geom_arrow_2, self.gym, self.viewer, self.envs[i], head_pose)
            
    def _draw_ee_goal_traj(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.005, 8, 8, None, color=(1, 0, 0))
        sphere_geom_yellow = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(1, 1, 0))

        t = torch.linspace(0, 1, 10, device=self.device)[None, None, None, :]
        ee_target_all_sphere = torch.lerp(self.ee_start_sphere[..., None], self.ee_goal_sphere[..., None], t).squeeze(0)
        ee_target_all_cart_world = torch.zeros_like(ee_target_all_sphere)
        for i in range(10):
            ee_target_cart = sphere2cart(ee_target_all_sphere[..., i])
            ee_target_all_cart_world[..., i] = quat_apply(self.base_yaw_quat, ee_target_cart)
        ee_target_all_cart_world += self.get_ee_goal_spherical_center()[:, :, None]
        for i in range(self.num_envs):
            for j in range(10):
                pose = gymapi.Transform(gymapi.Vec3(ee_target_all_cart_world[i, 0, j], ee_target_all_cart_world[i, 1, j], ee_target_all_cart_world[i, 2, j]), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)
    
    def _draw_sensor_axes(self):
        """
        绘制食指指尖传感器的坐标系（X红、Y绿、Z蓝）。
        """

        if not self.viewer or not self.debug_viz or self.finger_tips_idx == -1:
            return

        axis_len = 0.1  # 坐标轴长度

        for i in range(self.num_envs):
            # 获取传感器世界坐标系状态
            pos = self.sensors_world[i, 0, :3].cpu().numpy()     # (3,)
            quat = self.sensors_world[i, 0, 3:].cpu().numpy()    # (4,)

            # 使用 gymapi.Transform 将四元数和位置转换
            t = gymapi.Transform(
                p=gymapi.Vec3(*pos),
                r=gymapi.Quat(*quat)
            )

            # 世界坐标系中的三个轴向量
            x_axis = t.r.rotate(gymapi.Vec3(axis_len, 0, 0))
            y_axis = t.r.rotate(gymapi.Vec3(0, axis_len, 0))
            z_axis = t.r.rotate(gymapi.Vec3(0, 0, axis_len))

            # 每条线需要两个顶点：起点和终点
            verts = np.array([
                [t.p.x, t.p.y, t.p.z], [t.p.x + x_axis.x, t.p.y + x_axis.y, t.p.z + x_axis.z],
                [t.p.x, t.p.y, t.p.z], [t.p.x + y_axis.x, t.p.y + y_axis.y, t.p.z + y_axis.z],
                [t.p.x, t.p.y, t.p.z], [t.p.x + z_axis.x, t.p.y + z_axis.y, t.p.z + z_axis.z],
            ], dtype=np.float32)

            # 对应的颜色
            colors = np.array([
                [1, 0, 0], [1, 0, 0],  # X 红
                [0, 1, 0], [0, 1, 0],  # Y 绿
                [0, 0, 0], [0, 0, 0],  # Z 蓝
            ], dtype=np.float32)

            # 绘制 3 条线段
            self.gym.add_lines(
                self.viewer,
                self.envs[i],
                3,        # line 数量
                verts,
                colors
            )

    def _draw_sensor_force(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        sphere_geom_arrow_1 = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 0, 1))
        arrow_color_1 = [0, 0, 1]

        sensor_pose = self.sensors_world[:,0,:3]
        sensor_force = self.sensors_forces[:,0,:3]

        sensors_force_global = torch.norm(sensor_force, dim=-1, keepdim=True)

        # ee_pose = self.rigid_state[:, self.gripper_idx, :3]
        # forces_global = self.forces[:, self.gripper_idx, 0:3] / 100
        # forces_global_norm = torch.norm(forces_global, dim=-1, keepdim=True)
        # target_forces_global = forces_global / (forces_global_norm + 1e-5)
        # forces_cmd = self.current_Fxyz_gripper_cmd / 100
        # forces_cmd_global = quat_apply(self.base_yaw_quat, forces_cmd)
        # forces_cmd_norm = torch.norm(forces_cmd_global, dim=-1, keepdim=True)
        # target_forces_cmd = forces_cmd_global / (forces_cmd_norm + 1e-5)
        for i in range(self.num_envs):

            start_pos = sensor_pose[i].cpu().numpy()
            arrow_direction = sensor_force[i].cpu().numpy()
            arrow_length = sensors_force_global[i].item()
            end_pos = start_pos + arrow_direction * arrow_length
            verts = [start_pos, end_pos]
            colors = [arrow_color_1, arrow_color_1]
            self.gym.add_lines(self.viewer, self.envs[i], len(verts), verts, colors)
            head_pos = end_pos
            head_pose = gymapi.Transform(gymapi.Vec3(head_pos[0], head_pos[1], head_pos[2]), r=None)
            gymutil.draw_lines(sphere_geom_arrow_1, self.gym, self.viewer, self.envs[i], head_pose)
        
    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 256
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

            self.env_frictions[env_id] = self.friction_coeffs[env_id]
        else:
            self.friction_coeffs = torch.ones([self.num_envs, 1, 1])
        return props
    
   

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits - 使用配置的soft_dof_pos_limit缩小关节限制范围
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
             # ⭐ 关键修复：为灵巧手关节（索引4:8）设置位置控制模式和PD参数
            for i in range(len(props)):
                props["driveMode"][i] = gymapi.DOF_MODE_POS  # 位置控制模式
                # 从配置中获取 stiffness 和 damping
                name = self.dof_names[i]
                found = False
                for dof_name in self.cfg.control.stiffness.keys():
                    if dof_name in name:
                        props["stiffness"][i] = self.cfg.control.stiffness[dof_name]
                        props["damping"][i] = self.cfg.control.damping[dof_name]
                        found = True
                        print("found!")
                        break
                if not found:
                    # 使用默认值
                    props["stiffness"][i] = 120
                    props["damping"][i] = 0.5
        return props

    def _randomize_dof_props(self, env_ids):
        if self.cfg.commands.randomize_gripper_force_gains:
            min_kp, max_kp = self.cfg.commands.gripper_force_kp_range
            # min_kd, max_kd = self.cfg.commands.gripper_force_kd_range
            self.gripper_force_kps[env_ids,:,:] = torch.rand((len(env_ids), len(self.finger_tips_idx),3), dtype=torch.float, device=self.device,
                                                     requires_grad=False)* (max_kp - min_kp) + min_kp
            # if self.cfg.commands.gripper_prop_kd > 0:
            #     self.gripper_force_kds[env_ids, :] = self.gripper_force_kps[env_ids, :] * self.cfg.commands.gripper_prop_kd
            # else:
            #     self.gripper_force_kds[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
            #                                             requires_grad=False).unsqueeze(1) * (
            #                                         max_kd - min_kd) + min_kd
                
    def _process_rigid_body_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng[0], rng[1], size=(1, ))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros(1)

        if self.cfg.domain_rand.randomize_leg_mass:
            rng_link = self.cfg.domain_rand.leg_mass_scale_range
            rand_leg_masses = []
            for idx in range(17):
                rand_link_mass = np.random.uniform(rng_link[0], rng_link[1], size=(1, )) * props[idx].mass
                props[idx].mass += rand_link_mass
                rand_leg_masses.append(rand_link_mass)
            rand_leg_masses = np.concatenate(rand_leg_masses)
        else:
            rand_leg_masses = np.zeros(17)

        if self.cfg.domain_rand.randomize_gripper_mass:
            gripper_rng_mass = self.cfg.domain_rand.gripper_added_mass_range
            gripper_rand_mass = np.random.uniform(gripper_rng_mass[0], gripper_rng_mass[1], size=(1, ))
            props[self.gripper_idx].mass += gripper_rand_mass
        else:
            gripper_rand_mass = np.zeros(1)

        if self.cfg.domain_rand.randomize_base_com:
            rng_com_x = self.cfg.domain_rand.added_com_range_x
            rng_com_y = self.cfg.domain_rand.added_com_range_y
            rng_com_z = self.cfg.domain_rand.added_com_range_z
            rand_com = np.random.uniform([rng_com_x[0], rng_com_y[0], rng_com_z[0]], [rng_com_x[1], rng_com_y[1], rng_com_z[1]], size=(3, ))
            props[1].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        
        mass_params = np.concatenate([rand_mass, rand_com, gripper_rand_mass, rand_leg_masses])
        return props, mass_params
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._randomize_dof_props(env_ids)
        self._step_contact_targets()

    def _step_contact_targets(self):
        cycle_time = self.cfg.rewards.cycle_time
        standing_mask = ~self.get_walking_cmd_mask()
        self.gait_indices = torch.remainder(self.gait_indices + self.dt / cycle_time, 1.0)
        self.gait_indices[standing_mask] = 0

    def _update_realtime_plots(self):
        """
        实时更新三张图（x, y, z 三个分量）
        分别显示 forces[0,1,0:3] 和 sensors_forces[0,1,0:3]
        """
        if not self.enable_realtime_plot or self.headless or len(self.step_history) == 0:
            return
        
        try:
            # 转换为 numpy 数组
            steps = np.array(self.step_history)
            forces_data = np.array(self.forces_history)  # shape: (N, 3)
            cmd_forces_data = np.array(self.cmd_forces_history)  # shape: (N, 3)
            
            # 确保数据长度一致
            min_len = min(len(steps), len(forces_data), len(cmd_forces_data))
            steps = steps[-min_len:]
            forces_data = forces_data[-min_len:]
            cmd_forces_data = cmd_forces_data[-min_len:]
            
            # 更新三张图（x, y, z）
            for axis_idx, axis_name in enumerate(['X', 'Y', 'Z']):
                ax = self.axes[axis_idx]
                ax.clear()
                ax.set_title(f'{axis_name} Component')
                ax.set_ylabel('Force (N)')
                if axis_idx == 2:
                    ax.set_xlabel('Step')
                
                # 绘制 forces
                ax.plot(steps, forces_data[:, axis_idx], 'b-', label='Applied Forces', linewidth=1.5)
                
                # 绘制 sensor_forces
                ax.plot(steps, cmd_forces_data[:, axis_idx], 'r-', label='Cmd Forces', linewidth=1.5)
                
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
            
            # 更新图像
            plt.draw()
            plt.pause(0.001)  # 短暂暂停以更新图像
            
        except Exception as e:
            # 如果绘图出错，不影响主程序运行
            if self.global_steps % 1000 == 0:
                print(f"Warning: Error updating plots: {e}")

    # 归一化pos，方便observation
    def _normalize_pos(self, pos):
        assert pos.shape ==(self.num_envs, len(self.finger_tips_idx), 3)
        normalized_pos = torch.zeros_like(pos)
        for i in range(len(self.finger_tips_idx)):
            normalized_pos[:,i,0:1] = 2 * (pos[:,i,0:1] - self.cfg.normalization.obs_scales.finger_tip_pos_x_min[i]) / (self.cfg.normalization.obs_scales.finger_tip_pos_x_max[i] - self.cfg.normalization.obs_scales.finger_tip_pos_x_min[i]) - 1
            normalized_pos[:,i,1:2] = 2 * (pos[:,i,1:2] - self.cfg.normalization.obs_scales.finger_tip_pos_y_min[i]) / (self.cfg.normalization.obs_scales.finger_tip_pos_y_max[i] - self.cfg.normalization.obs_scales.finger_tip_pos_y_min[i]) - 1
            normalized_pos[:,i,2:3] = 2 * (pos[:,i,2:3] - self.cfg.normalization.obs_scales.finger_tip_pos_z_min[i]) / (self.cfg.normalization.obs_scales.finger_tip_pos_z_max[i] - self.cfg.normalization.obs_scales.finger_tip_pos_z_min[i]) - 1
        return normalized_pos

    # def _resample_commands(self, env_ids):
    #     """ Randommly select commands of some environments

    #     Args:
    #         env_ids (List[int]): Environments ids for which new commands are needed
    #     """
    #     if self.cfg.env.teleop_mode:
    #         return
    #     self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
    #     zero_cmd_mask = torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False) < self.cfg.commands.zero_vel_cmd_prob
    #     self.commands[env_ids, :3] *= ~zero_cmd_mask.unsqueeze(1)

    #     # set small commands to zero
    #     non_stop_sign = (torch.abs(self.commands[env_ids, 0]) > self.cfg.commands.lin_vel_x_clip) | (torch.abs(self.commands[env_ids, 1]) > self.cfg.commands.lin_vel_y_clip) | (torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_yaw_clip)
    #     self.commands[env_ids, :3] *= non_stop_sign.unsqueeze(1)

    def control_ik(self, dpose):
        # solve damped least squares
        j_eef_T = torch.transpose(self.ee_j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (0.05 ** 2)
        A = torch.bmm(self.ee_j_eef, j_eef_T) + lmbda[None, ...]
        u = torch.bmm(j_eef_T, torch.linalg.solve(A, dpose))#.view(self.num_envs, 6)
        return u.squeeze(-1)

    def _resample_ee_goal_cart_once(self, env_ids, max_retries=50, min_finger_distance=0.02):
        """
        随机采样关节角度，并确保手指不会发生自碰撞
        
        Args:
            env_ids: 需要采样的环境ID列表（tensor或list）
            max_retries: 每个环境的最大重试次数，默认50次
            min_finger_distance: 手指之间的最小安全距离（米），默认0.02m
        """
        if len(env_ids) == 0:
            return
        
        # 确保 env_ids 是 tensor
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(self.device)
        
        # 使用当前关节位置作为基础
        last_joint = self.dof_pos.clone()  # shape: (num_envs, num_dofs)
        # 初始化新的关节角度（基于当前关节位置）
        rand_joint = last_joint.clone()
        low = self.dof_pos_limits[:, 0]  # shape: (num_dofs,)
        high = self.dof_pos_limits[:, 1]  # shape: (num_dofs,)
        
        # 最大 delta 变化量 (随着训练不断增大，最大到0.5，非线性增长，且越增越快)
        # 用指数型增长，早期增长慢、后期增长快
        progress = torch.min(torch.tensor(self.global_steps) / 10000.0, torch.tensor(1.0))
        max_delta = 0.1 + 0.4 * (1 - torch.exp(-5 * progress))  # 指数型：增长初期缓慢，后期加速，最大0.5
        
        # 为每个环境单独采样，直到找到无碰撞的配置
        remaining_env_ids = env_ids.clone()
        
        for retry in range(max_retries):
            if len(remaining_env_ids) == 0:
                break
            
            # 为剩余的环境生成 delta joint pos（每次改变不超过 0.2）
            # 生成 [-max_delta, max_delta] 范围内的随机增量
            delta_joint = torch_rand_float(
                -max_delta, max_delta, 
                (len(remaining_env_ids), self.num_dofs), 
                device=self.device
            )
            
            # 这样写不正确，torch.max(0.1, delta_joint) 会报错，应该用下面的方式将 delta_joint 逐元素和 0.1 比较
            delta_joint = torch.clamp(delta_joint, min=0.1)

            # 基于当前关节位置加上 delta
            new_joint = last_joint[remaining_env_ids, :] + delta_joint
            
            # 确保新的关节角度在限制范围内
            new_joint = torch.clamp(new_joint, low.unsqueeze(0), high.unsqueeze(0))
            
            # 更新 rand_joint
            rand_joint[remaining_env_ids, :] = new_joint
            
            # 计算正向运动学
            self._pinocchio_forward_kinematics(rand_joint, remaining_env_ids)
            
            # 检查碰撞
            collision_mask = self._check_finger_self_collision(remaining_env_ids, min_distance=min_finger_distance)
            
            # 找出仍然有碰撞的环境
            colliding_env_ids = remaining_env_ids[~collision_mask]
            
            # 找出无碰撞的环境（成功采样的环境）
            success_env_ids = remaining_env_ids[collision_mask]
            
            # 如果所有环境都无碰撞，退出循环
            if len(colliding_env_ids) == 0:
                break
            
            # 如果达到最大重试次数，使用最后一次采样的结果（即使有碰撞）
            if retry == max_retries - 1:
                if len(colliding_env_ids) > 0:
                    print(f"Warning: {len(colliding_env_ids)} environments still have finger collisions after {max_retries} retries. Using last sampled configuration.")
                break
            
            # 更新 last_joint：对于成功采样的环境，更新为新的关节位置；对于有碰撞的环境，保持上一次的值以便下次重试
            if len(success_env_ids) > 0:
                last_joint[success_env_ids, :] = rand_joint[success_env_ids, :]
            
            # 继续为有碰撞的环境重试（基于上一次的关节位置）
            remaining_env_ids = colliding_env_ids

        # self.ee_goal_sphere[env_ids, 0] = torch_rand_float(self.goal_ee_ranges["pos_l"][0], self.goal_ee_ranges["pos_l"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # self.ee_goal_sphere[env_ids, 1] = torch_rand_float(self.goal_ee_ranges["pos_p"][0], self.goal_ee_ranges["pos_p"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # self.ee_goal_sphere[env_ids, 2] = torch_rand_float(self.goal_ee_ranges["pos_y"][0], self.goal_ee_ranges["pos_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    
    # def _resample_ee_goal_orn_once(self, env_ids):
    #     ee_goal_delta_orn_r = torch_rand_float(self.goal_ee_ranges["delta_orn_r"][0], self.goal_ee_ranges["delta_orn_r"][1], (len(env_ids), 1), device=self.device)
    #     ee_goal_delta_orn_p = torch_rand_float(self.goal_ee_ranges["delta_orn_p"][0], self.goal_ee_ranges["delta_orn_p"][1], (len(env_ids), 1), device=self.device)
    #     ee_goal_delta_orn_y = torch_rand_float(self.goal_ee_ranges["delta_orn_y"][0], self.goal_ee_ranges["delta_orn_y"][1], (len(env_ids), 1), device=self.device)
    #     self.ee_goal_orn_delta_rpy[env_ids, :] = torch.cat([ee_goal_delta_orn_r, ee_goal_delta_orn_p, ee_goal_delta_orn_y], dim=-1)

    def _resample_ee_goal(self, env_ids, is_init=False):

        if len(env_ids) > 0:
            init_env_ids = env_ids.clone()

            # ee_local_cart = quat_rotate_inverse(self.base_yaw_quat, self.ee_pos - self.get_ee_goal_spherical_center())
            #         # Spherical to cartesian coordinates in the arm base frame 
            # radius = torch.norm(ee_local_cart, dim=1).view(self.num_envs,1)
            # pitch = torch.asin(ee_local_cart[:,2].view(self.num_envs,1)/radius).view(self.num_envs,1)
            # yaw = torch.atan2(ee_local_cart[:,1].view(self.num_envs,1), ee_local_cart[:,0].view(self.num_envs,1)).view(self.num_envs,1)
            # ee_pos_sphe_arm = torch.cat((radius, pitch, yaw), dim=1).view(self.num_envs,3)
            if is_init:
                if self.global_steps < 0 * 24 and not self.play:
                    self.finger_tip_goal_cart[env_ids] = self.init_start_finger_tip_cart[env_ids]
                    self.finger_tip_goal_orn[env_ids] = self.init_start_finger_tip_orn[env_ids]
                else:

                    base_quat_reshaped = self.base_quat.unsqueeze(1).expand(self.num_envs,len(self.finger_tips_idx), 4) #(num_envs, len(finger_tips_idx), 4)
                    base_pos_reshaped = self.base_pos.unsqueeze(1).expand(self.num_envs,len(self.finger_tips_idx), 3) #(num_envs, len(finger_tips_idx), 3)
                    self.finger_tip_goal_cart[env_ids] = quat_rotate_inverse(
                        base_quat_reshaped[env_ids,:,:].reshape(-1, 4),
                        (self.rigid_state[env_ids[:,None],self.finger_tips_idx,:3].clone()-  base_pos_reshaped[env_ids,:,:]).reshape(len(env_ids)*len(self.finger_tips_idx), 3)
                        ).reshape(len(env_ids), len(self.finger_tips_idx), 3)

                    # ⭐ 正确方法：四元数坐标转换（世界系 -> base系）
                    # 不能用 quat_rotate_inverse（那是旋转向量的），要用四元数乘法
                    finger_tip_quat_world = self.rigid_state[env_ids[:,None], self.finger_tips_idx, 3:7].clone() #(num_envs, len(finger_tips_idx), 4)
                    finger_tip_quat_base = quat_mul(quat_conjugate(base_quat_reshaped[env_ids].reshape(-1, 4)), finger_tip_quat_world.reshape(-1, 4)).reshape(len(env_ids), len(self.finger_tips_idx), 4)
                    self.finger_tip_goal_orn[env_ids] = finger_tip_quat_base
            else:
                if self.global_steps < 0 * 24 and not self.play:
                    self.finger_tip_goal_cart[env_ids] = self.init_start_finger_tip_cart[:]
                    self.finger_tip_goal_cart[env_ids] = self.init_start_finger_tip_cart[:]
                else:
                    self.finger_tip_start_cart[env_ids] = self.finger_tip_goal_cart[env_ids].clone()
                    self.finger_tip_start_orn[env_ids] = self.finger_tip_goal_orn[env_ids].clone()
                    
                    
                    # 随机采样，确保手指不会碰撞在一起
                    self._resample_ee_goal_cart_once(
                        env_ids, 
                        max_retries=self.cfg.goal_ee.max_collision_check_retries,
                        min_finger_distance=self.cfg.goal_ee.min_finger_distance
                    )
                        # collision_mask = self.collision_check(env_ids)
                        # env_ids = env_ids[collision_mask]
                        # if len(env_ids) == 0:
                        #     break
            self.goal_timer[init_env_ids] = 0.0

    def collision_check(self, env_ids):
        ee_target_all_sphere = torch.lerp(self.ee_start_sphere[env_ids, ..., None], self.ee_goal_sphere[env_ids, ...,  None], self.collision_check_t).squeeze(-1)
        ee_target_cart = sphere2cart(torch.permute(ee_target_all_sphere, (2, 0, 1)).reshape(-1, 3)).reshape(self.num_collision_check_samples, -1, 3)
        collision_mask = torch.any(torch.logical_and(torch.all(ee_target_cart < self.collision_upper_limits, dim=-1), torch.all(ee_target_cart > self.collision_lower_limits, dim=-1)), dim=0)
        underground_mask = torch.any(ee_target_cart[..., 2] < self.underground_limit, dim=0)
        return collision_mask | underground_mask

    def update_curr_ee_goal(self):
        if not self.cfg.env.teleop_mode:
            t = torch.clip(self.goal_timer / self.traj_timesteps, 0, 1)

            # translation
            # t[:, None, None] 形状为 (num_envs, 1, 1)，可以正确广播到 (num_envs, len(finger_tips_idx), 3)
            self.curr_finger_tip_goal_cart[:] = torch.lerp(self.finger_tip_start_cart, self.finger_tip_goal_cart, t[:, None,None])
            # commands 形状: (num_envs, len(finger_tips_idx), num_commands_per_finger_tip)
            # curr_finger_tip_goal_cart 形状: (num_envs, len(finger_tips_idx), 3)
            self.commands[:, :, INDEX_TIP_POS_X_CMD:(INDEX_TIP_POS_Z_CMD+1)] = self.curr_finger_tip_goal_cart
            
            # rotation
            # t[:, None, None] 形状为 (num_envs, 1, 1)，可以正确广播到 (num_envs, len(finger_tips_idx), 4)
            self.curr_finger_tip_goal_orn[:] = slerp_xyzw(self.finger_tip_start_orn, self.finger_tip_goal_orn, t[:, None, None])
            # curr_finger_tip_goal_orn 形状: (num_envs, len(finger_tips_idx), 4)
            self.commands[:, :, INDEX_TIP_ORIENTATION_X_CMD:(INDEX_TIP_ORIENTATION_W_CMD+1)] = self.curr_finger_tip_goal_orn

        self.goal_timer += 1
        resample_id = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
              
        self._resample_ee_goal(resample_id)
    

    # def get_ee_goal_spherical_center(self):
    #     center = torch.cat([self.root_states[:, :2], torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
    #     center = center + quat_apply(self.base_quat, self.ee_goal_center_offset)
    #     return center
    
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos
        # self.dof_pos[env_ids, 4:8] = self.default_dof_pos[4:8] * torch_rand_float(0.5, 1.5, (len(env_ids), 4), device=self.device)
         # 在关节限制范围内随机初始化灵巧手关节位置，而不是乘以因子（避免默认位置为0时无法随机化）
        low = self.dof_pos_limits[:, 0]
        high = self.dof_pos_limits[:, 1]
        # 在[low + 0.1*(high-low), high - 0.1*(high-low)]范围内随机，避免初始化在极限位置
        margin = 0.1 * (high - low)
        self.dof_pos[env_ids, :] = (low + margin) + (high - low - 2*margin) * torch.rand((len(env_ids), self.num_dofs), device=self.device)
        
        
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
   
    # def _reset_root_states(self, env_ids):
    #     """ Resets ROOT states position and velocities of selected environmments
    #         Sets base position based on the curriculum
    #         Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
    #     Args:
    #         env_ids (List[int]): Environemnt ids
    #     """
    #     # base position
    #     if self.custom_origins:
    #         self.root_states[env_ids] = self.base_init_state
    #         self.root_states[env_ids, :3] += self.env_origins[env_ids]
    #         self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
    #     else:
    #         self.root_states[env_ids] = self.base_init_state
    #         self.root_states[env_ids, :3] += self.env_origins[env_ids]
        
    #     # base orientation
    #     rand_yaw = self.cfg.init_state.rand_yaw_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
    #     quat = quat_from_euler_xyz(0*rand_yaw, 0*rand_yaw, rand_yaw) 
    #     self.root_states[env_ids, 3:7] = quat[:, :]  

    #     # base velocities
    #     self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
    #     env_ids_int32 = env_ids.to(dtype=torch.int32)
    #     self.gym.set_actor_root_state_tensor_indexed(self.sim,
    #                                                  gymtorch.unwrap_tensor(self.root_states),
    #                                                  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # def _push_robots(self):
    #     """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
    #     """
    #     max_vel = self.cfg.domain_rand.max_push_vel_xy
    #     self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
    #     self.root_states[:, 7:9] = torch.where(
    #         self.commands.sum(dim=1).unsqueeze(-1) == 0,
    #         self.root_states[:, 7:9] * 2.5,
    #         self.root_states[:, 7:9]
    #     )
    #     self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_finger_tip(self, env_ids_all):
        """ Randomly pushes the finger tips. Emulates an impulse by setting a randomized finger tip velocity.
        """

        if self.cfg.commands.push_finger_tips:
            # cmd force
            # FORCE CONTROLLED ENVS
            new_selected_env_ids_cmd = env_ids_all[(self.episode_length_buf % self.push_interval_finger_tips_cmd[:, 0]) == 0]
            
            # Define force and duration for the push 
            if new_selected_env_ids_cmd.nelement() > 0:

                self.freed_envs_finger_tips_cmd[new_selected_env_ids_cmd] = torch.rand(len(new_selected_env_ids_cmd), dtype=torch.float, device=self.device, requires_grad=False) > self.cfg.commands.finger_tips_forced_prob_cmd
                min_force_cmd_x = self.cfg.commands.max_push_force_finger_tips_cmd_x[0]
                max_force_cmd_x = self.cfg.commands.max_push_force_finger_tips_cmd_x[1]
                min_force_cmd_y = self.cfg.commands.max_push_force_finger_tips_cmd_y[0]
                max_force_cmd_y = self.cfg.commands.max_push_force_finger_tips_cmd_y[1]
                min_force_cmd_z = self.cfg.commands.max_push_force_finger_tips_cmd_z[0]
                max_force_cmd_z = self.cfg.commands.max_push_force_finger_tips_cmd_z[1]
                # torch_rand_float 只接受2维shape (int, int)，生成 (len(env_ids), num_fingers) 的张量
                self.force_target_finger_tips_cmd[new_selected_env_ids_cmd, :, 0] = torch_rand_float(min_force_cmd_x, max_force_cmd_x, (len(new_selected_env_ids_cmd), len(self.finger_tips_idx)), device=self.device)
                self.force_target_finger_tips_cmd[new_selected_env_ids_cmd, :, 1] = torch_rand_float(min_force_cmd_y, max_force_cmd_y, (len(new_selected_env_ids_cmd), len(self.finger_tips_idx)), device=self.device)
                self.force_target_finger_tips_cmd[new_selected_env_ids_cmd, :, 2] = torch_rand_float(min_force_cmd_z, max_force_cmd_z, (len(new_selected_env_ids_cmd), len(self.finger_tips_idx)), device=self.device)


                push_duration_finger_tips_cmd = torch_rand_float(self.push_duration_finger_tips_cmd_min, self.push_duration_finger_tips_cmd_max, (len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd)) # 4.0/self.dt
                push_duration_finger_tips_cmd = torch.clip(push_duration_finger_tips_cmd, max=(self.push_interval_finger_tips_cmd[new_selected_env_ids_cmd, 0] - self.settling_time_force_finger_tips)/2).to(self.device)
                self.push_end_time_finger_tips_cmd[new_selected_env_ids_cmd] = self.episode_length_buf[new_selected_env_ids_cmd] + push_duration_finger_tips_cmd
                self.push_duration_finger_tips_cmd[new_selected_env_ids_cmd] = push_duration_finger_tips_cmd

                self.selected_env_ids_finger_tips_cmd[new_selected_env_ids_cmd] = 1
                
            # Get ids of all envs to apply a force to 
            if self.episode_length_buf[self.selected_env_ids_finger_tips_cmd == 1].nelement() > 0:
                subset_env_ids_selected = env_ids_all[self.selected_env_ids_finger_tips_cmd == 1]

                # Step 1: apply force from 0 to force_target_finger_tips_cmd
                env_ids_apply_push_step1 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_finger_tips_cmd == 1] < (self.push_end_time_finger_tips_cmd[self.selected_env_ids_finger_tips_cmd == 1]).type(torch.int32)]
                # print(env_ids_apply_push_step1)
                if env_ids_apply_push_step1.nelement() > 0:
                    push_duration_reshaped = self.push_duration_finger_tips_cmd[env_ids_apply_push_step1].unsqueeze(-1).unsqueeze(-1)
                    
                    self.current_Fxyz_finger_tips_cmd_local[env_ids_apply_push_step1, :, :3] = (self.force_target_finger_tips_cmd[env_ids_apply_push_step1, :, :3]/push_duration_reshaped)*(
                        torch.clamp(self.episode_length_buf[env_ids_apply_push_step1].unsqueeze(-1).unsqueeze(-1) - (self.push_end_time_finger_tips_cmd[env_ids_apply_push_step1].unsqueeze(-1).unsqueeze(1)-push_duration_reshaped), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                    
                    
                    forces_local = self.current_Fxyz_finger_tips_cmd_local[env_ids_apply_push_step1]
                    finger_tip_quat = self.rigid_state[env_ids_apply_push_step1[..., None], self.finger_tips_idx, 3:7]
                    base_quat_expanded = self.base_quat[env_ids_apply_push_step1].unsqueeze(1).expand(len(env_ids_apply_push_step1), len(self.finger_tips_idx), 4)
                    
                    # finger_tip local -> world
                    forces_world = quat_apply(
                        finger_tip_quat.reshape(-1, 4),
                        forces_local.reshape(-1, 3)
                    ).reshape(len(env_ids_apply_push_step1), len(self.finger_tips_idx), 3)
                    
                    # world -> base
                    F_cmd_base  = quat_rotate_inverse(
                        base_quat_expanded.reshape(-1, 4),
                        forces_world.reshape(-1, 3)
                    ).reshape(len(env_ids_apply_push_step1), len(self.finger_tips_idx), 3)

                    self.commands[env_ids_apply_push_step1, :, INDEX_TIP_FORCE_X] = F_cmd_base[:, :, 0]
                    self.commands[env_ids_apply_push_step1, :, INDEX_TIP_FORCE_Y] = F_cmd_base[:, :, 1]
                    self.commands[env_ids_apply_push_step1, :, INDEX_TIP_FORCE_Z] = F_cmd_base[:, :, 2]
 
                # Step 2: apply force from force_target_gripper_cmd back to 0
                env_ids_apply_push_step2 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_finger_tips_cmd == 1] > (self.push_end_time_finger_tips_cmd[self.selected_env_ids_finger_tips_cmd == 1] + self.settling_time_force_finger_tips).type(torch.int32)]
                if env_ids_apply_push_step2.nelement() > 0:
                    push_duration_reshaped = self.push_duration_finger_tips_cmd[env_ids_apply_push_step2].unsqueeze(-1).unsqueeze(-1)
                    self.current_Fxyz_finger_tips_cmd_local[env_ids_apply_push_step2,:, :3] = self.force_target_finger_tips_cmd[env_ids_apply_push_step2, :, :3] - (self.force_target_finger_tips_cmd[env_ids_apply_push_step2, :, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step2].unsqueeze(-1).unsqueeze(-1) - (self.push_end_time_finger_tips_cmd[env_ids_apply_push_step2].unsqueeze(-1).unsqueeze(1)+self.settling_time_force_finger_tips), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                


                    forces_local = self.current_Fxyz_finger_tips_cmd_local[env_ids_apply_push_step2]
                    finger_tip_quat = self.rigid_state[env_ids_apply_push_step2[..., None], self.finger_tips_idx, 3:7]
                    base_quat_expanded = self.base_quat[env_ids_apply_push_step2].unsqueeze(1).expand(len(env_ids_apply_push_step2), len(self.finger_tips_idx), 4)
                    
                    # finger_tip local -> world
                    forces_world = quat_apply(
                        finger_tip_quat.reshape(-1, 4),
                        forces_local.reshape(-1, 3)
                    ).reshape(len(env_ids_apply_push_step2), len(self.finger_tips_idx), 3)
                    
                    # world -> base
                    F_cmd_base  = quat_rotate_inverse(
                        base_quat_expanded.reshape(-1, 4),
                        forces_world.reshape(-1, 3)
                    ).reshape(len(env_ids_apply_push_step2), len(self.finger_tips_idx), 3)

                    self.commands[env_ids_apply_push_step2, :, INDEX_TIP_FORCE_X] = F_cmd_base[:, :, 0]
                    self.commands[env_ids_apply_push_step2, :, INDEX_TIP_FORCE_Y] = F_cmd_base[:, :, 1]
                    self.commands[env_ids_apply_push_step2, :, INDEX_TIP_FORCE_Z] = F_cmd_base[:, :, 2]
                    
                # Reset the tensors
                env_ids_to_reset = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_finger_tips_cmd == 1] >= (self.push_end_time_finger_tips_cmd[self.selected_env_ids_finger_tips_cmd == 1] + self.settling_time_force_finger_tips + self.push_duration_finger_tips_cmd[self.selected_env_ids_finger_tips_cmd == 1]).type(torch.int32)]
                if env_ids_to_reset.nelement() > 0:
                    self.selected_env_ids_finger_tips_cmd[env_ids_to_reset] = 0
                    self.force_target_finger_tips_cmd[env_ids_to_reset, :3] = 0.
                    self.current_Fxyz_finger_tips_cmd_local[env_ids_to_reset, :3] = 0.
                    self.push_end_time_finger_tips_cmd[env_ids_to_reset] = 0.
                    self.push_duration_finger_tips_cmd[env_ids_to_reset] = 0.
                    self.commands[env_ids_to_reset,:, INDEX_TIP_FORCE_X] = 0.0
                    self.commands[env_ids_to_reset,:, INDEX_TIP_FORCE_Y] = 0.0
                    self.commands[env_ids_to_reset,:, INDEX_TIP_FORCE_Z] = 0.0
                    self.push_interval_finger_tips_cmd[env_ids_to_reset, 0] = torch.randint(int(self.push_interval_finger_tips_cmd_min), int(self.push_interval_finger_tips_cmd_max), (len(env_ids_to_reset), 1), device=self.device)[:, 0]
                    
            self.selected_env_ids_finger_tips_cmd[self.freed_envs_finger_tips_cmd] = 0
            self.force_target_finger_tips_cmd[self.freed_envs_finger_tips_cmd,:, :3] = 0.
            self.current_Fxyz_finger_tips_cmd_local[self.freed_envs_finger_tips_cmd,:, :3] = 0.
            self.push_end_time_finger_tips_cmd[self.freed_envs_finger_tips_cmd] = 0.
            self.push_duration_finger_tips_cmd[self.freed_envs_finger_tips_cmd] = 0. 
            self.commands[self.freed_envs_finger_tips_cmd,:, INDEX_TIP_FORCE_X] = 0.0
            self.commands[self.freed_envs_finger_tips_cmd,:, INDEX_TIP_FORCE_Y] = 0.0
            self.commands[self.freed_envs_finger_tips_cmd,:, INDEX_TIP_FORCE_Z] = 0.0


            # ext force
            # FORCE CONTROLLED ENVS
            new_selected_env_ids_ext = env_ids_all[(self.episode_length_buf % self.push_interval_finger_tips_ext[:, 0]) == 0]
            
            # Define force and duration for the push 
            if new_selected_env_ids_ext.nelement() > 0:
                
                self.freed_envs_finger_tips_ext[new_selected_env_ids_ext] = torch.rand(len(new_selected_env_ids_ext), dtype=torch.float, device=self.device, requires_grad=False) > self.cfg.commands.finger_tips_forced_prob_ext
                min_force_ext_x = self.cfg.commands.max_push_force_finger_tips_ext_x[0]
                max_force_ext_x = self.cfg.commands.max_push_force_finger_tips_ext_x[1]
                min_force_ext_y = self.cfg.commands.max_push_force_finger_tips_ext_y[0]
                max_force_ext_y = self.cfg.commands.max_push_force_finger_tips_ext_y[1]
                min_force_ext_z = self.cfg.commands.max_push_force_finger_tips_ext_z[0]
                max_force_ext_z = self.cfg.commands.max_push_force_finger_tips_ext_z[1]

                # torch_rand_float 只接受2维shape (int, int)，生成 (len(env_ids), num_fingers) 的张量
                self.force_target_finger_tips_ext[new_selected_env_ids_ext, :, 0] = torch_rand_float(min_force_ext_x, max_force_ext_x, (len(new_selected_env_ids_ext), len(self.finger_tips_idx)), device=self.device)
                self.force_target_finger_tips_ext[new_selected_env_ids_ext, :, 1] = torch_rand_float(min_force_ext_y, max_force_ext_y, (len(new_selected_env_ids_ext), len(self.finger_tips_idx)), device=self.device)
                self.force_target_finger_tips_ext[new_selected_env_ids_ext, :, 2] = torch_rand_float(min_force_ext_z, max_force_ext_z, (len(new_selected_env_ids_ext), len(self.finger_tips_idx)), device=self.device)
                
                # # ⚠️ 修复：正确的形状设置 - 为所有手指设置 [1, 1, 1]
                # self.force_target_finger_tips_ext[new_selected_env_ids_ext, :, 0:3] = torch.ones(
                #     len(new_selected_env_ids_ext), len(self.finger_tips_idx), 3, 
                #     device=self.device
                # )
                
                push_duration_finger_tips_ext = torch_rand_float(self.push_duration_finger_tips_ext_min, self.push_duration_finger_tips_ext_max, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext)) # 4.0/self.dt
                push_duration_finger_tips_ext = torch.clip(push_duration_finger_tips_ext, max=(self.push_interval_finger_tips_ext[new_selected_env_ids_ext, 0] - self.settling_time_force_finger_tips)/2).to(self.device)
                self.push_end_time_finger_tips_ext[new_selected_env_ids_ext] = self.episode_length_buf[new_selected_env_ids_ext] + push_duration_finger_tips_ext
                self.push_duration_finger_tips_ext[new_selected_env_ids_ext] = push_duration_finger_tips_ext
                
                self.selected_env_ids_finger_tips_ext[new_selected_env_ids_ext] = 1
                
            # Get ids of all envs to apply a force to 
            if self.episode_length_buf[self.selected_env_ids_finger_tips_ext == 1].nelement() > 0:
                subset_env_ids_selected = env_ids_all[self.selected_env_ids_finger_tips_ext == 1]

                # Step 1: apply force from 0 to force_target_finger_tips_cmd
                env_ids_apply_push_step1 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_finger_tips_ext == 1] < (self.push_end_time_finger_tips_ext[self.selected_env_ids_finger_tips_ext == 1]).type(torch.int32)]
                # print(env_ids_apply_push_step1)
                if env_ids_apply_push_step1.nelement() > 0:
                    push_duration_reshaped = self.push_duration_finger_tips_ext[env_ids_apply_push_step1].unsqueeze(-1).unsqueeze(-1)
        
                    self.forces[env_ids_apply_push_step1[:, None], self.finger_tips_idx, :3] = (self.force_target_finger_tips_ext[env_ids_apply_push_step1,:, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step1].unsqueeze(-1).unsqueeze(-1) - (self.push_end_time_finger_tips_ext[env_ids_apply_push_step1].unsqueeze(-1).unsqueeze(1)-push_duration_reshaped), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                  
                # Step 2: apply force from force_target_finger_tips_cmd back to 0
                env_ids_apply_push_step2 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_finger_tips_ext == 1] > (self.push_end_time_finger_tips_ext[self.selected_env_ids_finger_tips_ext == 1] + self.settling_time_force_finger_tips).type(torch.int32)]
                if env_ids_apply_push_step2.nelement() > 0:
                    push_duration_reshaped = self.push_duration_finger_tips_ext[env_ids_apply_push_step2].unsqueeze(-1).unsqueeze(-1)
                    
                    # world frame
                    self.forces[env_ids_apply_push_step2[:, None], self.finger_tips_idx, :3] = self.force_target_finger_tips_ext[env_ids_apply_push_step2,:, :3] - (self.force_target_finger_tips_ext[env_ids_apply_push_step2,:, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step2].unsqueeze(-1).unsqueeze(-1) - (self.push_end_time_finger_tips_ext[env_ids_apply_push_step2].unsqueeze(-1).unsqueeze(1)+self.settling_time_force_finger_tips), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                
                    
                # Reset the tensors
                env_ids_to_reset = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_finger_tips_ext == 1] >= (self.push_end_time_finger_tips_ext[self.selected_env_ids_finger_tips_ext == 1] + self.settling_time_force_finger_tips + self.push_duration_finger_tips_ext[self.selected_env_ids_finger_tips_ext == 1]).type(torch.int32)]                                        
                if env_ids_to_reset.nelement() > 0:
                    self.selected_env_ids_finger_tips_ext[env_ids_to_reset] = 0
                    self.force_target_finger_tips_ext[env_ids_to_reset,:,:3] = 0.
                    self.push_end_time_finger_tips_ext[env_ids_to_reset] = 0.
                    self.push_duration_finger_tips_ext[env_ids_to_reset] = 0.
                    self.push_interval_finger_tips_ext[env_ids_to_reset, 0] = torch.randint(int(self.push_interval_finger_tips_ext_min), int(self.push_interval_finger_tips_ext_max), (len(env_ids_to_reset), 1), device=self.device)[:, 0]
                    
            self.selected_env_ids_finger_tips_ext[self.freed_envs_finger_tips_ext] = 0
            self.force_target_finger_tips_ext[self.freed_envs_finger_tips_ext,:, :3] = 0.
            self.push_end_time_finger_tips_ext[self.freed_envs_finger_tips_ext] = 0.
            self.push_duration_finger_tips_ext[self.freed_envs_finger_tips_ext] = 0. 

            freed_envs_finger_tips_ext_reshaped = torch.nonzero(self.freed_envs_finger_tips_ext).squeeze(1)
            self.forces[freed_envs_finger_tips_ext_reshaped[:, None], self.finger_tips_idx, :3] = 0

            

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # noise_vec[:2] = noise_scales.gravity * noise_level
        # noise_vec[2:5] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # noise_vec[5:5+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # noise_vec[5+self.num_actions:5+self.num_actions*2] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # noise_vec[5+self.num_actions*2:5+self.num_actions*3] = 0. # previous actions
        # noise_vec[5+self.num_actions*3:] = 0. # commands

        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        force_tensor = self.gym.acquire_force_sensor_tensor(self.sim)


        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]


        self.base_quat = self.root_states[:, 3:7]
        self.base_pos = self.root_states[:, :3]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        # base_yaw = euler_from_quat(self.base_quat)[2]
        # self.base_yaw_euler = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        # self.base_yaw_quat = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)


        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)
        # self.sensors_forces = gymtorch.wrap_tensor(force_tensor).view(self.num_envs, -1, 6) # shape: num_envs, num_sensors, xyz axis

        # # 记得添加其他sensor
        # self.sensors_world = torch.zeros(self.num_envs, len(self.finger_tips_idx), 7, device=self.device) # pos(3) + orn(4)
        # for i in range(len(self.finger_tips_idx)):
        #     tip_idx = self.finger_tips_idx[i]
        #     link_pos = self.rigid_state[:, tip_idx,:3] # shape: num_envs, 3
        #     link_q = self.rigid_state[:, tip_idx,3:7]
        #     offset = self.sensors_pos_link[i, :3]
        #     offset = offset.expand_as(link_pos)     
        #     self.sensors_world[:, i, :3] = (link_pos + offset).squeeze(1)          

        #     q_rel = self.sensors_pos_link[i, 3:7] 
        #     self.sensors_world[:, i, 3:7] = quat_mul(link_q , q_rel).squeeze(1)

        # # ee info
        self.finger_tips_pos = self.rigid_state[:, self.finger_tips_idx, :3] # shape: num_envs, num_finger_tips, 3
        self.finger_tips_orn = self.rigid_state[:, self.finger_tips_idx, 3:7] # shape: num_envs, num_finger_tips, 4

        # self.grasp_offset = self.cfg.arm.grasp_offset

        # target_ee info 一只手5个finger_tips同时开始同时结束
        self.traj_timesteps = torch_rand_float(self.cfg.goal_ee.traj_time[0], self.cfg.goal_ee.traj_time[1], (self.num_envs, 1), device=self.device).squeeze(1) / self.dt
        self.traj_total_timesteps = self.traj_timesteps + torch_rand_float(self.cfg.goal_ee.hold_time[0], self.cfg.goal_ee.hold_time[1], (self.num_envs, 1), device=self.device) / self.dt
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)

        self.finger_tip_start_cart =torch.zeros(self.num_envs, len(self.finger_tips_idx), 3, device=self.device)
        self.finger_tip_goal_cart = torch.zeros(self.num_envs, len(self.finger_tips_idx), 3, device=self.device)
        self.curr_finger_tip_goal_cart = torch.zeros(self.num_envs, len(self.finger_tips_idx), 3, device=self.device)
        self.init_start_finger_tip_cart = torch.tensor(self.cfg.goal_ee.ranges.init_pos_start, device=self.device,dtype = torch.float).repeat(self.num_envs, len(self.finger_tips_idx), 3)
        self.init_end_finger_tip_cart = torch.tensor(self.cfg.goal_ee.ranges.init_pos_end, device=self.device,dtype = torch.float).repeat(self.num_envs, len(self.finger_tips_idx), 3)

        # x,y,z,ws (初始化为单位四元数 [0, 0, 0, 1])
        self.finger_tip_start_orn = torch.zeros(self.num_envs,len(self.finger_tips_idx), 4, device=self.device)
        self.finger_tip_start_orn[:,:, 3] = 1.0  # w = 1
        self.finger_tip_goal_orn = torch.zeros(self.num_envs,len(self.finger_tips_idx), 4, device=self.device)
        self.finger_tip_goal_orn[:,:, 3] = 1.0  # w = 1
        self.curr_finger_tip_goal_orn = torch.zeros(self.num_envs,len(self.finger_tips_idx), 4, device=self.device)
        self.curr_finger_tip_goal_orn[:,:, 3] = 1.0  # w = 1
        self.init_start_finger_tip_orn = torch.tensor(self.cfg.arm.init_target_ee_orn, device=self.device).repeat(self.num_envs, 4)
        self.init_end_finger_tip_orn = torch.tensor(self.cfg.arm.init_target_ee_orn, device=self.device).repeat(self.num_envs, 4)
        # self.curr_finger_tip_goal_cart = self.init_start_finger_tip_cart
        # self.ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        # self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        
        # self.ee_goal_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        # self.ee_goal_orn_euler[:, 0] = np.pi / 2
        # self.ee_goal_orn_quat = quat_from_euler_xyz(self.ee_goal_orn_euler[:, 0], self.ee_goal_orn_euler[:, 1], self.ee_goal_orn_euler[:, 2])
        # self.ee_goal_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)

        # self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        # self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        # self.ee_pos_sphe_arm = torch.zeros(self.num_envs, 3, device=self.device)

        # self.init_start_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_start, device=self.device).unsqueeze(0)
        # self.init_end_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_end, device=self.device).unsqueeze(0)
        self.collision_lower_limits = torch.tensor(self.cfg.goal_ee.collision_lower_limits, device=self.device, dtype=torch.float)
        self.collision_upper_limits = torch.tensor(self.cfg.goal_ee.collision_upper_limits, device=self.device, dtype=torch.float)
        self.underground_limit = self.cfg.goal_ee.underground_limit
        self.num_collision_check_samples = self.cfg.goal_ee.num_collision_check_samples
        self.collision_check_t = torch.linspace(0, 1, self.num_collision_check_samples, device=self.device)[None, None, :]
        assert(self.cfg.goal_ee.command_mode in ['cart', 'sphere'])
        # self.sphere_error_scale = torch.tensor(self.cfg.goal_ee.sphere_error_scale, device=self.device)
        # self.orn_error_scale = torch.tensor(self.cfg.goal_ee.orn_error_scale, device=self.device)
        # self.ee_goal_center_offset = torch.tensor([self.cfg.goal_ee.sphere_center.x_offset, 
        #                                            self.cfg.goal_ee.sphere_center.y_offset, 
        #                                            self.cfg.goal_ee.sphere_center.z_invariant_offset], 
        #                                            device=self.device).repeat(self.num_envs, 1)
        
        # self.curr_ee_goal_cart_world = self.get_ee_goal_spherical_center() + quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)

        # initialize some data used later on
        self.gripper_force_kps = torch.zeros(self.num_envs, len(self.finger_tips_idx), 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
     
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_rigid_state = torch.zeros_like(self.rigid_state)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        self.commands = torch.zeros(self.num_envs, len(self.finger_tips_idx), self.cfg.commands.num_commands_per_finger_tip, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        # self.commands_scale = torch.tensor([self.obs_scales.lin_vel, 
        #                                     self.obs_scales.lin_vel, 
        #                                     self.obs_scales.ang_vel,
        #                                     self.obs_scales.ee_sphe_radius_cmd, 
        #                                     self.obs_scales.ee_sphe_pitch_cmd,
        #                                     self.obs_scales.ee_sphe_yaw_cmd,
        #                                     self.obs_scales.end_effector_roll_cmd, 
        #                                     self.obs_scales.end_effector_pitch_cmd,
        #                                     self.obs_scales.end_effector_yaw_cmd,
        #                                     self.obs_scales.ee_force,
        #                                     self.obs_scales.ee_force,
        #                                     self.obs_scales.ee_force,
        #                                     self.obs_scales.base_force,
        #                                     self.obs_scales.base_force,
        #                                     self.obs_scales.base_force,], device=self.device, requires_grad=False,) # TODO change this
      
        # 现在只考虑食指,以后记得修改
        # rand_joint = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        # low  = self.dof_pos_limits[4:8, 0]  # shape: (4,)
        # high = self.dof_pos_limits[4:8, 1]  # shape: (4,)

        # rand_joint[:, 4:8] = low + (high - low) * torch.rand((self.num_envs, 4), device=self.device)
        # rand_joint[:,4] = 0.6
        # self._pinocchio_forward_kinematics(rand_joint, torch.arange(self.num_envs, device=self.device))
        # self.curr_finger_tip_goal_cart = self.finger_tip_goal_cart.clone()


        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.cfg.env.num_single_obs, dtype=torch.float, device=self.device))
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(torch.zeros(
                self.num_envs, self.cfg.env.single_num_privileged_obs, dtype=torch.float, device=self.device))
        
        # 实时绘图数据缓冲区（用于记录 forces 和 sensor_forces）
        self.plot_history_length = 1000  # 记录最近1000步的数据
        self.forces_history = deque(maxlen=self.plot_history_length)  # 存储 forces[0,1,0:3]
        self.cmd_forces_history = deque(maxlen=self.plot_history_length)  # 存储 sensors_forces[0,1,0:3]
        self.step_history = deque(maxlen=self.plot_history_length)  # 存储步数
        
        # 初始化 matplotlib 实时绘图
        self.enable_realtime_plot = False  # 可以通过这个开关控制是否绘图
        if self.enable_realtime_plot and not self.headless:
            plt.ion()  # 开启交互模式
            self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 10))
            self.fig.suptitle('Forces and Sensor Forces (Env 0, Finger 1)', fontsize=14)
            self.axes[0].set_title('X Component')
            self.axes[0].set_ylabel('Force (N)')
            self.axes[1].set_title('Y Component')
            self.axes[1].set_ylabel('Force (N)')
            self.axes[2].set_title('Z Component')
            self.axes[2].set_ylabel('Force (N)')
            self.axes[2].set_xlabel('Step')
            plt.tight_layout()

        # joint positions offsets and PD gains
        self.default_dof_pos = 0.5* torch.ones(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
       
        self.dof_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        # force control

        # Push tip 
        self.freed_envs_finger_tips_cmd = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.freed_envs_finger_tips_ext = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.selected_env_ids_finger_tips_cmd = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        self.selected_env_ids_finger_tips_ext = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)


        self.push_interval_finger_tips_cmd = torch.randint(int(self.push_interval_finger_tips_cmd_min), int(self.push_interval_finger_tips_cmd_max), (self.num_envs, 1), device=self.device, requires_grad=False)
        self.push_interval_finger_tips_ext = torch.randint(int(self.push_interval_finger_tips_ext_min), int(self.push_interval_finger_tips_ext_max), (self.num_envs, 1), device=self.device, requires_grad=False)
        self.push_end_time_finger_tips_cmd = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_duration_finger_tips_cmd = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.settling_time_force_finger_tips_s = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.push_end_time_finger_tips_ext = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_duration_finger_tips_ext = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.settling_time_force_finger_tips_ext_s = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        #需要对外界表现出的力 现在只有食指
        self.force_target_finger_tips_cmd = torch.zeros(self.num_envs,len(self.finger_tips_idx), 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.current_Fxyz_finger_tips_cmd_local = torch.zeros(self.num_envs,len(self.finger_tips_idx), 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        # 当前需要对手指主动施加的外力
        self.force_target_finger_tips_ext = torch.zeros(self.num_envs,len(self.finger_tips_idx), 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.forces = torch.zeros(self.num_envs, self.num_bodies, 6, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        # self.forces_local = torch.zeros(self.num_envs, self.num_bodies, 6, dtype=torch.float, device=self.device,
        #                            requires_grad=False)

        self.global_steps = 0

        self.reward_logs = dict(    )

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)  
        print("Trimesh added")
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        # asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        asset_options.use_mesh_materials = True

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # pinocchio
        mesh_dir = os.path.dirname(asset_path)
        robot = RobotWrapper.BuildFromURDF(asset_path, mesh_dir)
        self.pinocchio_model = robot.model
        self.pinocchio_data = self.pinocchio_model.createData()
        self.pinocchio_tips_idx = [
                                    # self.pinocchio_model.getFrameId("fingertip_thumb"),
                                    # self.pinocchio_model.getFrameId("fingertip_index"),
                                    # self.pinocchio_model.getFrameId("fingertip_middle"),
                                    # self.pinocchio_model.getFrameId("fingertip_ring"),
                                    # self.pinocchio_model.getFrameId("fingertip_pinky"),
                                    self.pinocchio_model.getFrameId("finger1_tip_link"),
                                    self.pinocchio_model.getFrameId("finger2_tip_link"),
                                    self.pinocchio_model.getFrameId("finger3_tip_link"),
                                    self.pinocchio_model.getFrameId("finger4_tip_link"),
                                    self.pinocchio_model.getFrameId("finger5_tip_link"),
                                ]


        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        # self.finger_tips_idx = [self.body_names_to_idx[i] for i in self.cfg.asset.finger_tip_name]
        self.finger_tips_idx = [5,10,15,20,25]
        self.finger_tips_idx = torch.tensor(self.finger_tips_idx, dtype=torch.long, device=self.device)

        self.robot_palm_idx = [index for index, body_name in enumerate(body_names) if body_name == "palm_link"][0]
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)

        # feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        # thigh_names = [s for s in body_names if self.cfg.asset.thigh_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # base_init_state_list = self.cfg.init_state.default_joint_pos
        # self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.sensors_handle = [[] for _ in range(self.num_envs)]

        self.sensors_pos_link = torch.zeros(len(self.finger_tips_idx), 7, dtype=torch.float32, device=self.device) # x,y,z,(x,y,z,w)
        
        # # sensor相对于tip_link的坐标
        # self.sensors_pos_link = torch.tensor([[0.,0.,0.,0,0,0,1] for _ in range(len(self.finger_tips_idx))], dtype=torch.float32, device=self.device)


        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)

        self.mass_params_tensor = torch.zeros(self.num_envs, 22, dtype=torch.float, device=self.device, requires_grad=False)
        
        # # add tactile sensor on the finger tips
        # for i,tips_idx in enumerate(self.finger_tips_idx):
        #     p = gymapi.Vec3(self.sensors_pos_link[i][0], self.sensors_pos_link[i][1], self.sensors_pos_link[i][2])
        #     q = gymapi.Quat(self.sensors_pos_link[i][3], self.sensors_pos_link[i][4], self.sensors_pos_link[i][5], self.sensors_pos_link[i][6])
        #     sensor_pose = gymapi.Transform(
        #         p=p,
        #         r=q
        #     )
        #     sensor_props = gymapi.ForceSensorProperties()
        #     sensor_props.use_world_frame = False
        #     self.gym.create_asset_force_sensor(robot_asset, tips_idx, sensor_pose,sensor_props)
        
        for i in range(self.num_envs):

            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(self.env_origins[i, 0], self.env_origins[i, 1], self.env_origins[i, 2])

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)


            # num_sensors = self.gym.get_actor_force_sensor_count(env_handle, actor_handle)
            # for sensor_idx in range(num_sensors):
            #     sensor = self.gym.get_actor_force_sensor(env_handle, actor_handle, sensor_idx)
            #     self.sensors_handle[i].append(sensor)
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device)

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        # for i in range(len(feet_names)):
        #     self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        # self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        # for i in range(len(thigh_names)):
        #     self.thigh_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], thigh_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])


        self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).squeeze(-1)

        # if self.cfg.domain_rand.randomize_motor:
        #     self.motor_strength = torch.cat([
        #             torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, 20), device=self.device)
        #         ], dim=1)
        # else:
        #     self.motor_strength = torch.ones(self.num_envs, self.num_joints, device=self.device)
        
        # hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        # self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        # for i, name in enumerate(hip_names):
        #     self.hip_indices[i] = self.dof_names.index(name)

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        self.custom_origins = True
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # put robots at the origins defined by the terrain
        max_init_level = self.cfg.terrain.max_init_terrain_level  # start from 0
        if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
        self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
        self.max_terrain_level = self.cfg.terrain.num_rows
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

    def _parse_cfg(self, cfg):
        # self.num_torques = self.cfg.env.num_torques
        self.num_joints = self.cfg.env.num_joints
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)

        # self.goal_ee_ranges = class_to_dict(self.cfg.goal_ee.ranges)
     
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

       
        self.push_interval_finger_tips_cmd_min = np.ceil(self.cfg.commands.push_tip_interval_s_cmd[0] / self.dt)
        self.push_interval_finger_tips_cmd_max = np.ceil(self.cfg.commands.push_tip_interval_s_cmd[1] / self.dt)
        self.push_duration_finger_tips_cmd_min = np.ceil(self.cfg.commands.push_tip_duration_s_cmd[0] / self.dt)
        self.push_duration_finger_tips_cmd_max = np.ceil(self.cfg.commands.push_tip_duration_s_cmd[1] / self.dt)
        self.push_interval_finger_tips_ext_min = np.ceil(self.cfg.commands.push_tip_interval_s_ext[0] / self.dt)
        self.push_interval_finger_tips_ext_max = np.ceil(self.cfg.commands.push_tip_interval_s_ext[1] / self.dt)
        self.push_duration_finger_tips_ext_min = np.ceil(self.cfg.commands.push_tip_duration_s_ext[0] / self.dt)
        self.push_duration_finger_tips_ext_max = np.ceil(self.cfg.commands.push_tip_duration_s_ext[1] / self.dt)
        self.settling_time_force_finger_tips = np.ceil(self.cfg.commands.settling_time_force_finger_tips_s/ self.dt)
        # self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
       
        
        self.action_delay = self.cfg.env.action_delay

        self.stop_update_goal = False

    # def get_walking_cmd_mask(self, env_ids=None, return_all=False):
    #     if env_ids is None:
    #         env_ids = torch.arange(self.num_envs, device=self.device)
    #     walking_mask0 = torch.abs(self.commands[env_ids, 0]) > self.cfg.commands.lin_vel_x_clip
    #     walking_mask1 = torch.abs(self.commands[env_ids, 1]) > self.cfg.commands.lin_vel_y_clip
    #     walking_mask2 = torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_yaw_clip
    #     walking_mask = walking_mask0 | walking_mask1 | walking_mask2

    #     if return_all:
    #         return walking_mask0, walking_mask1, walking_mask2, walking_mask
    #     return walking_mask


    def _pinocchio_forward_kinematics(self, q, env_ids=None):
        """
        使用Pinocchio进行正向运动学计算，计算所有手指的位置和姿态
        
        Args:
            q: 关节角度，形状为 (num_envs, num_dofs)
            env_ids: 需要计算的环境ID列表
        """
        if len(env_ids)==0 :
            return 
        for i in range(len(env_ids)):
            idx = env_ids[i]
            q_i = q[idx].cpu().double().numpy().reshape(-1)
            pin.forwardKinematics(self.pinocchio_model, self.pinocchio_data, q_i)
            pin.updateFramePlacements(self.pinocchio_model, self.pinocchio_data)

            # 计算所有5个手指的位置和姿态
            for finger_idx in range(len(self.pinocchio_tips_idx)):
                pinocchio_tip_frame_idx = self.pinocchio_tips_idx[finger_idx]
                tip_link_idx = self.finger_tips_idx[finger_idx]
                # translation
                tip_goal_cart_base = self.pinocchio_data.oMf[pinocchio_tip_frame_idx].translation.copy()
                self.finger_tip_goal_cart[idx, finger_idx] = torch.from_numpy(tip_goal_cart_base).to(self.device, dtype=torch.float32)
                # rotation
                tip_goal_rotation_base = self.pinocchio_data.oMf[pinocchio_tip_frame_idx].rotation.copy() # 3*3 matrix
                R_tensor = torch.from_numpy(tip_goal_rotation_base).to(self.device, dtype=torch.float32)
                R_tensor = R_tensor.unsqueeze(0)
                tip_goal_orn_base = mat3x3_to_xyzw(R_tensor).squeeze(0)  # (1, 4) -> (4,)
                self.finger_tip_goal_orn[idx, finger_idx] = tip_goal_orn_base
    
    def _check_finger_self_collision(self, env_ids, min_distance=0.005):
        """
        检查手指之间是否发生自碰撞
        
        Args:
            env_ids: 需要检查的环境ID列表
            min_distance: 手指之间的最小安全距离（米），默认0.02m
        
        Returns:
            collision_mask: 布尔张量，True表示无碰撞，False表示有碰撞
        """
        if len(env_ids) == 0:
            return torch.tensor([], dtype=torch.bool, device=self.device)
        
        # 获取所有手指的位置 (len(env_ids), num_fingers, 3)
        finger_positions = self.finger_tip_goal_cart[env_ids]  # shape: (len(env_ids), 5, 3)
        num_fingers = finger_positions.shape[1]
        
        # 计算所有手指对之间的距离
        collision_mask = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        
        for i in range(num_fingers):
            for j in range(i + 1, num_fingers):
                # 计算手指i和手指j之间的距离
                pos_i = finger_positions[:, i, :]  # (len(env_ids), 3)
                pos_j = finger_positions[:, j, :]  # (len(env_ids), 3)
                distances = torch.norm(pos_i - pos_j, dim=1)  # (len(env_ids),)
                
                # 如果距离小于最小安全距离，则认为发生碰撞
                collision_mask = collision_mask & (distances >= min_distance)
        
        return collision_mask 


    # def compute_ref_state(self):
    #     phase = self._get_phase()
    #     sin_pos = torch.sin(2 * torch.pi * phase)
    #     sin_pos_l = sin_pos.clone() + self.cfg.rewards.target_joint_pos_thd
    #     sin_pos_r = sin_pos.clone() - self.cfg.rewards.target_joint_pos_thd
    #     repeat_default_pos = self.default_dof_pos[:, :12].repeat(self.num_envs, 1)
    #     # self.ref_dof_pos = torch.zeros_like(self.dof_pos)
    #     self.ref_dof_pos = repeat_default_pos.clone()
    #     scale_1 = self.cfg.rewards.target_joint_pos_scale / (1 - self.cfg.rewards.target_joint_pos_thd)
    #     scale_2 = scale_1 * 2
    #     # left foot stance phase set to default joint pos
    #     sin_pos_l[sin_pos_l > 0] = sin_pos_l[sin_pos_l > 0] * (1 - self.cfg.rewards.target_joint_pos_thd) / (1 + self.cfg.rewards.target_joint_pos_thd) * 0.0
    #     self.ref_dof_pos[:, 1] -= sin_pos_l * scale_1 # FL_thigh_joint
    #     self.ref_dof_pos[:, 2] += sin_pos_l * scale_2 # FL_calf_joint
    #     self.ref_dof_pos[:, 10] -= sin_pos_l * scale_1 # RR_thigh_joint
    #     self.ref_dof_pos[:, 11] += sin_pos_l * scale_2 # RR_calf_joint

    #     sin_pos_r[sin_pos_r < 0] = sin_pos_r[sin_pos_r < 0] * (1 - self.cfg.rewards.target_joint_pos_thd) / (1 + self.cfg.rewards.target_joint_pos_thd) * 0.0
    #     self.ref_dof_pos[:, 4] += sin_pos_r * scale_1 # FR_thigh_joint
    #     self.ref_dof_pos[:, 5] -= sin_pos_r * scale_2 # FR_calf_joint
    #     self.ref_dof_pos[:, 7] += sin_pos_r * scale_1 # RL_thigh_joint
    #     self.ref_dof_pos[:, 8] -= sin_pos_r * scale_2 # RL_calf_joint
 
    def subscribe_viewer_keyboard_events(self):
        super().subscribe_viewer_keyboard_events()
        if self.cfg.env.teleop_mode:
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "forward")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "reverse")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Z, "left")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_C, "right")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "turn_left")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "turn_right")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "stop_x")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "stop_angular")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "stop_y")

    
    def handle_viewer_action_event(self, evt):
        super().handle_viewer_action_event(evt)

        if evt.value <= 0:
            return
        
        if evt.action == "stop_x":
            self.commands[:, 0] = 0
        elif evt.action == "forward":
            self.commands[:, 0] += 0.1
        elif evt.action == "reverse":
            self.commands[:, 0] -= 0.1
        
        elif evt.action == "stop_y":
            self.commands[:, 1] = 0
        elif evt.action == "left":
            self.commands[:, 1] += 0.1
        elif evt.action == "right":
            self.commands[:, 1] -= 0.1

        if evt.action == "stop_angular":
            self.commands[:, 2] = 0
        if evt.action == "turn_left":
            self.commands[:, 2] += 0.1
        elif evt.action == "turn_right":
            self.commands[:, 2] -= 0.1
        # print(evt.action, self.commands)

        
    
    def transform_force_finger_tip_local_to_base(self, forces_local):
        """
        将 finger_tip local 坐标系的力转换到 base 坐标系
        
        Args:
            forces_local: (num_envs, num_fingers, 3) - finger_tip local 坐标系
        
        Returns:
            forces_base: (num_envs, num_fingers, 3) - base 坐标系
        """
        num_fingers = forces_local.shape[1]
        finger_tip_quat = self.rigid_state[:, self.finger_tips_idx, 3:7]
        base_quat_reshaped = self.base_quat.unsqueeze(1).expand(self.num_envs, num_fingers, 4).reshape(self.num_envs*num_fingers,4)
        
        # finger_tip local -> world
        forces_world = quat_apply(
            finger_tip_quat,
            forces_local.reshape(-1, 3)
        ).reshape(self.num_envs, num_fingers, 3)
        
        # world -> base
        forces_base = quat_rotate_inverse(
            base_quat_reshaped,
            forces_world.reshape(-1, 3)
        ).reshape(self.num_envs, num_fingers, 3)
        
        return forces_base

    def transform_force_base_to_world(self, forces_base):
        """
        将 base 坐标系的力转换到 world 坐标系（用于可视化）
        
        Args:
            forces_base: (num_envs, num_fingers, 3) - base 坐标系
        
        Returns:
            forces_world: (num_envs, num_fingers, 3) - world 坐标系
        """
        assert forces_base.shape == (self.num_envs, len(self.finger_tips_idx), 3)
        num_fingers = forces_base.shape[1]
        base_quat_expanded = self.base_quat.unsqueeze(1).expand(self.num_envs, num_fingers, 4)
        
        forces_world = quat_apply(
            base_quat_expanded.reshape(-1, 4),
            forces_base.reshape(-1, 3)
        ).reshape(self.num_envs, num_fingers, 3)
        
        return forces_world

    def transform_pos_base_to_world(self, pos_base):
        """
        将 base 坐标系的位置转换到 world 坐标系（用于可视化）
        
        Args:
            pos_base: (num_envs, num_fingers, 3) - base 坐标系
        
        Returns:
            pos_world: (num_envs, num_fingers, 3) - world 坐标系
        """
        assert pos_base.shape == (self.num_envs, len(self.finger_tips_idx), 3)
        num_fingers = pos_base.shape[1]
        base_quat_expanded = self.base_quat.unsqueeze(1).expand(self.num_envs, num_fingers, 4)
        
        # 将 base 坐标系的位置向量旋转到 world 坐标系的方向
        pos_world_rotated = quat_apply(
            base_quat_expanded.reshape(-1, 4),
            pos_base.reshape(-1, 3)
        ).reshape(self.num_envs, num_fingers, 3)
        
        # 加上 base 在 world 坐标系中的位置，得到绝对位置
        pos_world = pos_world_rotated + self.base_pos.unsqueeze(1)
        
        return pos_world

    #------------ reward functions----------------
    # def _reward_tracking_ee_world(self):

    #     finger_tip_pos_base = quat_rotate_inverse(self.base_quat, self.finger_tips_pos-self.base_pos)
    #     ee_pos_error = torch.sum(torch.abs(finger_tip_pos_base- self.curr_finger_tip_goal_cart), dim=1)
    #     rew = torch.exp(-ee_pos_error/self.cfg.rewards.tracking_ee_sigma * 2)
    #     return rew
    

    # 为每根手指都添加pos_track和orn_track奖励
    def _reward_tracking_ee_position_base_finger0(self):
        """
        Args:
            finger_idx: int, 手指索引 0, 1 2 3 4
        Returns:
            rew: tensor of shape (num_envs,)  奖励值
        """
        finger_idx = 0

        base_quat_reshaped = self.base_quat
        finger_idx_in_isaac = self.finger_tips_idx[finger_idx]

        forces_local = self.forces[:, finger_idx_in_isaac, :3].clone() # 手动施加的外力
        forces_cmd_local = self.current_Fxyz_finger_tips_cmd_local[:, finger_idx, :3].clone()  # shape: (num_envs, num_fingers, 3)
        forces_offset_local = (forces_local + forces_cmd_local)
        
        forces_offset_world = quat_apply(self.rigid_state[:, finger_idx_in_isaac, 3:7].clone(), forces_offset_local)
        forces_offset_base = quat_rotate_inverse(base_quat_reshaped, forces_offset_world)

        curr_ee_goal_cart_base_offset = forces_offset_base / (self.gripper_force_kps[:,finger_idx]) + self.curr_finger_tip_goal_cart[:, finger_idx].clone()   
        

        finger_tip_pos_world = self.rigid_state[:, finger_idx_in_isaac, :3].clone()
        finger_tip_pos_base = quat_rotate_inverse(
            base_quat_reshaped,
            (finger_tip_pos_world - self.base_pos)
        ).reshape(self.num_envs, 3)
        
        ee_pos_error_per_finger = torch.sum(torch.abs(finger_tip_pos_base - curr_ee_goal_cart_base_offset), dim=-1) # shape: (num_envs, num_fingers)
        
        # 自适应 sigma：误差越小，sigma 越小（奖励曲线更陡），鼓励精确贴合
        base_sigma = self.cfg.rewards.scales.tracking_ee_position_base_finger0
        sigma_scale = 0.3 + 0.7 * (ee_pos_error_per_finger / (ee_pos_error_per_finger + 1.0))  # err→0 时约 0.3，err 大时→1.0
        adaptive_sigma = torch.clamp(base_sigma * sigma_scale, min=base_sigma * 0.3)

        rew = torch.exp(-ee_pos_error_per_finger / (adaptive_sigma + 1e-6) * 2)
        return rew

    def _reward_tracking_ee_position_base_finger1(self):
        """
        Args:
            finger_idx: int, 手指索引 0, 1 2 3 4
        Returns:
            rew: tensor of shape (num_envs,)  奖励值
        """
        finger_idx = 1

        base_quat_reshaped = self.base_quat
        finger_idx_in_isaac = self.finger_tips_idx[finger_idx]

        forces_local = self.forces[:, finger_idx_in_isaac, :3].clone() # 手动施加的外力
        forces_cmd_local = self.current_Fxyz_finger_tips_cmd_local[:, finger_idx, :3].clone()  # shape: (num_envs, num_fingers, 3)
        forces_offset_local = (forces_local + forces_cmd_local)
        
        forces_offset_world = quat_apply(self.rigid_state[:, finger_idx_in_isaac, 3:7].clone(), forces_offset_local)
        forces_offset_base = quat_rotate_inverse(base_quat_reshaped, forces_offset_world)
        
        curr_ee_goal_cart_base_offset = forces_offset_base / (self.gripper_force_kps[:,finger_idx]) + self.curr_finger_tip_goal_cart[:, finger_idx].clone()   
        

        finger_tip_pos_world = self.rigid_state[:, finger_idx_in_isaac, :3].clone()
        finger_tip_pos_base = quat_rotate_inverse(
            base_quat_reshaped,
            (finger_tip_pos_world - self.base_pos)
        ).reshape(self.num_envs, 3)
        
        ee_pos_error_per_finger = torch.sum(torch.abs(finger_tip_pos_base - curr_ee_goal_cart_base_offset), dim=-1) # shape: (num_envs, num_fingers)
        
        # 自适应 sigma：误差越小，sigma 越小（奖励曲线更陡），鼓励精确贴合
        base_sigma = self.cfg.rewards.scales.tracking_ee_position_base_finger1
        sigma_scale = 0.3 + 0.7 * (ee_pos_error_per_finger / (ee_pos_error_per_finger + 1.0))  # err→0 时约 0.3，err 大时→1.0
        adaptive_sigma = torch.clamp(base_sigma * sigma_scale, min=base_sigma * 0.3)

        rew = torch.exp(-ee_pos_error_per_finger / (adaptive_sigma + 1e-6) * 2)
        return rew
    
    def _reward_tracking_ee_position_base_finger2(self):
        """
        Args:
            finger_idx: int, 手指索引 0, 1 2 3 4
        Returns:
            rew: tensor of shape (num_envs,)  奖励值
        """
        finger_idx = 2

        base_quat_reshaped = self.base_quat
        finger_idx_in_isaac = self.finger_tips_idx[finger_idx]

        forces_local = self.forces[:, finger_idx_in_isaac, :3].clone() # 手动施加的外力
        forces_cmd_local = self.current_Fxyz_finger_tips_cmd_local[:, finger_idx, :3].clone()  # shape: (num_envs, num_fingers, 3)
        forces_offset_local = (forces_local + forces_cmd_local)
        
        forces_offset_world = quat_apply(self.rigid_state[:, finger_idx_in_isaac, 3:7].clone(), forces_offset_local)
        forces_offset_base = quat_rotate_inverse(base_quat_reshaped, forces_offset_world)
        
        curr_ee_goal_cart_base_offset = forces_offset_base / (self.gripper_force_kps[:,finger_idx]) + self.curr_finger_tip_goal_cart[:, finger_idx].clone()   
        

        finger_tip_pos_world = self.rigid_state[:, finger_idx_in_isaac, :3].clone()
        finger_tip_pos_base = quat_rotate_inverse(
            base_quat_reshaped,
            (finger_tip_pos_world - self.base_pos)
        ).reshape(self.num_envs, 3)
        
        ee_pos_error_per_finger = torch.sum(torch.abs(finger_tip_pos_base - curr_ee_goal_cart_base_offset), dim=-1) # shape: (num_envs, num_fingers)
        
        # 自适应 sigma：误差越小，sigma 越小（奖励曲线更陡），鼓励精确贴合
        base_sigma = self.cfg.rewards.scales.tracking_ee_position_base_finger2
        sigma_scale = 0.3 + 0.7 * (ee_pos_error_per_finger / (ee_pos_error_per_finger + 1.0))  # err→0 时约 0.3，err 大时→1.0
        adaptive_sigma = torch.clamp(base_sigma * sigma_scale, min=base_sigma * 0.3)

        rew = torch.exp(-ee_pos_error_per_finger / (adaptive_sigma + 1e-6) * 2)
        return rew
    
    def _reward_tracking_ee_position_base_finger3(self):
        """
        Args:
            finger_idx: int, 手指索引 0, 1 2 3 4
        Returns:
            rew: tensor of shape (num_envs,)  奖励值
        """
        finger_idx = 3

        base_quat_reshaped = self.base_quat
        finger_idx_in_isaac = self.finger_tips_idx[finger_idx]

        forces_local = self.forces[:, finger_idx_in_isaac, :3].clone() # 手动施加的外力
        forces_cmd_local = self.current_Fxyz_finger_tips_cmd_local[:, finger_idx, :3].clone()  # shape: (num_envs, num_fingers, 3)
        forces_offset_local = (forces_local + forces_cmd_local)
        
        forces_offset_world = quat_apply(self.rigid_state[:, finger_idx_in_isaac, 3:7].clone(), forces_offset_local)
        forces_offset_base = quat_rotate_inverse(base_quat_reshaped, forces_offset_world)
        
        curr_ee_goal_cart_base_offset = forces_offset_base / (self.gripper_force_kps[:,finger_idx]) + self.curr_finger_tip_goal_cart[:, finger_idx].clone()   
        

        finger_tip_pos_world = self.rigid_state[:, finger_idx_in_isaac, :3].clone()
        finger_tip_pos_base = quat_rotate_inverse(
            base_quat_reshaped,
            (finger_tip_pos_world - self.base_pos)
        ).reshape(self.num_envs, 3)
        
        ee_pos_error_per_finger = torch.sum(torch.abs(finger_tip_pos_base - curr_ee_goal_cart_base_offset), dim=-1) # shape: (num_envs, num_fingers)
        
        # 自适应 sigma：误差越小，sigma 越小（奖励曲线更陡），鼓励精确贴合
        base_sigma = self.cfg.rewards.scales.tracking_ee_position_base_finger3
        sigma_scale = 0.3 + 0.7 * (ee_pos_error_per_finger / (ee_pos_error_per_finger + 1.0))  # err→0 时约 0.3，err 大时→1.0
        adaptive_sigma = torch.clamp(base_sigma * sigma_scale, min=base_sigma * 0.3)

        rew = torch.exp(-ee_pos_error_per_finger / (adaptive_sigma + 1e-6) * 2)
        return rew
    
    def _reward_tracking_ee_position_base_finger4(self):
        """
        Args:
            finger_idx: int, 手指索引 0, 1 2 3 4
        Returns:
            rew: tensor of shape (num_envs,)  奖励值
        """
        finger_idx = 4

        base_quat_reshaped = self.base_quat
        finger_idx_in_isaac = self.finger_tips_idx[finger_idx]

        forces_local = self.forces[:, finger_idx_in_isaac, :3].clone() # 手动施加的外力
        forces_cmd_local = self.current_Fxyz_finger_tips_cmd_local[:, finger_idx, :3].clone()  # shape: (num_envs, num_fingers, 3)
        forces_offset_local = (forces_local + forces_cmd_local)
        
        forces_offset_world = quat_apply(self.rigid_state[:, finger_idx_in_isaac, 3:7].clone(), forces_offset_local)
        forces_offset_base = quat_rotate_inverse(base_quat_reshaped, forces_offset_world)
        
        curr_ee_goal_cart_base_offset = forces_offset_base / (self.gripper_force_kps[:,finger_idx]) + self.curr_finger_tip_goal_cart[:, finger_idx].clone()   
        

        finger_tip_pos_world = self.rigid_state[:, finger_idx_in_isaac, :3].clone()
        finger_tip_pos_base = quat_rotate_inverse(
            base_quat_reshaped,
            (finger_tip_pos_world - self.base_pos)
        ).reshape(self.num_envs, 3)
        
        ee_pos_error_per_finger = torch.sum(torch.abs(finger_tip_pos_base - curr_ee_goal_cart_base_offset), dim=-1) # shape: (num_envs, num_fingers)
        
        # 自适应 sigma：误差越小，sigma 越小（奖励曲线更陡），鼓励精确贴合
        base_sigma = self.cfg.rewards.scales.tracking_ee_position_base_finger4
        sigma_scale = 0.3 + 0.7 * (ee_pos_error_per_finger / (ee_pos_error_per_finger + 1.0))  # err→0 时约 0.3，err 大时→1.0
        adaptive_sigma = torch.clamp(base_sigma * sigma_scale, min=base_sigma * 0.3)

        rew = torch.exp(-ee_pos_error_per_finger / (adaptive_sigma + 1e-6) * 2)
        return rew
        
    def _reward_tracking_ee_orientation_6d_base(self):
        """
        6D旋转追踪奖励（支持多手指版本）
        对每个手指分别计算姿态跟踪奖励，然后取平均
        
        Returns:
            reward: tensor of shape (num_envs,)  奖励值
        """
        num_fingers = len(self.finger_tips_idx)
        # 获取当前指尖姿态（世界坐标系）
        base_quat_reshaped = self.base_quat.unsqueeze(1).expand(self.num_envs, num_fingers, 4).reshape(self.num_envs*num_fingers,4)
        finger_tip_orn_quat_world = self.rigid_state[:, self.finger_tips_idx, 3:7].clone()

        # 转换到base坐标系
        finger_tip_orn_quat_base = quat_mul(quat_conjugate(base_quat_reshaped), finger_tip_orn_quat_world.reshape(self.num_envs*num_fingers,4)).reshape(self.num_envs, num_fingers, 4)
        # # 归一化四元数（防止NaN）
        # finger_tip_orn_quat_base = finger_tip_orn_quat_base / (torch.norm(finger_tip_orn_quat_base, dim=-1, keepdim=True) + 1e-8)
        
        # 获取目标姿态（已经在base坐标系）shape: (num_envs, num_fingers, 4)
        curr_finger_tip_goal_quat_base = self.curr_finger_tip_goal_orn.clone()
        # # 归一化四元数（防止NaN）
        # curr_finger_tip_goal_quat_base = curr_finger_tip_goal_quat_base / (torch.norm(curr_finger_tip_goal_quat_base, dim=-1, keepdim=True) + 1e-8)

        # 转换为6D表示 shape: (num_envs, num_fingers, 6)
        mat6d_ee_base = quat_to_mat6d(finger_tip_orn_quat_base)
        mat6d_target_base = quat_to_mat6d(curr_finger_tip_goal_quat_base)
        
        # 计算每个手指的6D误差的L2距离平方和 shape: (num_envs, num_fingers)
        diff_per_finger = torch.sum((mat6d_ee_base - mat6d_target_base)**2, dim=-1)
        
        # 自适应 sigma：误差越小，sigma 越小（奖励曲线更陡），鼓励精确贴合
        base_sigma = self.cfg.rewards.scales.tracking_ee_position_base_finger4
        sigma_scale = 0.3 + 0.7 * (diff_per_finger / (diff_per_finger + 1.0))  # err→0 时约 0.3，err 大时→1.0
        adaptive_sigma = torch.clamp(base_sigma * sigma_scale, min=base_sigma * 0.3)

        rew = torch.exp(-diff_per_finger / (adaptive_sigma + 1e-6) * 2)
        rew = torch.mean(rew, dim=-1)
        return rew
        

    def _reward_tracking_ee_force_base(self):
        """
        位置和力跟踪奖励（支持多手指版本）
        对每个手指分别计算位置跟踪奖励，然后取平均
        
        Returns:
            reward: tensor of shape (num_envs,)  奖励值
        """
        num_fingers = len(self.finger_tips_idx)
        base_quat_reshaped = self.base_quat.unsqueeze(1).expand(self.num_envs, num_fingers, 4).reshape(self.num_envs*num_fingers,4)
        # 获取所有手指的传感器力 shape: (num_envs, num_fingers, 3)
        # forces_local = self.sensors_forces[:, :, :3]  # shape: (num_envs, num_fingers, 3)
        forces_local = self.forces[:, self.finger_tips_idx, :3].clone() # 手动施加的外力
        forces_cmd_local = self.current_Fxyz_finger_tips_cmd_local  # shape: (num_envs, num_fingers, 3)
        forces_offset_local = (forces_local + forces_cmd_local)

        # forces_offset_global = quat_apply(self.rigid_state[:,self.finger_tips_idx,3:7].clone(), forces_offset_local)
        # # forces_offset_global = quat_apply(self.base_quat, forces_offset_local)
        # forces_offset_base = quat_rotate_inverse(
        #     self.base_quat.unsqueeze(1).expand(self.num_envs, num_fingers, 4).reshape(self.num_envs*num_fingers,4), 
        #     forces_offset_global.reshape(self.num_envs*num_fingers,3)
        #     ).reshape(self.num_envs, num_fingers, 3)
        forces_offset_base = self.transform_force_finger_tip_local_to_base(forces_offset_local)

        # 计算考虑力偏移的目标位置 shape: (num_envs, num_fingers, 3)
        curr_ee_goal_cart_base_offset = forces_offset_base / (self.gripper_force_kps) + self.curr_finger_tip_goal_cart
       
        # 获取当前所有手指的位置 shape: (num_envs, num_fingers, 3)
        finger_tip_pos_world = self.rigid_state[:, self.finger_tips_idx, :3].clone()
        finger_tip_pos_base = quat_rotate_inverse(
            base_quat_reshaped,
            (finger_tip_pos_world - self.base_pos.unsqueeze(1)).reshape(self.num_envs*num_fingers,3)
        ).reshape(self.num_envs, num_fingers, 3)
        
        # 计算每个手指的位置误差 shape: (num_envs, num_fingers)
        ee_pos_error_per_finger = torch.sum(torch.abs(finger_tip_pos_base - curr_ee_goal_cart_base_offset), dim=-1) # shape: (num_envs, num_fingers)
        
        # 自适应 sigma：对每个手指分别计算
        base_sigma = self.cfg.rewards.tracking_ee_sigma
        sigma_scale = 0.3 + 0.7 * (ee_pos_error_per_finger / (ee_pos_error_per_finger + 1.0))
        adaptive_sigma = torch.clamp(base_sigma * sigma_scale, min=base_sigma * 0.3)

        # 计算每个手指的奖励 shape: (num_envs, num_fingers)
        rew_per_finger = torch.exp(-ee_pos_error_per_finger / (adaptive_sigma + 1e-6) * 2)
        
        # 对所有手指取平均（也可以使用最小奖励或其他聚合方式）
        rew = torch.mean(rew_per_finger, dim=-1)  # shape: (num_envs,)
        
        return rew
    
    def _reward_finger_tracking_consistency(self):
        """
        手指跟踪一致性奖励：鼓励所有手指的跟踪误差相近
        避免某些手指跟踪很好，而其他手指跟踪很差的情况
        
        Returns:
            reward: tensor of shape (num_envs,)  奖励值
        """
        num_fingers = len(self.finger_tips_idx)
        
        # 计算每个手指的位置跟踪误差
        finger_tip_pos_world = self.rigid_state[:, self.finger_tips_idx, :3].clone()
        finger_tip_pos_base = quat_rotate_inverse(
            self.base_quat.unsqueeze(1).expand(-1, num_fingers, -1).reshape(-1, 4),
            (finger_tip_pos_world - self.base_pos.unsqueeze(1)).reshape(-1, 3)
        ).view(self.num_envs, num_fingers, 3)
        
        # 计算位置误差 shape: (num_envs, num_fingers)
        pos_error_per_finger = torch.norm(finger_tip_pos_base - self.curr_finger_tip_goal_cart, dim=-1)
        
        # 计算误差的标准差（一致性指标）
        mean_error = torch.mean(pos_error_per_finger, dim=1, keepdim=True)
        error_std = torch.std(pos_error_per_finger, dim=1)
        
        # 奖励：标准差越小，一致性越好
        consistency_sigma = 0.05  # 可配置参数
        rew = torch.exp(-error_std / (consistency_sigma + 1e-6))
        
        return rew
    
    # def _reward_tracking_ee_force(self):
    #     """纯力跟踪奖励"""
    #     force_error = torch.norm(
    #         self.sensors_forces[:, 0, :3] - self.commands[:, INDEX_TIP_FORCE_X:(INDEX_TIP_FORCE_Z+1)],
    #         dim=1
    #     )
    #     return torch.exp(-force_error / 5.0)  # 5N误差衰减

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel)[:, :], dim=1)

    def _reward_contact_stability(self):
        """接触稳定性（力的方向稳定性）"""
        force_norm = torch.norm(self.sensors_forces[:, 0, :3], dim=1, keepdim=True) + 1e-6
        force_direction = self.sensors_forces[:, 0, :3] / force_norm
        
        if not hasattr(self, 'last_force_direction'):
            self.last_force_direction = force_direction.clone()
        
        direction_change = torch.norm(force_direction - self.last_force_direction, dim=1)
        self.last_force_direction = force_direction.clone()
        return torch.exp(-direction_change / 0.2)
   
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel)[:, :] / self.dt), dim=1)
    
   
    # def _reward_action_rate(self):
    #     # Penalize changes in actions
    #     return torch.sum(torch.square(self.last_actions - self.actions)[:, 4:8], dim=1)
    
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits[:,:], dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.)[:, :], dim=1)

   
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions)[:, :], dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions)[:, :], dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions)[:, :], dim=1)
        return term_1 + term_2 + term_3
    
    
    def _reward_ee_force_x(self):
        # 获取传感器力（finger_tip local）
        sensor_forces_local = self.sensors_forces[:, :, :3]  # (num_envs, num_fingers, 3)
        
        # 转换到 base 坐标系
        sensor_forces_base = self.transform_force_finger_tip_local_to_base(sensor_forces_local)
        
        # 从 commands 获取 F_cmd（已经是 base 坐标系）
        F_cmd_base = self.commands[:, :, INDEX_TIP_FORCE_X:(INDEX_TIP_FORCE_Z+1)]  # (num_envs, num_fingers, 3)
        
        # 计算误差（都在 base 坐标系）
        force_magn_meas = sensor_forces_base[:, :, 0]  # (num_envs, num_fingers)
        force_magn_cmd = F_cmd_base[:, :, 0]  # (num_envs, num_fingers)
        force_magn_error = torch.abs(force_magn_meas - force_magn_cmd).mean(dim=1)  # (num_envs,) - 对所有手指取平均
        
        force_magn_coeff = self.cfg.rewards.sigma_force
        return torch.exp(-force_magn_coeff * force_magn_error)
    
    def _reward_ee_force_y(self):
        # 获取传感器力（finger_tip local）
        sensor_forces_local = self.sensors_forces[:, :, :3]  # (num_envs, num_fingers, 3)
        
        # 转换到 base 坐标系
        sensor_forces_base = self.transform_force_finger_tip_local_to_base(sensor_forces_local)
        
        # 从 commands 获取 F_cmd（已经是 base 坐标系）
        F_cmd_base = self.commands[:, :, INDEX_TIP_FORCE_X:(INDEX_TIP_FORCE_Z+1)]  # (num_envs, num_fingers, 3)
        
        # 计算误差（都在 base 坐标系）
        force_magn_meas = sensor_forces_base[:, :, 1]  # (num_envs, num_fingers)
        force_magn_cmd = F_cmd_base[:, :, 1]  # (num_envs, num_fingers)
        force_magn_error = torch.abs(force_magn_meas - force_magn_cmd).mean(dim=1)  # (num_envs,) - 对所有手指取平均
        
        force_magn_coeff = self.cfg.rewards.sigma_force
        return torch.exp(-force_magn_coeff * force_magn_error)
    
    def _reward_ee_force_z(self):
        # 获取传感器力（finger_tip local）
        sensor_forces_local = self.sensors_forces[:, :, :3]  # (num_envs, num_fingers, 3)
        
        # 转换到 base 坐标系
        sensor_forces_base = self.transform_force_finger_tip_local_to_base(sensor_forces_local)
        
        # 从 commands 获取 F_cmd（已经是 base 坐标系）
        F_cmd_base = self.commands[:, :, INDEX_TIP_FORCE_X:(INDEX_TIP_FORCE_Z+1)]  # (num_envs, num_fingers, 3)
        
        # 计算误差（都在 base 坐标系）
        force_magn_meas = sensor_forces_base[:, :, 2]  # (num_envs, num_fingers)
        force_magn_cmd = F_cmd_base[:, :, 2]  # (num_envs, num_fingers)
        force_magn_error = torch.abs(force_magn_meas - force_magn_cmd).mean(dim=1)  # (num_envs,) - 对所有手指取平均
        
        force_magn_coeff = self.cfg.rewards.sigma_force
        return torch.exp(-force_magn_coeff * force_magn_error)