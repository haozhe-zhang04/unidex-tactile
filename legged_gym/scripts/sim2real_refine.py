import os
import sys
unitree_rl_gym_path = os.path.abspath(__file__ + "../../../../")
sys.path.append(unitree_rl_gym_path)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from wuji.wuji_hand import wuji_node
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin

from b2_gym_learn.ppo_cse_pf.actor_critic import ActorCritic
from b2_gym_learn.ppo_cse_pf.ppo import PPO
import torch
import torch.nn.functional as F


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
    assert R.dim() == 3 and R.shape == (5, 3, 3)
    return R

def quat_to_mat6d(quat):

    assert quat.dim() == 2 and quat.shape == (5, 4)
    rot_matrices = quat_to_rotation_matrix(quat) # shape: (num_envs, num_fingers, 3, 3)
    
    # 2. 提取前两列
    # col1: X轴方向 (num_envs, num_fingers, 3)
    col1 = rot_matrices[:, :, 0] 
    # col2: Y轴方向 (num_envs, num_fingers, 3)
    col2 = rot_matrices[:, :, 1]
    
    # 3. 拼接成 6D 向量 (num_envs, num_fingers, 6)
    mat6d = torch.cat([col1, col2], dim=-1)
    assert mat6d.dim() == 2 and mat6d.shape == (5, 6)
    return mat6d

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

class Sim2Real:

    def __init__(self,urdf_path,mesh_dir,task,config):
        self.hand = wuji_node(serial_number="337238723233", is_use_csp=False)
        robot = RobotWrapper.BuildFromURDF(urdf_path, mesh_dir)

        self.pinocchio_model = robot.model
        self.pinocchio_data = self.pinocchio_model.createData()
        self.pinocchio_tips_idx = [
                            self.pinocchio_model.getFrameId("finger1_tip_link"),
                            self.pinocchio_model.getFrameId("finger2_tip_link"),
                            self.pinocchio_model.getFrameId("finger3_tip_link"),
                            self.pinocchio_model.getFrameId("finger4_tip_link"),
                            self.pinocchio_model.getFrameId("finger5_tip_link"),
                        ]
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.commands = torch.zeros((5, 13), device=self.device, dtype=torch.float32, requires_grad=False)

        self._init_policy()

    def _init_policy(self):
        num_obs = self.config["env"]["num_obs"]
        num_privileged_obs = self.config["env"]["num_privileged_obs"]
        num_single_obs = self.config["env"]["num_single_obs"]
        num_actions = self.config["env"]["num_actions"]
        actor_hidden_dims = self.config["policy"]["actor_hidden_dims"]
        critic_hidden_dims = self.config["policy"]["critic_hidden_dims"]
        activation=self.config["policy"]["activation"]
        init_noise_std=self.config["policy"]["init_noise_std"]

        model_state_dict = self.config["model"]["model_state_dict"]
        actor_critic = ActorCritic(num_obs, 
                            num_privileged_obs, 
                            num_single_obs, 
                            num_actions, 
                            actor_hidden_dims, 
                            critic_hidden_dims, 
                            activation, 
                            init_noise_std)

        actor_critic.load_state_dict(torch.load(model_state_dict)["model_state_dict"])
        actor_critic.to(self.device)

        actor_critic.eval() # switch to evaluation mode (dropout for example))
        self.policy = actor_critic.act_inference

    def _normalize_pos(self, pos):
        assert pos.shape ==(5, 3)
        normalized_pos = torch.zeros_like(pos)
        finger_tip_pos_x_min = self.config["obs_scales"]["finger_tip_pos_x_min"]
        finger_tip_pos_x_max = self.config["obs_scales"]["finger_tip_pos_x_max"]
        finger_tip_pos_y_min = self.config["obs_scales"]["finger_tip_pos_y_min"]
        finger_tip_pos_y_max = self.config["obs_scales"]["finger_tip_pos_y_max"]
        finger_tip_pos_z_min = self.config["obs_scales"]["finger_tip_pos_z_min"]
        finger_tip_pos_z_max = self.config["obs_scales"]["finger_tip_pos_z_max"]
        for i in range(5):
            normalized_pos[i,0:1] = 2 * (pos[i,0:1] - finger_tip_pos_x_min[i]) / (finger_tip_pos_x_max[i] - finger_tip_pos_x_min[i]) - 1
            normalized_pos[i,1:2] = 2 * (pos[i,1:2] - finger_tip_pos_y_min[i]) / (finger_tip_pos_y_max[i] - finger_tip_pos_y_min[i]) - 1
            normalized_pos[i,2:3] = 2 * (pos[i,2:3] - finger_tip_pos_z_min[i]) / (finger_tip_pos_z_max[i] - finger_tip_pos_z_min[i]) - 1
        return normalized_pos
    
    def set_commands(self, command):
        self.commands[:, INDEX_TIP_FORCE_X:INDEX_TIP_FORCE_Z+1] = command[:, INDEX_TIP_FORCE_X:INDEX_TIP_FORCE_Z+1]
        self.commands[:, INDEX_TIP_TORQUE_X:INDEX_TIP_TORQUE_Z+1] = command[:, INDEX_TIP_TORQUE_X:INDEX_TIP_TORQUE_Z+1]
        self.commands[:, INDEX_TIP_POS_X_CMD:INDEX_TIP_POS_Z_CMD+1] = command[:, INDEX_TIP_POS_X_CMD:INDEX_TIP_POS_Z_CMD+1]
        self.commands[:, INDEX_TIP_ORIENTATION_X_CMD:INDEX_TIP_ORIENTATION_W_CMD+1] = command[:, INDEX_TIP_ORIENTATION_X_CMD:INDEX_TIP_ORIENTATION_W_CMD+1]
    
    def set_pose(self, pose):
        self.hand.set_all_joints_pos(pose)

    def get_joint_pose(self):
        return self.hand.get_pose()

    def get_certain_finger_tip_pos(self, joint_pos):
        assert joint_pos.shape == (20,)
        pin.forwardKinematics(self.pinocchio_model, self.pinocchio_data, joint_pos)
        pin.updateFramePlacements(self.pinocchio_model, self.pinocchio_data)

        for finger_idx in range(len(self.pinocchio_tips_idx)):
            pinocchio_tip_frame_idx = self.pinocchio_tips_idx[finger_idx]
            pos = self.pinocchio_data.oMf[pinocchio_tip_frame_idx].translation
            orn = self.pinocchio_data.oMf[pinocchio_tip_frame_idx].rotation
            quat = mat3x3_to_xyzw(torch.from_numpy(orn).to(self.device, dtype=torch.float32))
            self.commands[finger_idx, INDEX_TIP_POS_X_CMD:INDEX_TIP_POS_Z_CMD+1] = torch.from_numpy(pos).to(self.device, dtype=torch.float32)
            self.commands[finger_idx, INDEX_TIP_ORIENTATION_X_CMD:INDEX_TIP_ORIENTATION_W_CMD+1] = quat


    def get_current_finger_tip_pos(self):
        """
        return x,y,z,quat
        """
        joint_pos = self.get_joint_pose()

        pin.forwardKinematics(self.pinocchio_model, self.pinocchio_data, joint_pos)
        pin.updateFramePlacements(self.pinocchio_model, self.pinocchio_data)

        tip_pos = torch.zeros(5, 3, device=self.device)
        tip_orn = torch.zeros(5, 4, device=self.device)
        for finger_idx in range(len(self.pinocchio_tips_idx)):
            pinocchio_tip_frame_idx = self.pinocchio_tips_idx[finger_idx]
            # translation
            tip_goal_cart_base = self.pinocchio_data.oMf[pinocchio_tip_frame_idx].translation.copy()
            tip_goal_rotation_base = self.pinocchio_data.oMf[pinocchio_tip_frame_idx].rotation.copy() # 3*3 matrix
            R_tensor = torch.from_numpy(tip_goal_rotation_base).to(self.device, dtype=torch.float32)
            R_tensor = R_tensor.unsqueeze(0)
            tip_goal_orn_base = mat3x3_to_xyzw(R_tensor).squeeze(0)  # (1, 4) -> (4,)
            tip_pos[finger_idx] = torch.from_numpy(tip_goal_cart_base).to(self.device, dtype=torch.float32)
            tip_orn[finger_idx] = tip_goal_orn_base
        return tip_pos, tip_orn
    
    def get_obs(self):
        dof_pos = torch.from_numpy(self.get_joint_pose()).to(self.device, dtype=torch.float32)
        finger_tip_pos, finger_tip_orn = self.get_current_finger_tip_pos()
        finger_tip_pos_normalized = self._normalize_pos(finger_tip_pos)
        finger_tip_orn_6d_base = quat_to_mat6d(finger_tip_orn)

        curr_finger_tip_goal_orn_6d_base = quat_to_mat6d(self.commands[:, INDEX_TIP_ORIENTATION_X_CMD:INDEX_TIP_ORIENTATION_W_CMD+1])

        finger_tip_orn_6d_error = finger_tip_orn_6d_base - curr_finger_tip_goal_orn_6d_base
        forces_base = torch.zeros(5, 3, device=self.device)
        forces_error = torch.zeros(5, 3, device=self.device)
        print(self.commands[:, INDEX_TIP_POS_X_CMD:INDEX_TIP_POS_Z_CMD+1])
        pos_cmd_normalized = self._normalize_pos(self.commands[:, INDEX_TIP_POS_X_CMD:INDEX_TIP_POS_Z_CMD+1])
        
       
        obs = torch.cat([dof_pos* self.config["obs_scales"]["dof_pos"], 
                    finger_tip_pos_normalized.flatten(), 
                    finger_tip_orn_6d_base.flatten(), 
                    finger_tip_orn_6d_error.flatten() * self.config["obs_scales"]["orn_error"],
                    forces_base.flatten() * self.config["obs_scales"]["sensor_force"],
                    forces_error.flatten() * self.config["obs_scales"]["force_error"],
                    pos_cmd_normalized.flatten(),
                    self.commands[:, INDEX_TIP_ORIENTATION_X_CMD:INDEX_TIP_ORIENTATION_W_CMD+1].flatten() * self.config["obs_scales"]["orientation_cmd"],
                    self.commands[:, INDEX_TIP_FORCE_X:INDEX_TIP_FORCE_Z+1].flatten() * self.config["obs_scales"]["force_cmd"]], dim=-1)
        obs_full = {"obs":obs.unsqueeze(0)}
        return obs_full

    def get_action(self, obs):
        return self.policy(obs)


def main():
    urdf_path = "/home/hz01/haozhe_workspace/UniFP/wujihand-urdf/urdf/right.urdf"
    mesh_dir = os.path.dirname(urdf_path)
    task = "wuji_pos_force"
    config = {
        "env": {
            "num_obs": 32*175,
            "num_privileged_obs": 3*240,
            "num_single_obs": 175,
            "num_actions": 20,
        },
        "policy": {
            "activation": "elu",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
        },
        "model": {
            "model_state_dict": "/home/hz01/haozhe_workspace/UniFP/logs/wuji_pos_force/Dec27_20-20-03_/model_400.pt"
        },
        "obs_scales":{
            "dof_pos": 1.0,
            "finger_tip_pos_x_min" : [0.01051468, -0.01408456,-0.02309369, -0.01941702,-0.01198453],      
            "finger_tip_pos_x_max" : [0.08438421, 0.096,      0.09327604 , 0.09595964, 0.10003495],
            "finger_tip_pos_y_min" : [-0.04537681,-0.03664682,-0.03951727, -0.06925806, -0.09866672],
            "finger_tip_pos_y_max" : [0.12591782, 0.05499584, 0.05443658, 0.02417522,-0.00785031],
            "finger_tip_pos_z_min" : [0.05079299, 0.03804432, 0.03675804, 0.03169484, 0.02277947],
            "finger_tip_pos_z_max" : [0.13120497, 0.19527645, 0.19265326, 0.1871634, 0.1754161],
        
            "sensor_force" : 0.1,
            "pose_error" : 5.0,
            "orn_error" : 2.0,
            "force_error" : 0.1,
            "force_cmd" : 0.1,
            "orientation_cmd" : 1.0,
        },
    }
    sim2real = Sim2Real(urdf_path=urdf_path, mesh_dir=mesh_dir, task = task, config = config)
    sim2real.set_pose(np.zeros(20))
    import pdb; pdb.set_trace()

    sim2real.get_certain_finger_tip_pos(np.zeros(20))
    import time
    while True:
        obs = sim2real.get_obs()

        action = sim2real.get_action(obs)
        action_np = action.detach().cpu().numpy()
        curr_action = sim2real.get_joint_pose()
        new_action = curr_action + action_np
        print(new_action)
        sim2real.set_pose(new_action.squeeze(0))
        time.sleep(0.1)
        import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()