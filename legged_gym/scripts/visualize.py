import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin
import numpy as np
import os
import torch
def visualize_finger_workspace(model, data, tip_name="finger1_tip_link", num_samples=10000):
    """
    通过随机采样关节空间来可视化特定手指的可达空间点云
    """

    fid = model.getFrameId(tip_name)

    # 2. 获取关节限位 (nq 是广义坐标维度)
    lower_limits = model.lowerPositionLimit
    upper_limits = model.upperPositionLimit

    # 处理一些没有限位的关节（如连续旋转关节可能被设为 -1e10 to 1e10）
    # 实际应用中，Pinocchio 从 URDF 读取的限位通常是准确的
    
    workspace_points = np.zeros((num_samples, 3))

    print(f"正在为 {tip_name} 采样 {num_samples} 个点...")
    for i in range(num_samples):
        # 在限位内均匀随机采样
        q_rand = np.random.uniform(lower_limits, upper_limits)
        
        # 执行正向运动学
        pin.forwardKinematics(model, data, q_rand)
        pin.updateFramePlacements(model, data)
        
        # 记录末端在世界坐标系（或机器人基座系）下的位置
        workspace_points[i] = data.oMf[fid].translation.copy()

    # 3. 计算边界框 (Bounding Box)，这对你 PPO 采样非常有用
    min_bound = np.min(workspace_points, axis=0)
    max_bound = np.max(workspace_points, axis=0)
    print(f"\n[可达空间边界统计]")
    print(f"X range: [{min_bound[0]:.3f}, {max_bound[0]:.3f}]")
    print(f"Y range: [{min_bound[1]:.3f}, {max_bound[1]:.3f}]")
    print(f"Z range: [{min_bound[2]:.3f}, {max_bound[2]:.3f}]")

    # 4. 绘图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点云
    scatter = ax.scatter(workspace_points[:, 0], 
                         workspace_points[:, 1], 
                         workspace_points[:, 2], 
                         c=workspace_points[:, 2], # 按高度着色
                         cmap='viridis', 
                         s=1, 
                         alpha=0.3)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Reachable Workspace: {tip_name}')
    
    # 设置等比例坐标轴
    max_range = np.array([max_bound[0]-min_bound[0], 
                          max_bound[1]-min_bound[1], 
                          max_bound[2]-min_bound[2]]).max() / 2.0
    mid_x = (max_bound[0]+min_bound[0]) * 0.5
    mid_y = (max_bound[1]+min_bound[1]) * 0.5
    mid_z = (max_bound[2]+min_bound[2]) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.colorbar(scatter, label='Z height')
    plt.show()

    return workspace_points, min_bound, max_bound

# --- 调用示例 ---
# 假设你已经定义好了 model 和 data

urdf_path = "wujihand-urdf/urdf/right.urdf"
mesh_dir  = os.path.dirname(urdf_path)

# 自动根据 URDF 判断是否是 fixed base 或 free-flyer
robot = RobotWrapper.BuildFromURDF(urdf_path, mesh_dir)

model = robot.model
data = model.createData()
points, b_min, b_max = visualize_finger_workspace(model, data, "finger1_tip_link")