from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import math

gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.dt = 0.005

compute_device_id = 0
graphics_device_id = 0
physics_engine = gymapi.SIM_PHYSX
sim = gym.create_sim(compute_device_id, graphics_device_id, physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load asset
asset_root = "/home/hz01/haozhe_workspace/UniFP/wujihand-urdf"
asset_file = "right.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = False
asset_options.use_mesh_materials = True

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)


# get array of DOF names
dof_names = gym.get_asset_dof_names(asset)

# get array of DOF properties
dof_props = gym.get_asset_dof_properties(asset)

# create an array of DOF states that will be used to update the actors
num_dofs = gym.get_asset_dof_count(asset)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

# get list of DOF types
dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

# get the position slice of the DOF state array
dof_positions = dof_states['pos']

# get the limit-related slices of the DOF properties array
stiffnesses = dof_props['stiffness']
dampings = dof_props['damping']
armatures = dof_props['armature']
has_limits = dof_props['hasLimits']
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']

defaults = np.zeros(num_dofs)
speeds = np.zeros(num_dofs)

dof_positions = defaults

# set up the env grid
num_envs = 36
num_per_row = 6
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)




# --- START: 缺失的代码块 ---

# 定义 actor 的初始位姿
initial_pose = gymapi.Transform()
# 假设机械臂放在地平面上，稍微抬高一点
initial_pose.p = gymapi.Vec3(0.0, 0.0, 0.5) 
initial_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0) # 无旋转

envs = [] # 用于存储创建的环境

# 循环创建环境和 Actor
for i in range(num_envs):
    # 创建环境
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # 在环境中创建 Frana Panda 机械臂 Actor
    name = "franka_%d" % i
    # actor_id = i (或者根据需要设置)
    # segment_id = 0 (用于分割ID，通常为0)
    # filter = 0 (碰撞过滤器，通常为0)
    actor_handle = gym.create_actor(env, asset, initial_pose, name, i, 0, 0) 

    # 还可以设置 Actor 的颜色或属性
    # color = gymapi.Vec3(np.random.rand(), np.random.rand(), np.random.rand())
    # gym.set_rigid_body_color(env, actor_handle, 0, gymapi.MESH_VISUAL, color)

    # 设置 DOF 属性 (您可以将之前的 dof_props 设置代码放在这里)
    gym.set_actor_dof_properties(env, actor_handle, dof_props)

# position the camera
cam_pos = gymapi.Vec3(17.2, 2.0, 16)
cam_target = gymapi.Vec3(5, -2.5, 13)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)