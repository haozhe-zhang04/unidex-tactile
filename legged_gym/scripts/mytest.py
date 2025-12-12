from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin
import numpy as np
import os
import torch
from pytorch3d.transforms import quaternion_slerp
# 旋转矩阵转换为四元数
def mat3x3_to_quat(R):
    """
    R: (..., 3, 3)
    return: (..., 4) quaternion (w, x, y, z)
    """
    m00 = R[..., 0, 0]
    m11 = R[..., 1, 1]
    m22 = R[..., 2, 2]

    t = m00 + m11 + m22

    # quaternion allocation
    qw = torch.empty_like(t)
    qx = torch.empty_like(t)
    qy = torch.empty_like(t)
    qz = torch.empty_like(t)

    cond1 = t > 0
    cond2 = (m00 >= m11) & (m00 >= m22)
    cond3 = m11 >= m22

    # Case 1: trace > 0
    t1 = t[cond1]
    s = 0.5 / torch.sqrt(t1 + 1.0)
    qw[cond1] = 0.25 / s
    qx[cond1] = (R[..., 2, 1][cond1] - R[..., 1, 2][cond1]) * s
    qy[cond1] = (R[..., 0, 2][cond1] - R[..., 2, 0][cond1]) * s
    qz[cond1] = (R[..., 1, 0][cond1] - R[..., 0, 1][cond1]) * s

    # Case 2: R00 is largest diagonal
    c2 = ~cond1 & cond2
    t2 = torch.sqrt(1.0 + m00[c2] - m11[c2] - m22[c2]) * 2
    qx[c2] = 0.25 * t2
    qw[c2] = (R[..., 2, 1][c2] - R[..., 1, 2][c2]) / t2
    qy[c2] = (R[..., 1, 0][c2] + R[..., 0, 1][c2]) / t2
    qz[c2] = (R[..., 2, 0][c2] + R[..., 0, 2][c2]) / t2

    # Case 3: R11 is largest diagonal
    c3 = ~cond1 & ~cond2 & cond3
    t3 = torch.sqrt(1.0 + m11[c3] - m00[c3] - m22[c3]) * 2
    qy[c3] = 0.25 * t3
    qw[c3] = (R[..., 0, 2][c3] - R[..., 2, 0][c3]) / t3
    qx[c3] = (R[..., 1, 0][c3] + R[..., 0, 1][c3]) / t3
    qz[c3] = (R[..., 2, 1][c3] + R[..., 1, 2][c3]) / t3

    # Case 4: R22 is largest diagonal
    c4 = ~cond1 & ~cond2 & ~cond3
    t4 = torch.sqrt(1.0 + m22[c4] - m00[c4] - m11[c4]) * 2
    qz[c4] = 0.25 * t4
    qw[c4] = (R[..., 1, 0][c4] - R[..., 0, 1][c4]) / t4
    qx[c4] = (R[..., 2, 0][c4] + R[..., 0, 2][c4]) / t4
    qy[c4] = (R[..., 2, 1][c4] + R[..., 1, 2][c4]) / t4

    quat = torch.stack([qx, qy, qz, qw], dim=-1)
    quat = quat / quat.norm(dim=-1, keepdim=True)  # normalize
    return quat

urdf_path = "wujihand-urdf/urdf/right.urdf"
mesh_dir  = os.path.dirname(urdf_path)

# 自动根据 URDF 判断是否是 fixed base 或 free-flyer
robot = RobotWrapper.BuildFromURDF(urdf_path, mesh_dir)

model = robot.model
data = model.createData()

# Show DOFs
print("nq:", model.nq)
print("nv:", model.nv)

# create q
q = pin.neutral(model)
q = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
print(q)
# run fk
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

# fingers
tip_names = [
    "finger1_tip_link",
    "finger2_tip_link",
    "finger3_tip_link",
    "finger4_tip_link",
    "finger5_tip_link",
]

for name in tip_names:
    fid = model.getFrameId(name)
    pos = data.oMf[fid].translation
    orn = data.oMf[fid].rotation
    quat = mat3x3_to_quat(torch.tensor(orn))
    quat = quaternion_slerp(quat, quat, 0.5)
    print(name, pos, quat.numpy())
