from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin
import numpy as np
import os

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
    print(name, pos)
