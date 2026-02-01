
目前所用的urdf文件:
/home/hz01/haozhe_workspace/UniFP/wujihand-urdf/urdf/right.urdf

sim2real函数：
sim2real.py:/home/hz01/haozhe_workspace/UniFP/legged_gym/scripts/sim2real.py

仿真：
wuji_robot_pos_force.py：/home/hz01/haozhe_workspace/UniFP/legged_gym/envs/wuji/
可以关注的函数：
1. compute_observations()其中的obs：所有的command都是在urdf的base_link坐标系下
2. step() 包括整个roll out的过程，在322-327行self._push_finger_tip表示是否加入F_cmd和F_ext
3. update_curr_ee_goal()：对当前五根手指的x_cmd的更新。

wuji_pos_force_config.py：/home/hz01/haozhe_workspace/UniFP/legged_gym/envs/wuji/wuji_pos_force_config.py
包含所有可调的参数，WujiPosForceRoughCfg.rewards.scales是所有奖励的系数（在这个类中，把奖励的系数注释掉即可在训练时去除该奖励）


可以注意的点：
wuji关节的stiffness和damping目前是随意设置的，可在/home/hz01/haozhe_workspace/UniFP/legged_gym/envs/wuji/wuji_pos_force_config.py中的control.stiffness和control.damping中设置


代码：
在仿真中test初步版本的纯位置控制：
python legged_gym/scripts/play_wuji.py --task=wuji_pos_force --load_run=/home/hz01/haozhe_workspace/UniFP/logs/wuji_pos_force/Dec27_20-20-03_

在仿真中test初步版本的力位混合控制：
python legged_gym/scripts/play_wuji.py --task=wuji_pos_force --load_run=/home/hz01/haozhe_workspace/UniFP/logs/wuji_pos_force/Dec27_23-50-35_

在仿真中train：
python legged_gym/scripts/train_wuji.py --task=wuji_pos_force

本台机器使用梯子：
在命令行依次执行
clashtun on
clashon