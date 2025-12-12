# force_sensor_tensor 坐标系分析

## 问题
`force_tensor = self.gym.acquire_force_sensor_tensor(self.sim)` 获得的力是相对于什么坐标系的？

## 答案
**`force_tensor` 中的力是相对于传感器所在刚体的局部坐标系（传感器坐标系），而不是世界坐标系。**

## 关键证据

### 1. 传感器创建时的配置

**文件**: `legged_gym/envs/wuji/wuji_robot_pos_force.py` 第1588-1590行

```python
sensor_props = gymapi.ForceSensorProperties()
sensor_props.use_world_frame = False  # ← 关键：不使用世界坐标系
self.gym.create_asset_force_sensor(robot_asset, tips_idx, sensor_pose, sensor_props)
```

**关键点**：
- `use_world_frame = False` 明确指定传感器数据在**局部坐标系**中返回
- 如果设置为 `True`，则会在世界坐标系中返回

### 2. 代码中的使用方式

**文件**: `legged_gym/envs/wuji/wuji_robot_pos_force.py` 第355-359行

```python
forces_local = self.sensors_forces[:, 0, :3]  # ← 变量名明确标注为 local
forces_cmd_local = self.current_Fxyz_finger_tips_cmd
forces_offset_local = (forces_local + forces_cmd_local)

forces_offset_global = quat_apply(self.base_quat, forces_offset_local)  # ← 需要转换到世界坐标
```

**关键观察**：
- 变量命名为 `forces_local`，说明是局部坐标
- 需要转换到世界坐标时使用 `quat_apply(self.base_quat, ...)`
- 这证明原始数据是局部坐标系的

### 3. 传感器安装位置

**文件**: `legged_gym/envs/wuji/wuji_robot_pos_force.py` 第1573行

```python
self.sensors_pos_link[0] = torch.tensor([0.,0.,0.,0,0,0,1], ...)  # 位置和姿态都是单位
```

传感器安装在 finger_tip 上，位置偏移为 `[0, 0, 0]`，姿态为单位四元数，说明：
- 传感器坐标系与 finger_tip link 的坐标系对齐
- 力的方向是相对于 finger_tip link 的局部坐标系

### 4. 坐标系转换的使用

**文件**: `legged_gym/envs/wuji/wuji_robot_pos_force.py` 第580行

```python
forces_offset_global = quat_apply(self.base_quat, forces_offset_local)
```

**注意**：这里使用 `base_quat` 转换，但传感器实际上在 finger_tip 上。这可能意味着：
- 代码假设 finger_tip 的朝向与 base 相同（对于某些机器人可能成立）
- 或者代码需要修正，应该使用 finger_tip 的姿态进行转换

**正确的转换方式应该是**：
```python
# 获取 finger_tip 的姿态（从 rigid_state）
finger_tip_quat = self.rigid_state[:, self.finger_tips_idx, 3:7]
# 转换到世界坐标系
forces_offset_global = quat_apply(finger_tip_quat, forces_offset_local)
```

## 详细说明

### force_tensor 的数据格式

**文件**: `legged_gym/envs/wuji/wuji_robot_pos_force.py` 第1288行

```python
self.sensors_forces = gymtorch.wrap_tensor(force_tensor).view(self.num_envs, -1, 6)
```

**数据格式**：
- 形状: `(num_envs, num_sensors, 6)`
- 每个传感器返回 6 个值：
  - `[0:3]`: 力 (force) - 在传感器局部坐标系中
  - `[3:6]`: 力矩 (torque) - 在传感器局部坐标系中

### 坐标系选择的影响

| `use_world_frame` | 坐标系 | 说明 |
|-------------------|--------|------|
| `False` (当前设置) | **传感器局部坐标系** | 相对于传感器所在刚体的坐标系 |
| `True` | **世界坐标系** | 相对于 sim 的世界坐标系 |

### 传感器局部坐标系的特点

1. **原点**：传感器安装位置（在 finger_tip link 上）
2. **方向**：与传感器所在刚体的坐标系对齐
3. **随刚体运动**：当 finger_tip 旋转时，坐标系也随之旋转
4. **独立性**：每个环境中的每个传感器都有自己的局部坐标系

## 坐标系转换示例

### 从传感器局部坐标到世界坐标

```python
# 1. 获取传感器力（局部坐标）
forces_local = self.sensors_forces[:, 0, :3]  # (num_envs, 3)

# 2. 获取传感器所在刚体的姿态（世界坐标）
finger_tip_quat = self.rigid_state[:, self.finger_tips_idx, 3:7]  # (num_envs, 4)

# 3. 转换到世界坐标系
forces_world = quat_apply(finger_tip_quat, forces_local)  # (num_envs, 3)
```

### 从传感器局部坐标到基座坐标系

```python
# 1. 获取传感器力（局部坐标）
forces_local = self.sensors_forces[:, 0, :3]

# 2. 获取传感器所在刚体的姿态（世界坐标）
finger_tip_quat = self.rigid_state[:, self.finger_tips_idx, 3:7]

# 3. 获取基座姿态（世界坐标）
base_quat = self.base_quat

# 4. 先转换到世界坐标，再转换到基座坐标
forces_world = quat_apply(finger_tip_quat, forces_local)
forces_base = quat_rotate_inverse(base_quat, forces_world)
```

### 从传感器局部坐标到环境局部坐标

```python
# 1. 获取传感器力（局部坐标）
forces_local = self.sensors_forces[:, 0, :3]

# 2. 转换到世界坐标
finger_tip_quat = self.rigid_state[:, self.finger_tips_idx, 3:7]
forces_world = quat_apply(finger_tip_quat, forces_local)

# 3. 环境局部坐标（相对于环境原点）
# 注意：力是向量，不受位置偏移影响，所以不需要减去 env_origins
# 但如果需要表示力的作用点，则需要考虑位置
```

## 代码中的潜在问题

**文件**: `legged_gym/envs/wuji/wuji_robot_pos_force.py` 第359、580行

```python
forces_offset_global = quat_apply(self.base_quat, forces_offset_local)
```

**问题**：
- 使用 `base_quat` 转换，但传感器在 `finger_tip` 上
- 如果 finger_tip 的朝向与 base 不同，转换结果会不正确

**建议修正**：
```python
# 获取 finger_tip 的姿态
finger_tip_quat = self.rigid_state[:, self.finger_tips_idx, 3:7].squeeze(1)
# 使用 finger_tip 的姿态转换
forces_offset_global = quat_apply(finger_tip_quat, forces_offset_local)
```

## 总结

### force_tensor 的坐标系
- ✅ **力 (force)**: 相对于传感器所在刚体的局部坐标系
- ✅ **力矩 (torque)**: 相对于传感器所在刚体的局部坐标系
- ✅ **当前设置**: `use_world_frame = False`，所以是局部坐标
- ⚠️ **代码问题**: 使用 `base_quat` 转换可能不正确，应该使用传感器所在刚体的姿态

### 与其他坐标系的关系

| 数据源 | 坐标系 | 说明 |
|--------|--------|------|
| `force_tensor` (当前设置) | **传感器局部坐标系** | 相对于传感器所在刚体 |
| `force_tensor` (如果 `use_world_frame=True`) | **世界坐标系** | 相对于 sim 的世界坐标系 |
| `rigid_body_state` | **世界坐标系** | 所有刚体的绝对位置和姿态 |
| `root_states` | **世界坐标系** | Actor根节点的绝对位置和姿态 |



