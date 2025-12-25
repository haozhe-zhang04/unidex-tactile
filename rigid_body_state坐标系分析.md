# rigid_body_state 坐标系分析

## 问题
`rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)` 获得的 translation 和 rotation 是相对于什么坐标系的？

## 答案
**`rigid_body_state` 中的 translation 和 rotation 是相对于 sim 的世界坐标系的，而不是相对于每个环境的 robot base 坐标系。**

## 证据

### 1. 与 root_states 的一致性

**文件**: `legged_gym/envs/wuji/wuji_robot_pos_force.py`

```python
# 第1256行：获取actor根状态（世界坐标系）
actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)

# 第1259行：获取所有刚体状态
rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

# 第1270行：包装为tensor
self.root_states = gymtorch.wrap_tensor(actor_root_state)
self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

# 第1277-1278行：提取基座位置和姿态
self.base_quat = self.root_states[:, 3:7]  # 世界坐标系中的姿态
self.base_pos = self.root_states[:, :3]    # 世界坐标系中的位置
```

**关键观察**：
- `root_states` 是 actor 的根状态，明确在世界坐标系中
- `rigid_state` 包含所有刚体的状态，包括基座
- 基座在 `rigid_state` 中的位置应该与 `root_states` 中的位置一致

### 2. 基座位置对比

**文件**: `legged_gym/envs/wuji/wuji_robot_pos_force.py` 第650行
```python
base_pose = self.rigid_state[:, self.robot_base_idx, :3]
```

这里直接从 `rigid_state` 获取基座位置，如果它是相对于 base 坐标系的，那么这个值应该是 `[0, 0, 0]`（基座相对于自己的位置），但实际使用中它被当作世界坐标使用。

### 3. 可视化代码的修复

**文件**: `legged_gym/envs/wuji/wuji_robot_pos_force.py` 第586-588行
```python
# 将世界坐标转换为环境局部坐标（相对于env_origins）
curr_finger_tip_goal_cart_local = self.curr_finger_tip_goal_cart - self.env_origins
curr_ee_goal_cart_world_offset_local = curr_ee_goal_cart_world_offset - self.env_origins
ee_pose_local = ee_pose.squeeze(1) - self.env_origins  # ee_pose 来自 rigid_state
```

**关键点**：
- `ee_pose` 来自 `self.rigid_state[:, self.finger_tips_idx, :3]`
- 需要减去 `env_origins` 才能正确可视化
- 这证明 `rigid_state` 中的位置是世界坐标系的绝对位置

### 4. Isaac Gym 的设计原则

Isaac Gym 的设计原则是：
- **所有状态张量（state tensors）都返回世界坐标系中的值**
- 包括：
  - `actor_root_state_tensor`: 世界坐标系
  - `rigid_body_state_tensor`: 世界坐标系
  - `dof_state_tensor`: 关节角度（相对于父关节，但位置在世界坐标系）
  - `net_contact_force_tensor`: 世界坐标系

### 5. 代码验证

可以通过以下方式验证：

```python
# 验证基座位置的一致性
base_pos_from_root = self.root_states[:, :3]  # 世界坐标
base_pos_from_rigid = self.rigid_state[:, self.robot_base_idx, :3]  # 应该也是世界坐标

# 这两个值应该相等（或非常接近，因为可能有数值误差）
print(f"Base pos from root_states: {base_pos_from_root[0]}")
print(f"Base pos from rigid_state: {base_pos_from_rigid[0]}")
print(f"Difference: {base_pos_from_root[0] - base_pos_from_rigid[0]}")

# 验证需要减去 env_origins
finger_tip_world = self.rigid_state[:, self.finger_tips_idx, :3]  # 世界坐标
finger_tip_local = finger_tip_world - self.env_origins  # 相对于环境原点
print(f"Finger tip world: {finger_tip_world[0]}")
print(f"Env origin: {self.env_origins[0]}")
print(f"Finger tip local: {finger_tip_local[0]}")
```

## 总结

### rigid_body_state 的坐标系
- ✅ **Translation (位置)**: 世界坐标系中的绝对位置
- ✅ **Rotation (姿态)**: 世界坐标系中的绝对姿态（四元数）
- ✅ **所有刚体的状态都在同一个世界坐标系中表示**

### 与其他坐标系的关系

1. **世界坐标系 (World Frame)**
   - `rigid_state[:, body_idx, :3]`: 刚体在世界坐标系中的位置
   - `rigid_state[:, body_idx, 3:7]`: 刚体在世界坐标系中的姿态

2. **环境局部坐标系 (Environment Local Frame)**
   - 需要减去 `env_origins` 才能得到相对于环境原点的位置
   - `rigid_state[:, body_idx, :3] - self.env_origins`

3. **Body坐标系 (Robot Base Frame)**
   - 需要先获取基座位置和姿态，然后进行坐标转换
   - 基座位置: `rigid_state[:, robot_base_idx, :3]` 或 `base_pos`
   - 基座姿态: `rigid_state[:, robot_base_idx, 3:7]` 或 `base_quat`
   - 转换到body坐标系: `quat_rotate_inverse(base_quat, world_pos - base_pos)`

## 关于 Pinocchio 的坐标系

**文件**: `legged_gym/envs/wuji/wuji_robot_pos_force.py` 第1715-1726行

```python
def _pinocchio_forward_kinematics(self, q, env_ids=None):
    # ...
    index_tip_goal_cart = self.pinocchio_data.oMf[self.pinocchio_tips_idx[1]].translation.copy()
    self.finger_tip_goal_cart[idx] = torch.from_numpy(index_tip_goal_cart).to(self.device, dtype=torch.float32)
```

**Pinocchio 的坐标系**：
- `oMf` 表示从 frame 到 origin（基座）的变换
- `translation` 是**相对于机器人基座坐标系的位置**
- **不是世界坐标系，也不是环境坐标系**

**转换到世界坐标系**：
```python
# 如果 finger_tip_goal_cart 是相对于基座的局部坐标
finger_tip_world = self.base_pos + quat_apply(self.base_quat, self.finger_tip_goal_cart)
```

**转换到环境局部坐标系**：
```python
# 先转换到世界坐标，再减去环境原点
finger_tip_world = self.base_pos + quat_apply(self.base_quat, self.finger_tip_goal_cart)
finger_tip_env_local = finger_tip_world - self.env_origins
```

## 关键区别总结

| 数据源 | 坐标系 | 说明 |
|--------|--------|------|
| `rigid_body_state` | 世界坐标系 | 所有刚体的绝对位置和姿态 |
| `root_states` | 世界坐标系 | Actor根节点的绝对位置和姿态 |
| `pinocchio oMf.translation` | 基座坐标系 | 相对于机器人基座的局部位置 |
| `env_origins` | 世界坐标系 | 每个环境在世界坐标系中的原点位置 |







