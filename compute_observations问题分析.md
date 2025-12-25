# compute_observations 函数问题分析

## 🔴 发现的严重问题

### 问题1：除零风险 - gripper_force_kps（最严重）

**位置**：第558行
```python
curr_ee_goal_cart_world_offset = forces_offset_global / self.gripper_force_kps + ...
```

**问题**：
- `gripper_force_kps` 形状是 `(num_envs, num_fingers, 3)`
- 如果某个环境的某个手指的某个维度的 kp 接近 0，会导致除零或非常大的值
- 这会导致 observation 中出现异常大的值，value function 无法预测

**修复**：
```python
# 添加保护，避免除零
gripper_force_kps_safe = torch.clamp(self.gripper_force_kps, min=1e-3)
curr_ee_goal_cart_world_offset = forces_offset_global / gripper_force_kps_safe + ...
```

### 问题2：归一化除零风险

**位置**：`_normalize_pos` 函数（1179-1181行）
```python
normalized_pos[:,i,0:1] = 2 * (pos[:,i,0:1] - min[i]) / (max[i] - min[i]) - 1
```

**问题**：
- 如果 `max[i] - min[i] = 0`，会导致除零
- 即使不除零，如果差值很小，归一化后的值会非常大

**修复**：
```python
def _normalize_pos(self, pos):
    assert pos.shape ==(self.num_envs, len(self.finger_tips_idx), 3)
    normalized_pos = torch.zeros_like(pos)
    for i in range(len(self.finger_tips_idx)):
        x_range = self.cfg.normalization.obs_scales.finger_tip_pos_x_max[i] - self.cfg.normalization.obs_scales.finger_tip_pos_x_min[i]
        y_range = self.cfg.normalization.obs_scales.finger_tip_pos_y_max[i] - self.cfg.normalization.obs_scales.finger_tip_pos_y_min[i]
        z_range = self.cfg.normalization.obs_scales.finger_tip_pos_z_max[i] - self.cfg.normalization.obs_scales.finger_tip_pos_z_min[i]
        
        # 添加保护，避免除零
        x_range = torch.clamp(torch.tensor(x_range), min=1e-6)
        y_range = torch.clamp(torch.tensor(y_range), min=1e-6)
        z_range = torch.clamp(torch.tensor(z_range), min=1e-6)
        
        normalized_pos[:,i,0:1] = 2 * (pos[:,i,0:1] - self.cfg.normalization.obs_scales.finger_tip_pos_x_min[i]) / x_range - 1
        normalized_pos[:,i,1:2] = 2 * (pos[:,i,1:2] - self.cfg.normalization.obs_scales.finger_tip_pos_y_min[i]) / y_range - 1
        normalized_pos[:,i,2:3] = 2 * (pos[:,i,2:3] - self.cfg.normalization.obs_scales.finger_tip_pos_z_min[i]) / z_range - 1
        
        # 裁剪到合理范围
        normalized_pos[:,i,:] = torch.clamp(normalized_pos[:,i,:], min=-10.0, max=10.0)
    
    return normalized_pos
```

### 问题3：6D旋转表示可能不稳定

**位置**：第546行和564行
```python
finger_tip_orn_6d_base = quat_to_mat6d(finger_tip_orn_quat_base)
finger_tip_orn_6d_error = finger_tip_orn_6d_base - curr_finger_tip_goal_orn_6d_base
```

**问题**：
- 如果四元数没有正确归一化，`quat_to_mat6d` 可能产生异常值
- 6D 表示的范围是 [-1, 1]，但如果旋转矩阵不合法，可能超出范围

**修复**：
```python
# 确保四元数归一化
finger_tip_orn_quat_base = finger_tip_orn_quat_base / (torch.norm(finger_tip_orn_quat_base, dim=-1, keepdim=True) + 1e-8)
finger_tip_orn_6d_base = quat_to_mat6d(finger_tip_orn_quat_base)

# 裁剪6D表示到合理范围
finger_tip_orn_6d_base = torch.clamp(finger_tip_orn_6d_base, min=-2.0, max=2.0)
```

### 问题4：Observation clip 可能太严格

**位置**：第348-352行
```python
clip_obs = self.cfg.normalization.clip_observations  # 默认是 1.0
self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
```

**问题**：
- `clip_observations = 1.0` 可能太小
- 如果某些 observation 值超过 1.0，会被裁剪，导致信息丢失
- 这会让 value function 学习到不连续的函数

**建议**：
- 检查实际的 observation 范围
- 如果经常被 clip，考虑增大 `clip_observations` 或调整 normalization

### 问题5：四元数归一化可能不充分

**位置**：第545行
```python
finger_tip_orn_quat_base = quat_mul(quat_conjugate(base_quat_reshaped), finger_tip_orn_quat_world.reshape(-1,4)).reshape(self.num_envs,num_fingers,4)
```

**问题**：
- `quat_mul` 后没有显式归一化
- 虽然理论上应该保持归一化，但数值误差可能导致不归一化

**修复**：
```python
finger_tip_orn_quat_base = quat_mul(quat_conjugate(base_quat_reshaped), finger_tip_orn_quat_world.reshape(-1,4)).reshape(self.num_envs,num_fingers,4)
# 显式归一化
finger_tip_orn_quat_base = finger_tip_orn_quat_base / (torch.norm(finger_tip_orn_quat_base, dim=-1, keepdim=True) + 1e-8)
```

### 问题6：NaN/Inf 检查缺失

**问题**：
- 没有检查 observation 中是否有 NaN 或 Inf
- 这些值会导致 value function 训练崩溃

**修复**：
```python
# 在 compute_observations 最后添加检查
if torch.any(torch.isnan(self.obs_buf)) or torch.any(torch.isinf(self.obs_buf)):
    print(f"Warning: NaN or Inf in obs_buf at step {self.global_steps}")
    self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=1.0, neginf=-1.0)
```

## 🔧 推荐的修复顺序

1. **立即修复**：添加 gripper_force_kps 的除零保护
2. **立即修复**：添加归一化的除零保护
3. **立即修复**：添加 NaN/Inf 检查
4. **考虑修复**：显式归一化四元数
5. **长期优化**：检查并调整 clip_observations



