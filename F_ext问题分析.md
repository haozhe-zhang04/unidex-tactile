# F_ext 施加问题分析

## 🔍 发现的问题

### 问题1：形状不匹配（最严重）

**代码**：
```python
self.force_target_finger_tips_ext[new_selected_env_ids_ext,:,0:3] = torch.ones(3)
```

**问题**：
- `force_target_finger_tips_ext` 的形状是 `(num_envs, num_fingers, 3)`
- `torch.ones(3)` 的形状是 `(3,)`
- 这会导致广播，但可能不是你想要的效果

**应该改为**：
```python
# 选项1：所有手指都设置为 [1, 1, 1]
self.force_target_finger_tips_ext[new_selected_env_ids_ext, :, 0:3] = torch.ones(1, 1, 3, device=self.device)

# 选项2：每个手指分别设置（如果需要不同值）
self.force_target_finger_tips_ext[new_selected_env_ids_ext, :, 0] = 1.0
self.force_target_finger_tips_ext[new_selected_env_ids_ext, :, 1] = 1.0
self.force_target_finger_tips_ext[new_selected_env_ids_ext, :, 2] = 1.0
```

### 问题2：力的渐变过程

**代码位置**：第1573行
```python
self.forces[env_ids_apply_push_step1[:, None], self.finger_tips_idx, :3] = \
    (self.force_target_finger_tips_ext[env_ids_apply_push_step1,:, :3]/push_duration_reshaped) * \
    (torch.clamp(self.episode_length_buf[...] - ..., torch.zeros_like(...), push_duration_reshaped))
```

**问题**：
- 力不是立即达到目标值，而是从0逐渐增加到目标值
- 在 ramp up 阶段，实际施加的力 = `target_force * (current_step / push_duration)`
- 所以即使设置了 `torch.ones(3)`，在 ramp up 阶段实际力会小于 `[1, 1, 1]`

**解决方案**：
- 如果想立即达到目标值，需要修改渐变逻辑
- 或者等待 ramp up 完成后再检查 sensor 值

### 问题3：坐标系

**力的施加**（第320行）：
```python
gymapi.LOCAL_SPACE  # 局部坐标系
```

**Sensor读取**（第1989行）：
```python
sensor_props.use_world_frame = False  # 局部坐标系
```

**结论**：坐标系是一致的（都是局部坐标系），所以这不是问题。

## 🔧 修复建议

### 修复1：正确的形状设置

```python
# 在 _push_finger_tip 函数中
self.force_target_finger_tips_ext[new_selected_env_ids_ext, :, 0:3] = torch.ones(
    len(new_selected_env_ids_ext), len(self.finger_tips_idx), 3, 
    device=self.device
)
```

### 修复2：添加调试代码

在 `_push_finger_tip` 和 `compute_observations` 中添加：

```python
# 在 _push_finger_tip 中，设置 force_target 后
if len(new_selected_env_ids_ext) > 0:
    print(f"force_target_finger_tips_ext shape: {self.force_target_finger_tips_ext[new_selected_env_ids_ext, :, 0:3].shape}")
    print(f"force_target_finger_tips_ext values: {self.force_target_finger_tips_ext[new_selected_env_ids_ext, :, 0:3]}")

# 在 step 中，施加力后
print(f"forces applied shape: {self.forces[env_ids_apply_push_step1[:, None], self.finger_tips_idx, :3].shape}")
print(f"forces applied values: {self.forces[env_ids_apply_push_step1[:, None], self.finger_tips_idx, :3]}")

# 在 compute_observations 中
print(f"sensor_forces_local shape: {self.sensors_forces[:,:,:3].shape}")
print(f"sensor_forces_local values: {self.sensors_forces[:,:,:3]}")
```

### 修复3：检查 ramp up 阶段

如果想立即看到效果，可以临时禁用渐变：

```python
# 临时修改：立即达到目标值（跳过渐变）
self.forces[env_ids_apply_push_step1[:, None], self.finger_tips_idx, :3] = \
    self.force_target_finger_tips_ext[env_ids_apply_push_step1,:, :3]
```

## 📊 调试步骤

1. **检查形状**：打印 `force_target_finger_tips_ext` 的形状和值
2. **检查施加的力**：打印 `self.forces` 的值（在施加后）
3. **检查 ramp up**：确认是否在 ramp up 阶段
4. **检查 sensor**：确认 sensor 读取的时机（应该在 physics step 之后）




