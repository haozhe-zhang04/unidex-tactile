# Value Function Loss 后期振荡原因分析

## 🔍 可能的根本原因

### 1. **Domain Randomization 持续变化**（最可能）

**配置发现**：
```python
randomize_gripper_force_gains = True
gripper_force_kp_range = [100., 200.]
```

**问题**：
- 每个 episode 都会随机化 `gripper_force_kps` (100-200 的范围)
- 这导致目标位置计算 `curr_ee_goal_cart_base_offset = forces_offset_global / gripper_force_kps` 具有随机性
- **后期影响**：训练初期 kp 随机性对学习影响不大，后期 policy 已经学到一些模式，但 kp 变化破坏了这些模式，导致 value function 预测不准确

**证据**：
- Value loss 在后期振荡说明前期学习稳定，但后期环境变化破坏了学习到的模式
- Force 相关 reward 可能在不同 kp 下有不同表现

### 2. **Observation 维度过高**（可能）

**配置发现**：
```python
frame_stack = 32
num_single_obs = 225
num_observations = int(frame_stack * num_single_obs)  # 32 * 225 = 7200
```

**问题**：
- Observation 维度极高（7200），value network 难以有效学习
- 后期训练时，temporal dependency 更复杂，value function 难以拟合

### 3. **Value Loss Coefficient 过高**（可能）

**配置发现**：
```python
value_loss_coef = 2.0  # 比标准值 1.0 高
```

**问题**：
- Value loss 权重过高，可能导致 policy 梯度被 value loss 影响
- 后期训练时，value prediction error 积累，可能导致振荡

### 4. **Reward 分布变化**（可能）

**配置发现**：
- 多个 reward 组件都有非零 scale
- 自适应 sigma 可能在后期导致 reward 分布变化

**问题**：
- 后期 policy 更复杂，reward 分布可能发生变化
- Value function 需要适应新的 reward 分布

## 🔧 建议的修复方案

### 方案1：控制 Domain Randomization（推荐优先）

**修改1：减少 kp 随机化范围**
```python
# wuji_pos_force_config.py
gripper_force_kp_range = [150., 170.]  # 原来 [100., 200.]，现在范围缩小
```

**修改2：添加 curriculum 到 kp**
```python
# 训练初期使用固定 kp，后期才随机化
if self.global_steps < 1000 * 24:  # 前 1000 episodes 使用固定值
    self.gripper_force_kps.fill_(160.0)  # 固定值
else:
    # 正常随机化
    min_kp, max_kp = self.cfg.commands.gripper_force_kp_range
    self.gripper_force_kps = torch.rand((self.num_envs, len(self.finger_tips_idx), 3), device=self.device) * (max_kp - min_kp) + min_kp
```

### 方案2：减少 Observation 维度

**修改 frame_stack**：
```python
# wuji_pos_force_config.py
frame_stack = 8  # 从 32 降低到 8
c_frame_stack = 1  # 从 3 降低到 1
```

### 方案3：调整 Value Loss 权重

**修改 value_loss_coef**：
```python
# wuji.py
value_loss_coef = 1.0  # 从 2.0 降低到 1.0
```

### 方案4：检查 Reward 稳定性

添加 reward 分布监控：
```python
# 在 compute_reward 中添加
if self.global_steps % 1000 == 0:  # 更频繁监控
    for name, rew in self.reward_logs.items():
        rew_mean = rew.mean().item()
        rew_std = rew.std().item()
        print(f"Reward {name}: mean={rew_mean:.6f}, std={rew_std:.6f}")
```

## 📊 诊断步骤

### 1. 立即检查
```bash
# 运行训练，观察 reward_logs 输出
# 检查是否在特定 step 后 reward 分布发生变化
```

### 2. 临时禁用 Domain Randomization
```python
# 临时修改，验证是否是 kp 随机化导致的问题
randomize_gripper_force_gains = False
self.gripper_force_kps.fill_(160.0)  # 固定值
```

### 3. 监控 Value Prediction Error
```python
# 在 PPO update 中添加
if it % 100 == 0:
    value_pred_error = (returns_batch - value_batch).abs().mean().item()
    print(f"Value prediction error: {value_pred_error:.4f}")
```

## 🎯 最可能的解决方案

**优先尝试**：
1. 控制 `gripper_force_kp_range` 范围（从 [100,200] 改为 [150,170]）
2. 或者完全禁用 kp 随机化验证效果
3. 如果问题解决，再考虑其他优化

**如果上述无效**：
1. 降低 `frame_stack` 从 32 到 8
2. 降低 `value_loss_coef` 从 2.0 到 1.0

## 💡 根本原因猜测

**最可能**：Domain randomization 在后期破坏了已经学到的 value function 模式。

**为什么后期才振荡**：前期 policy 简单，kp 变化影响小；后期 policy 复杂，kp 变化导致 value prediction error 积累。


