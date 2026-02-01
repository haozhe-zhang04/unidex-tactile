# Reward 设计问题分析

## 🔴 发现的严重问题

### 问题1：自适应 sigma 设计反直觉（最严重）

**位置**：`_reward_tracking_ee_orientation_6d_base` (2259-2261行) 和 `_reward_tracking_ee_force_base` (2309-2310行)

**当前实现**：
```python
sigma_scale = 0.3 + 0.7 * (diff_per_finger / (diff_per_finger + adaptive_threshold))
adaptive_sigma = torch.clamp(base_sigma * sigma_scale, min=base_sigma * 0.3, max=base_sigma * 1.0)
rew_per_finger = torch.exp(-diff_per_finger / (adaptive_sigma + 1e-6) * 2)
```

**问题分析**：
- **误差大时，sigma 变大**：这会导致奖励衰减变慢
- **反直觉**：误差大应该得到更少的奖励，但当前设计让误差大时奖励衰减更慢
- **Value function 难以学习**：reward scale 在不同状态下变化很大，导致 value function 预测困难

**数学分析**：
- 当 `diff_per_finger = 0` 时：`sigma_scale = 0.3`，`adaptive_sigma = 0.3 * base_sigma`（最小）
- 当 `diff_per_finger → ∞` 时：`sigma_scale → 1.0`，`adaptive_sigma = base_sigma`（最大）
- 这意味着：**误差越大，sigma 越大，奖励衰减越慢** ❌

**影响**：
- Value function 需要学习一个非常复杂的 reward 分布
- 不同误差状态下 reward scale 不同，导致 value loss 振荡

### 问题2：Reward 公式中的系数 2 可能导致数值不稳定

**位置**：2264行和2313行

```python
rew_per_finger = torch.exp(-diff_per_finger / (adaptive_sigma + 1e-6) * 2)
```

**问题**：
- 系数 `* 2` 会让奖励衰减更快
- 结合自适应 sigma，可能导致某些状态下 reward 接近 0，梯度消失
- 不同状态下 reward 的 scale 差异很大

### 问题3：多手指平均可能掩盖问题

**位置**：2267行和2316行

```python
rew = torch.mean(rew_per_finger, dim=1)  # shape: (num_envs,)
```

**问题**：
- 如果某些手指跟踪很好（reward ≈ 1），某些很差（reward ≈ 0），平均后可能 ≈ 0.5
- 这会让 agent 认为当前状态还可以，但实际上某些手指需要改进
- 可能导致训练不稳定

### 问题4：Reward scale 配置可能不合适

**配置**：
```python
tracking_ee_force_base = 1.0
tracking_ee_orientation_6d_base = 0.5
```

**问题**：
- 如果 reward 本身的值域是 [0, 1]，scale 1.0 和 0.5 是合理的
- 但由于自适应 sigma，实际 reward 的值域可能变化很大
- 需要检查实际 reward 的分布

## 🔧 修复建议

### 修复1：移除或反转自适应 sigma

**方案A：使用固定 sigma（推荐）**
```python
# 简单、稳定、易于调试
rew_per_finger = torch.exp(-diff_per_finger / (base_sigma + 1e-6))
```

**方案B：反转自适应 sigma（如果确实需要自适应）**
```python
# 误差大时，sigma 变小，奖励衰减更快
sigma_scale = 1.0 - 0.7 * (diff_per_finger / (diff_per_finger + adaptive_threshold))
adaptive_sigma = torch.clamp(base_sigma * sigma_scale, min=base_sigma * 0.3, max=base_sigma * 1.0)
```

### 修复2：调整 reward 公式

**移除系数 2**：
```python
# 更平滑的奖励衰减
rew_per_finger = torch.exp(-diff_per_finger / (sigma + 1e-6))
```

**或者使用平方误差**：
```python
# 更标准的 reward 形式
rew_per_finger = torch.exp(-diff_per_finger / (sigma + 1e-6))
```

### 修复3：改进多手指聚合方式

**方案A：使用最小奖励（更严格）**
```python
rew = torch.min(rew_per_finger, dim=1)[0]  # 所有手指都要好
```

**方案B：使用加权平均**
```python
# 给每个手指不同的权重
finger_weights = torch.ones(num_fingers, device=self.device) / num_fingers
rew = torch.sum(rew_per_finger * finger_weights.unsqueeze(0), dim=1)
```

**方案C：使用几何平均（更平衡）**
```python
rew = torch.prod(rew_per_finger, dim=1) ** (1.0 / num_fingers)
```

### 修复4：添加 reward 归一化

**在 compute_reward 中添加**：
```python
# 归一化 reward 到合理范围
self.rew_buf = torch.clamp(self.rew_buf, min=-10.0, max=10.0)
```

## 📊 诊断建议

### 1. 检查 reward 分布

添加代码记录 reward 的统计信息：
```python
def compute_reward(self):
    # ... existing code ...
    
    # 添加诊断信息
    if self.global_steps % 100 == 0:
        print(f"Reward stats: mean={self.rew_buf.mean():.4f}, std={self.rew_buf.std():.4f}, "
              f"min={self.rew_buf.min():.4f}, max={self.rew_buf.max():.4f}")
```

### 2. 检查 value function 预测误差

在训练日志中关注：
- Value loss 的大小和变化趋势
- Value prediction 和实际 return 的差异
- Reward 的方差

### 3. 可视化 reward 函数

绘制 reward 随误差变化的曲线，检查是否平滑

## 🎯 推荐的修复方案（优先级排序）

1. **立即修复**：移除自适应 sigma，使用固定 sigma
2. **立即修复**：移除 reward 公式中的系数 2
3. **考虑修复**：改用最小奖励而不是平均
4. **长期优化**：添加 reward 归一化和诊断工具




