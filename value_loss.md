# Value Loss 振荡问题分析与修复方案

## 🔴 发现的严重问题

### 问题1：Reward 计算错误（⚠️ 严重）

**位置**：`_reward_tracking_ee_force_base` 函数（2305行）

**原代码**：
```python
ee_pos_error_per_finger = torch.sum(finger_tip_pos_base - curr_ee_goal_cart_base_offset, dim=-1)
```

**问题**：
- 使用了 `torch.sum` 而不是 `torch.abs` 或 `torch.norm`
- 这会导致误差可能为**负值**，而且不同方向的误差会**相互抵消**
- 例如：x方向误差 +0.1m，y方向误差 -0.1m，z方向误差 0，sum = 0，但实际上误差很大！

**修复**：
```python
ee_pos_error_per_finger = torch.norm(finger_tip_pos_base - curr_ee_goal_cart_base_offset, dim=-1)
# 或者
ee_pos_error_per_finger = torch.sum(torch.abs(finger_tip_pos_base - curr_ee_goal_cart_base_offset), dim=-1)
```

### 问题2：Reward 可能不稳定

**位置**：`_reward_tracking_ee_force_base` 函数（2307-2310行）

**问题**：
- 自适应 sigma 可能导致 reward 变化很大
- 当误差大时，sigma 变大，reward 衰减慢，可能导致 reward 不稳定

**建议**：
- 使用固定的 sigma，或者更保守的自适应策略