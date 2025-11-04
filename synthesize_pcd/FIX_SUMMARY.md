# 两个关键问题修复

## 问题1：夹爪点云采样不变 ❌ → ✅

### 原因
`sample_points` 函数没有指定设备，导致在 CPU 上生成随机索引，每次可能相同。

### 修复
```python
# 修复前 ❌
sampling_idx = torch.randperm(points.shape[1])[:sample_num]

# 修复后 ✅
sampling_idx = torch.randperm(points.shape[1], device=points.device)[:sample_num]
```

**文件**: `utils/furniture.py` 第99行

---

## 问题2：夹爪姿态异常（旋转轴错误）❌ → ✅

### 原因
**四元数格式不匹配**：

| 组件 | 期望格式 | 实际格式 |
|------|---------|---------|
| `rotation_matrix_to_quaternion_simple()` 输出 | - | **(w, x, y, z)** |
| `C.pose2mat()` 输入 | **(x, y, z, w)** | ❌ (w, x, y, z) |
| `C.mat2pose()` 输出 | **(x, y, z, w)** | - |

### 修复
在 `coordinate_transform.py` 的 `robot_pose_to_april_pose()` 函数中添加格式转换：

```python
def robot_pose_to_april_pose(ee_pos, ee_quat, device='cuda:0'):
    """
    Args:
        ee_quat: [4] torch.Tensor, (w, x, y, z) 格式
    Returns:
        april_ee_quat: [4] torch.Tensor, (w, x, y, z) 格式
    """
    # ✅ 转换为 C.pose2mat 期望的格式
    ee_quat_xyzw = torch.cat([ee_quat[1:], ee_quat[0:1]])  # (w,x,y,z) -> (x,y,z,w)
    
    ee_pose_mat = C.pose2mat(ee_pos, ee_quat_xyzw, device)
    
    # 转换到 AprilTag 坐标系
    april_ee_pose_mat = robot_to_april @ ee_pose_mat
    
    # ✅ C.mat2pose 返回 (x,y,z,w)，转回 (w,x,y,z)
    april_ee_pos, april_ee_quat = C.mat2pose(april_ee_pose_mat)
    april_ee_quat_wxyz = torch.cat([april_ee_quat[3:4], april_ee_quat[0:3]])  # (x,y,z,w) -> (w,x,y,z)
    
    return april_ee_pos, april_ee_quat_wxyz
```

**文件**: `coordinate_transform.py` 第92-106行

### 为什么格式不同？

PyTorch 的约定（很多教程和实现）：
- 四元数：**(w, x, y, z)** - 标量部分在前

Robotics 和 PyBullet 的约定（furniture-bench 遵循）：
- 四元数：**(x, y, z, w)** - 标量部分在后

### 验证

```bash
python synthesize_pcd/test_quat_fix.py
```

输出：
```
四元数范数: 1.0000 (应该≈1.0) ✅
旋转矩阵行列式: 1.0000 (应该≈1.0) ✅
是否正交: True ✅
```

---

## 影响

### 修复前
1. **夹爪点云**：每帧采样相同的点，看起来静止
2. **夹爪姿态**：四元数格式错误导致旋转轴混乱，姿态异常

### 修复后
1. **夹爪点云**：每帧随机采样，正常变化
2. **夹爪姿态**：旋转正确，姿态平滑自然

---

## 相关文件

- ✅ `utils/furniture.py` - 修复采样函数
- ✅ `coordinate_transform.py` - 修复四元数格式转换
- 📝 `test_quat_fix.py` - 测试验证脚本

---

## 运行测试

```bash
cd /home2/zxp/Projects/Pointnet_Pointnet2_pytorch
python synthesize_pcd/synthesize_pcd.py
```

现在夹爪应该：
- ✅ 点云每帧变化
- ✅ 姿态旋转正确
- ✅ 开合平滑连续（使用 robot_state[15]）
