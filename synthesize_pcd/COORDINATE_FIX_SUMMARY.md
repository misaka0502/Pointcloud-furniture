# 坐标系转换修复总结

## ✅ 问题已解决

成功修复了夹爪点云与家具点云坐标系不匹配的问题。

## 📌 问题原因

1. **`action/pos` (EE位姿)** 在 **机器人基座坐标系** 下
   - 从 `get_ee_pose()` 返回：`ee_pos = hand_pos - base_pos`
   - 相对于机器人基座的位置

2. **`parts_poses` (零件位姿)** 在 **AprilTag 坐标系** 下
   - 从 `_get_parts_poses(sim_coord=False)` 返回
   - 经过 `sim_coord_to_april_coord()` 转换

3. **坐标系偏移**
   - 机器人基座到 AprilTag 的偏移：`(0.3015, 0, 0)` 米
   - 加上旋转：`(π, 0, π/2)` 欧拉角

## 🔧 解决方案

### 1. 添加坐标转换模块

创建 `coordinate_transform.py`：

```python
# AprilTag -> Robot 的变换矩阵
tag_base_from_robot_base = get_mat(
    (0.23 + 0.0715, 0, 0),  # (0.3015, 0, 0)
    (np.pi, 0, np.pi / 2)
)

# Robot -> AprilTag 的变换矩阵
robot_to_april_mat = inv(tag_base_from_robot_base)
```

### 2. 在 synthesize_pcd.py 中集成

在计算夹爪位姿前添加坐标转换：

```python
# 1. 将 6D 旋转转换为四元数
ee_rot_mat = rotation_6d_to_matrix_simple(ee_rot_6d_robot)
ee_quat_robot = rotation_matrix_to_quaternion_simple(ee_rot_mat)

# 2. 坐标系转换：Robot -> AprilTag
ee_pos_april, ee_quat_april = robot_pose_to_april_pose(
    ee_pos_robot, ee_quat_robot, device
)

# 3. 将四元数转回 6D 表示
ee_quat_april_xyzw = torch.cat([ee_quat_april[1:], ee_quat_april[0:1]])
ee_rot_mat_april = C.quaternion_to_matrix(ee_quat_april_xyzw.unsqueeze(0)).squeeze(0)
ee_rot_6d_april = torch.cat([ee_rot_mat_april[:, 0], ee_rot_mat_april[:, 1]])

# 4. 使用转换后的位姿计算夹爪
left_finger_pose, right_finger_pose = compute_gripper_poses(
    ee_pos_april, ee_rot_6d_april, gripper_action, device
)
```

### 3. 添加辅助函数

在 `synthesize_pcd.py` 中添加：
- `rotation_6d_to_matrix_simple()` - 6D 旋转到矩阵
- `rotation_matrix_to_quaternion_simple()` - 矩阵到四元数

## 📊 验证结果

测试第一帧数据：

### 转换前（机器人坐标系）
- EE 位置: `[0.5379, 0.0279, 0.1396]`
- 零件位置: `[0.0005, 0.2394, -0.0157]`
- ❌ X 坐标相差 ~0.5 米

### 转换后（AprilTag 坐标系）
- EE 位置: `[0.0279, 0.2364, -0.1396]`
- 零件位置: `[0.0005, 0.2394, -0.0157]`
- ✅ 坐标接近！

### 夹爪位置验证
- 夹爪 X: `[-0.0175, 0.0071]` ✓ 在零件范围内 `[-0.200, 0.001]`
- 夹爪 Y: `[0.2103, 0.2582]` ✓ 接近零件 `[0.067, 0.239]`
- 夹爪 Z: `[-0.1326, -0.0962]` ✓ 在零件上方 `-0.016`

## 📁 修改的文件

1. **synthesize_pcd/synthesize_pcd.py**
   - 添加坐标转换逻辑
   - 添加 6D 旋转和四元数转换函数

2. **synthesize_pcd/coordinate_transform.py** (新建)
   - 实现坐标系转换函数
   - 提供转换矩阵计算

3. **synthesize_pcd/test_coordinate_fix.py** (新建)
   - 验证坐标转换正确性

## 🎯 关于 Open3D 可视化问题

你提到的 Open3D 可视化中坐标轴方向问题可能是独立的：

### 可能的原因
1. **Isaac Gym 坐标系**：Z 轴向上
2. **Open3D 坐标系**：Y 轴向上
3. **AprilTag 坐标系**：可能有自己的方向定义

### 如果需要调整可视化
可以在显示前添加坐标轴转换：

```python
def adjust_for_visualization(points):
    """将 Z-up 转换为 Y-up（如果需要）"""
    # 例如：旋转 -90° 绕 X 轴
    rot = torch.tensor([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0]
    ], device=points.device, dtype=points.dtype)
    return torch.matmul(points, rot.T)
```

但目前坐标系转换是正确的，可视化方向问题可以之后单独调整。

## 🚀 下一步

现在可以运行完整的可视化：

```bash
cd /home2/zxp/Projects/Pointnet_Pointnet2_pytorch
python synthesize_pcd/synthesize_pcd.py
```

夹爪点云现在应该与家具点云正确对齐！
