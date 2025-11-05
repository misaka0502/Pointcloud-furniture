# 夹爪点云处理 - 快速参考

## 🎯 核心思路

从 **EE 位姿 + gripper_width** → 重建完整夹爪点云

## 📊 关键参数

```python
# 输入
ee_pos: [3]           # EE 位置
ee_rot_6d: [6]        # EE 姿态 (6D rotation)
gripper_width: float  # 夹爪宽度

# 结构参数
finger_base_z = -0.0416  # finger 相对 EE (gripper site) 的 Z 偏移
finger_1_inner_y = -0.000084  # 原始 finger 内侧位置
```

## 🔄 变换流程

### 1. 右 Finger（Y < 0）

```python
# 旋转 180° (让内侧朝向 +Y)
right[:, 0] = -original[:, 0]  # X 取反
right[:, 1] = -original[:, 1]  # Y 取反

# 平移到目标位置
target_y = -gripper_width / 2
right[:, 1] += (target_y - (-finger_1_inner_y))
right[:, 2] += finger_base_z
```

### 2. 左 Finger（Y > 0）

```python
# X 镜像 (保持内侧朝向 -Y)
left[:, 0] = -original[:, 0]  # X 取反

# 平移到目标位置
target_y = +gripper_width / 2
left[:, 1] += (target_y - finger_1_inner_y)
left[:, 2] += finger_base_z
```

## ✅ 验证检查点

```python
# 1. 夹爪宽度
actual_width = left_finger_1.Y_min - right_finger_1.Y_max
assert abs(actual_width - gripper_width) < 0.001

# 2. Finger 朝向
assert left_finger_1.Y_min < left_finger_1.Y_mean  # 朝内 (-Y)
assert right_finger_1.Y_max > right_finger_1.Y_mean  # 朝内 (+Y)

# 3. Finger 位置
assert gripper_pcd_local[:, 2].min() ≈ -0.0415  # 底端
assert gripper_pcd_local[:, 2].max() ≈ 0.0123   # 顶端
```

## 🔍 常见问题

### Q1: 为什么 finger_base_z = -0.0416 而不是 0.0584?

**A**: EE 是 gripper site (hand 上方 0.1m)，不是 hand frame
```
finger_base 在 hand 上方: 0.0584m
gripper site 在 hand 上方: 0.1m
相对位置: 0.0584 - 0.1 = -0.0416m
```

### Q2: 为什么右 finger 要旋转 180°?

**A**: 原始 finger 内侧朝向 -Y，右 finger 需要朝向 +Y（朝内）
```
旋转 180°: (x, y) → (-x, -y)
内侧从 -Y 翻转到 +Y ✅
```

### Q3: gripper_width 是什么？

**A**: **finger_1 内侧**之间的距离（不是 finger 中心）
```
gripper_width = left_finger_1_inner - right_finger_1_inner
              = Y_at(+gripper_width/2) - Y_at(-gripper_width/2)
```

## 📈 性能

- **速度**: 316 FPS (3.16 ms/帧)
- **点云**: 636 点/夹爪 (finger_0: 185, finger_1: 133 × 2)
- **精度**: 夹爪宽度误差 < 0.001m

## 🆚 与 TRANSIC-envs 对比

| 特性 | TRANSIC-envs | 我们 |
|-----|-------------|------|
| 输入 | finger 位姿 | EE 位姿 + width |
| 旋转 | quat 变换 | 坐标取反 |
| 点云 | finray_finger | finger_0 + finger_1 |
| 优势 | 简单准确 | 适用离线数据 |

## 🛠️ 使用示例

```python
# 1. 加载点云
gripper_pcds = get_pcd_from_offline_data(asset_path)

# 2. 构建夹爪
gripper_pcd_local = build_gripper_pcd_in_hand_frame(
    gripper_pcds, 
    gripper_width=0.065,
    device='cuda'
)

# 3. 变换到世界坐标系
ee_pose_mat = batched_pose2mat(ee_pos, ee_quat)
gripper_pcd_world = (gripper_pcd_local @ ee_pose_mat.T)[:, :3]
```

## 📚 相关文件

- **详细文档**: `gripper_pointcloud_implementation.md`
- **实现代码**: `synthesize_pcd.py::build_gripper_pcd_in_hand_frame()`
- **测试脚本**: `test_gripper_site_offset.py`
- **可视化**: `visualize_gripper_debug.py`
