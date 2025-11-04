# 夹爪点云可视化功能

## 概述

本次修改为 `synthesize_pcd.py` 添加了夹爪（gripper）点云的可视化功能。现在程序可以同时绘制：
- 家具部件的点云（桌面、桌腿等）
- 机器人夹爪的点云（左右两个指头，每个指头包含 finger_0 和 finger_1 两个部分）

## 实现细节

### 1. 夹爪点云数据

夹爪点云数据位于：`synthesize_pcd/assets/furniture_bench/mesh/square_table/`
- `finger_0.npy` - 185个点（夹爪主体）
- `finger_1.npy` - 133个点（夹爪辅助部分）

这两个文件是通过 `obj2npy.py` 从对应的 `.obj` 文件转换而来。

### 2. 夹爪位姿计算

夹爪位姿通过以下步骤计算：

#### a) 从 dataset 读取 action/pos
- `action/pos[i]` 是一个 10 维向量：
  - `[0:3]` - EE（End Effector）位置 (x, y, z)
  - `[3:9]` - EE 旋转（6D 表示）
  - `[9]` - 夹爪开合状态（-1: 开启, 1: 闭合）

#### b) 根据 mjx_panda.xml 计算位姿关系
根据 `synthesize_pcd/assets/franka_emika_panda/mjx_panda.xml` 中的定义：
- EE 到 hand 的偏移：(0, 0, -0.107m)，旋转偏移 quat="0.9238795 0 0 -0.3826834"
- hand 到 finger base 的偏移：(0, 0, 0.0584m)
- 夹爪最大宽度：0.065m（根据 furniture_rl_sim_env.py）

#### c) 计算左右指头位置
- **左指头**：沿 hand 坐标系的 +Y 方向偏移 `gripper_width / 2`
- **右指头**：沿 hand 坐标系的 -Y 方向偏移 `gripper_width / 2`，同时旋转 180°

### 3. 代码结构

#### 新增函数

```python
def rotation_matrix_to_quaternion_simple(rot_mat)
```
将旋转矩阵转换为四元数

```python
def quat_apply_simple(quat, vec)
```
使用四元数旋转向量

```python
def quat_mul_simple(q1, q2)
```
四元数乘法

```python
def compute_gripper_poses(ee_pos, ee_rot_6d, gripper_action, device)
```
根据 EE 位姿和夹爪状态计算左右指头的位姿

#### 主程序修改

在 `main()` 函数中：
1. 加载夹爪点云数据
2. 对每一帧：
   - 从 `action/pos` 提取 EE 信息和夹爪状态
   - 调用 `compute_gripper_poses()` 计算左右指头位姿
   - 使用位姿矩阵变换夹爪点云到世界坐标系
   - 合并家具和夹爪点云
   - 采样并可视化

### 4. 点云变换

夹爪点云变换使用齐次坐标和批量矩阵乘法：

```python
# 扩展点云以匹配环境数量
finger_0_expanded = gripper_pcds['finger_0'].unsqueeze(0).expand(n_envs, -1, -1)  # [N_env, N_points, 4]

# 变换点云
left_pose_mat = C.batched_pose2mat(left_pose[:, :3], left_pose[:, 3:7], device)  # [N_env, 4, 4]
gripper_pcds_world['finger_0_left'] = torch.matmul(
    finger_0_expanded, 
    left_pose_mat.transpose(1, 2)
)[:, :, :3]  # [N_env, N_points, 3]
```

## 使用方法

### 运行主程序

```bash
cd /home2/zxp/Projects/Pointnet_Pointnet2_pytorch/synthesize_pcd
/home2/zxp/miniconda3/envs/pointnet2/bin/python3 synthesize_pcd.py
```

### 运行测试

```bash
# 测试夹爪点云生成
/home2/zxp/miniconda3/envs/pointnet2/bin/python3 test_gripper_full.py
```

## 测试结果

测试显示每一帧包含：
- **家具点云**：约 214,326 个点（5个部件：桌面+4个桌腿）
- **夹爪点云**：636 个点（4个部分：左指头finger_0+finger_1，右指头finger_0+finger_1）
- **总点云**：214,962 个点
- **采样后**：4,096 个点

夹爪位置会根据：
1. EE 位置随时间变化
2. 夹爪开合状态（开启时两指头分开，闭合时靠拢）

## 文件清单

### 修改的文件
- `synthesize_pcd/synthesize_pcd.py` - 添加夹爪点云处理逻辑
- `synthesize_pcd/utils/furniture.py` - 移除对 furniture_bench 的依赖，跳过加载 finger 部件
- `synthesize_pcd/utils/obj2npy.py` - 新创建，用于转换 OBJ 到 NPY

### 新增的文件
- `synthesize_pcd/assets/furniture_bench/mesh/square_table/finger_0.npy` - 夹爪点云数据
- `synthesize_pcd/assets/furniture_bench/mesh/square_table/finger_1.npy` - 夹爪点云数据
- `synthesize_pcd/test_gripper_full.py` - 完整测试脚本

## 依赖说明

程序需要以下 Python 环境：
- Python 3.8+
- PyTorch
- NumPy
- Open3D
- zarr（用于读取数据集）

推荐使用：`/home2/zxp/miniconda3/envs/pointnet2/bin/python3`

## 注意事项

1. **坐标系统**：所有点云都在世界坐标系下，单位为米（m）
2. **夹爪方向**：右指头相对左指头旋转了180°（绕Z轴）
3. **性能**：夹爪点云较小（636点），对性能影响很小
4. **可视化**：如果在无头模式下运行，需要设置 `RENDER=False`

## 未来改进

可能的改进方向：
1. 支持更多夹爪类型
2. 添加夹爪关节动画
3. 优化点云密度和采样策略
4. 添加碰撞检测可视化
