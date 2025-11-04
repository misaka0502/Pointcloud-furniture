#!/usr/bin/env python3
"""可视化夹爪，检查镜像和位置"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'furniture-bench'))
sys.path.insert(0, '/home2/zxp/Projects/Juicer_ws/juicer')

import torch
import numpy as np
import open3d as o3d
from utils.read_dataset import read_zarr
from synthesize_pcd import rotation_6d_to_matrix_simple, rotation_matrix_to_quaternion_simple, build_gripper_pcd_in_hand_frame
from coordinate_transform import robot_pose_to_april_pose
from utils.get_pcd_from_npy import get_pcd_from_offline_data
import utils.fb_control_utils as C

DATA_PATH = "/home2/zxp/Projects/Juicer_ws/juicer_dataset/processed/diffik/sim/one_leg/teleop/low/success.zarr"
ASSET_PATH = "assets/furniture_bench/mesh/square_table"

data = read_zarr(DATA_PATH)
device = 'cuda:0'

# 加载点云
gripper_pcds = get_pcd_from_offline_data(ASSET_PATH, device=device)

# 选择一帧
frame_idx = 0
action_pos = data['action/pos'][frame_idx]
robot_state = data['robot_state'][frame_idx]

ee_pos_robot = torch.tensor(action_pos[:3], device=device)
ee_rot_6d_robot = torch.tensor(action_pos[3:9], device=device)
gripper_width = robot_state[15]

# 转换
ee_rot_mat_robot = rotation_6d_to_matrix_simple(ee_rot_6d_robot)
ee_quat_robot = rotation_matrix_to_quaternion_simple(ee_rot_mat_robot)
ee_pos_april, ee_quat_april = robot_pose_to_april_pose(ee_pos_robot, ee_quat_robot, device)

print("=" * 70)
print(f"可视化夹爪 - 帧 {frame_idx}")
print("=" * 70)
print(f"EE 位置 (AprilTag): {ee_pos_april.cpu().numpy()}")
print(f"EE 四元数 (w,x,y,z): {ee_quat_april.cpu().numpy()}")
print(f"夹爪宽度: {gripper_width:.6f}")

# 1. 构建局部坐标系中的夹爪
gripper_pcd_local = build_gripper_pcd_in_hand_frame(
    gripper_pcds, gripper_width, device
)

print(f"\n局部坐标系中的夹爪:")
print(f"  点数: {gripper_pcd_local.shape[0]}")
print(f"  Y 范围: [{gripper_pcd_local[:, 1].min():.4f}, {gripper_pcd_local[:, 1].max():.4f}]")

# 2. 变换到世界坐标系
ee_pose_mat = C.batched_pose2mat(
    ee_pos_april.unsqueeze(0),
    ee_quat_april.unsqueeze(0),
    device
)

gripper_pcd_world = torch.matmul(
    gripper_pcd_local.unsqueeze(0),
    ee_pose_mat.transpose(1, 2)
).squeeze(0)[:, :3]

print(f"\n世界坐标系中的夹爪:")
print(f"  X 范围: [{gripper_pcd_world[:, 0].min():.4f}, {gripper_pcd_world[:, 0].max():.4f}]")
print(f"  Y 范围: [{gripper_pcd_world[:, 1].min():.4f}, {gripper_pcd_world[:, 1].max():.4f}]")
print(f"  Z 范围: [{gripper_pcd_world[:, 2].min():.4f}, {gripper_pcd_world[:, 2].max():.4f}]")

# 3. 创建可视化
pcd_gripper = o3d.geometry.PointCloud()
pcd_gripper.points = o3d.utility.Vector3dVector(gripper_pcd_world.cpu().numpy())
pcd_gripper.paint_uniform_color([0.0, 0.7, 0.0])  # 绿色

# 创建坐标系
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

# EE 位置标记
ee_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
ee_sphere.translate(ee_pos_april.cpu().numpy())
ee_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色

# hand 坐标系（在 EE 位置）
ee_rot_mat_april = C.quaternion_to_matrix(ee_quat_april.unsqueeze(0)).squeeze(0)
hand_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
hand_frame.rotate(ee_rot_mat_april.cpu().numpy(), center=[0, 0, 0])
hand_frame.translate(ee_pos_april.cpu().numpy())

print(f"\n可视化说明:")
print(f"  - 绿色点云: 夹爪")
print(f"  - 红色球: EE 位置")
print(f"  - RGB 坐标系 (大): 世界坐标系")
print(f"  - RGB 坐标系 (小): hand 坐标系")
print(f"  - 检查点:")
print(f"    1. 夹爪是否在 hand 坐标系的 +Z 方向？")
print(f"    2. 左右 finger 是否镜像对称（指腹对指腹）？")

o3d.visualization.draw_geometries(
    [pcd_gripper, coord_frame, ee_sphere, hand_frame],
    window_name=f"夹爪可视化 - 帧 {frame_idx}",
    width=1200,
    height=900
)
