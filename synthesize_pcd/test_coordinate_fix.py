#!/usr/bin/env python3
"""测试坐标转换修复"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'furniture-bench'))
sys.path.insert(0, '/home2/zxp/Projects/Juicer_ws/juicer')

import torch
from utils.read_dataset import read_zarr
from utils.get_pcd_from_npy import get_pcd_from_offline_data
from utils.furniture import Furniture
import utils.fb_control_utils as C
from synthesize_pcd import compute_gripper_poses, rotation_matrix_to_quaternion_simple, rotation_6d_to_matrix_simple
from coordinate_transform import robot_pose_to_april_pose

DATA_PATH = "/home2/zxp/Projects/Juicer_ws/juicer_dataset/processed/diffik/sim/one_leg/teleop/low/success.zarr"
ASSET_PATH = 'assets/furniture_bench/mesh/square_table'

print("=" * 70)
print("测试坐标转换修复")
print("=" * 70)

# 加载数据
data = read_zarr(DATA_PATH)
furniture = Furniture(ASSET_PATH, device='cuda:0', downsample_voxel_size=0.001)

# 测试第一帧
i = 0
print(f"\n=== 第 {i+1} 帧 ===")

# 零件位姿（AprilTag 坐标系）
parts_poses = data['parts_poses'][i]
print(f"\n零件位姿 (AprilTag 坐标系):")
print(f"  square_table_top: {parts_poses[0:3]}")
print(f"  square_table_leg1: {parts_poses[7:10]}")

# Action 中的 EE 位姿（机器人基座坐标系）
action_pos = data['action/pos'][i]
ee_pos_robot = torch.tensor(action_pos[:3], device='cuda:0', dtype=torch.float32)
ee_rot_6d_robot = torch.tensor(action_pos[3:9], device='cuda:0', dtype=torch.float32)
gripper_action = action_pos[9]

print(f"\nEE 位姿 (机器人基座坐标系):")
print(f"  位置: {ee_pos_robot}")
print(f"  6D旋转: {ee_rot_6d_robot}")
print(f"  夹爪: {gripper_action}")

# 转换到 AprilTag 坐标系
ee_rot_mat = rotation_6d_to_matrix_simple(ee_rot_6d_robot)
ee_quat_robot = rotation_matrix_to_quaternion_simple(ee_rot_mat)

ee_pos_april, ee_quat_april = robot_pose_to_april_pose(
    ee_pos_robot, ee_quat_robot, 'cuda:0'
)

print(f"\nEE 位姿 (AprilTag 坐标系):")
print(f"  位置: {ee_pos_april}")
print(f"  四元数: {ee_quat_april}")

# 计算夹爪位姿
ee_quat_april_xyzw = torch.cat([ee_quat_april[1:], ee_quat_april[0:1]])
ee_rot_mat_april = C.quaternion_to_matrix(ee_quat_april_xyzw.unsqueeze(0)).squeeze(0)
ee_rot_6d_april = torch.cat([ee_rot_mat_april[:, 0], ee_rot_mat_april[:, 1]])

left_finger_pose, right_finger_pose = compute_gripper_poses(
    ee_pos_april, ee_rot_6d_april, gripper_action, 'cuda:0'
)

print(f"\n夹爪位姿 (AprilTag 坐标系):")
print(f"  左指头: pos={left_finger_pose[:3]}, quat={left_finger_pose[3:]}")
print(f"  右指头: pos={right_finger_pose[:3]}, quat={right_finger_pose[3:]}")

# 验证：夹爪位置应该接近零件位置范围
print(f"\n✅ 验证:")
print(f"  零件 X 范围: [{parts_poses[7]:.3f}, {parts_poses[0]:.3f}]")
print(f"  夹爪左指 X: {left_finger_pose[0]:.3f}")
print(f"  夹爪右指 X: {right_finger_pose[0]:.3f}")
print(f"  → 夹爪 X 在零件范围内: {parts_poses[7] <= left_finger_pose[0] <= parts_poses[0] + 0.5}")

print(f"\n  零件 Y 范围: [{parts_poses[8]:.3f}, {parts_poses[1]:.3f}]")
print(f"  夹爪左指 Y: {left_finger_pose[1]:.3f}")
print(f"  夹爪右指 Y: {right_finger_pose[1]:.3f}")
print(f"  → 夹爪 Y 接近零件: {abs(left_finger_pose[1] - parts_poses[1]) < 0.2}")

print(f"\n  零件 Z: {parts_poses[2]:.3f}")
print(f"  夹爪左指 Z: {left_finger_pose[2]:.3f}")
print(f"  夹爪右指 Z: {right_finger_pose[2]:.3f}")
print(f"  → 夹爪在零件上方: {left_finger_pose[2] < parts_poses[2]}")  # 注意：Z 轴向下

print("\n" + "=" * 70)
print("如果以上验证都通过，说明坐标转换正确！")
print("=" * 70)
