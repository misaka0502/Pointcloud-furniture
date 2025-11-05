#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐标系转换工具

根据 furniture-bench/furniture_bench/config.py:
tag_base_from_robot_base = get_mat(
    (0.23 + 0.0715, 0, -ROBOT_HEIGHT),  # = (0.3015, 0, 0)
    (np.pi, 0, np.pi / 2)  # 欧拉角 (roll, pitch, yaw)
)

其中 ROBOT_HEIGHT = 0.0
"""

import torch
import numpy as np


def rot_mat(angles):
    """
    根据欧拉角(x, y, z)构建旋转矩阵
    旋转顺序：R = Rz @ Ry @ Rx
    """
    x, y, z = angles
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def get_mat(pos, angles):
    """
    构建齐次变换矩阵
    Args:
        pos: [x, y, z] 平移
        angles: [roll, pitch, yaw] 欧拉角
    Returns:
        4x4 齐次变换矩阵
    """
    transform = np.zeros((4, 4), dtype=np.float32)
    transform[:3, :3] = rot_mat(angles)
    transform[:3, 3] = pos
    transform[3, 3] = 1.0
    return transform


def get_april_to_robot_mat():
    """
    获取 AprilTag 到机器人基座的变换矩阵
    
    注意：根据 furniture-bench/config.py 的注释：
        "tag_base_from_robot_base: This can be used to convert the pose 
         in the base tag's coordinate to the robot's coordinate."
        即：tag_base_from_robot_base 实际上是 AprilTag -> Robot 的变换
    
    Returns:
        4x4 numpy array: AprilTag -> Robot 的变换矩阵
    """
    ROBOT_HEIGHT = 0.0
    pos = (0.23 + 0.0715, 0, -ROBOT_HEIGHT)  # (0.3015, 0, 0)
    angles = (np.pi, 0, np.pi / 2)
    return get_mat(pos, angles)


def get_robot_to_april_mat():
    """
    获取机器人基座到 AprilTag 的变换矩阵
    
    这是 april_to_robot 的逆变换
    
    Returns:
        4x4 numpy array: Robot -> AprilTag 的变换矩阵
    """
    april_to_robot = get_april_to_robot_mat()
    return np.linalg.inv(april_to_robot)


def robot_pose_to_april_pose(ee_pos, ee_quat, device='cuda:0'):
    """
    将 EE 位姿从机器人基座坐标系转换到 AprilTag 坐标系
    
    Args:
        ee_pos: [3] torch.Tensor, EE 在机器人坐标系下的位置
        ee_quat: [4] torch.Tensor, EE 的四元数 (w, x, y, z) - PyTorch 格式
        device: torch device
        
    Returns:
        april_ee_pos: [3] torch.Tensor, EE 在 AprilTag 坐标系下的位置
        april_ee_quat: [4] torch.Tensor, EE 在 AprilTag 坐标系下的四元数 (w, x, y, z)
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'furniture-bench'))
    import utils.fb_control_utils as C
    
    # 获取转换矩阵
    robot_to_april_np = get_robot_to_april_mat()
    robot_to_april = torch.tensor(robot_to_april_np, device=device, dtype=torch.float32)
    
    # C.pose2mat 期望四元数格式为 (x, y, z, w)，需要转换
    # 输入: (w, x, y, z) -> 输出: (x, y, z, w)
    ee_quat_xyzw = torch.cat([ee_quat[1:], ee_quat[0:1]])
    
    # 构建 EE 在机器人坐标系下的齐次变换矩阵
    ee_pose_mat = C.pose2mat(ee_pos, ee_quat_xyzw, device)
    
    # 转换到 AprilTag 坐标系
    # AprilTag_T_EE = AprilTag_T_Robot @ Robot_T_EE
    april_ee_pose_mat = robot_to_april @ ee_pose_mat
    
    # 提取位置和四元数
    april_ee_pos, april_ee_quat = C.mat2pose(april_ee_pose_mat)
    # C.mat2pose 返回的四元数也是 (x, y, z, w) 格式，转回 (w, x, y, z)
    april_ee_quat_wxyz = torch.cat([april_ee_quat[3:4], april_ee_quat[0:3]])
    
    return april_ee_pos, april_ee_quat_wxyz


def transform_pcd_april_to_robot(pcd_april, device='cuda:0'):
    """
    将点云从 AprilTag 坐标系转换到机器人基座坐标系
    
    Args:
        pcd_april: [..., 3] torch.Tensor, AprilTag 坐标系下的点云
        device: torch device
        
    Returns:
        pcd_robot: [..., 3] torch.Tensor, 机器人坐标系下的点云
        
    Examples:
        >>> # 单个点云
        >>> pcd = torch.rand(1000, 3)  # [N, 3]
        >>> pcd_robot = transform_pcd_april_to_robot(pcd)
        
        >>> # 批量点云
        >>> pcd_batch = torch.rand(32, 4096, 3)  # [B, N, 3]
        >>> pcd_robot_batch = transform_pcd_april_to_robot(pcd_batch)
    """
    # 获取 AprilTag -> Robot 变换矩阵
    april_to_robot_np = get_april_to_robot_mat()
    april_to_robot = torch.tensor(april_to_robot_np, device=device, dtype=torch.float32)
    
    # 保存原始形状
    original_shape = pcd_april.shape
    
    # 展平为 [N, 3]
    pcd_flat = pcd_april.reshape(-1, 3)
    
    # 转换为齐次坐标 [N, 4]
    ones = torch.ones(pcd_flat.shape[0], 1, device=device, dtype=pcd_flat.dtype)
    pcd_homogeneous = torch.cat([pcd_flat, ones], dim=1)  # [N, 4]
    
    # 应用变换: [N, 4] @ [4, 4]^T = [N, 4]
    pcd_robot_homogeneous = pcd_homogeneous @ april_to_robot.T
    
    # 提取前3维并恢复原始形状
    pcd_robot = pcd_robot_homogeneous[:, :3].reshape(original_shape)
    
    return pcd_robot


def transform_pcd_robot_to_april(pcd_robot, device='cuda:0'):
    """
    将点云从机器人基座坐标系转换到 AprilTag 坐标系
    
    Args:
        pcd_robot: [..., 3] torch.Tensor, 机器人坐标系下的点云
        device: torch device
        
    Returns:
        pcd_april: [..., 3] torch.Tensor, AprilTag 坐标系下的点云
    """
    # 获取 Robot -> AprilTag 变换矩阵
    robot_to_april_np = get_robot_to_april_mat()
    robot_to_april = torch.tensor(robot_to_april_np, device=device, dtype=torch.float32)
    
    # 保存原始形状
    original_shape = pcd_robot.shape
    
    # 展平为 [N, 3]
    pcd_flat = pcd_robot.reshape(-1, 3)
    
    # 转换为齐次坐标 [N, 4]
    ones = torch.ones(pcd_flat.shape[0], 1, device=device, dtype=pcd_flat.dtype)
    pcd_homogeneous = torch.cat([pcd_flat, ones], dim=1)  # [N, 4]
    
    # 应用变换: [N, 4] @ [4, 4]^T = [N, 4]
    pcd_april_homogeneous = pcd_homogeneous @ robot_to_april.T
    
    # 提取前3维并恢复原始形状
    pcd_april = pcd_april_homogeneous[:, :3].reshape(original_shape)
    
    return pcd_april


if __name__ == '__main__':
    print("=" * 70)
    print("坐标系转换矩阵")
    print("=" * 70)
    
    # AprilTag -> Robot  
    april_to_robot = get_april_to_robot_mat()
    print("\nAprilTag -> Robot (tag_base_from_robot_base):")
    print(april_to_robot)
    print("注：根据 furniture-bench 注释，这个矩阵用于将 AprilTag 坐标转换到机器人坐标")
    
    # Robot -> AprilTag
    robot_to_april = get_robot_to_april_mat()
    print("\nRobot -> AprilTag (inv(tag_base_from_robot_base)):")
    print(robot_to_april)
    
    # 验证：应该是单位矩阵
    identity = april_to_robot @ robot_to_april
    print("\n验证 (应该是单位矩阵):")
    print(identity)
    print(f"是单位矩阵: {np.allclose(identity, np.eye(4))}")
    
    # 测试转换
    print("\n" + "=" * 70)
    print("测试坐标转换")
    print("=" * 70)
    
    import torch
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 示例：EE 在机器人坐标系下的位置
    ee_pos_robot = torch.tensor([0.5379, 0.0279, 0.1396], device=device, dtype=torch.float32)
    ee_quat_robot = torch.tensor([0.7948, -0.6030, -0.0678, 0.6047], device=device, dtype=torch.float32)  # (w,x,y,z)
    
    print(f"\nEE 在机器人坐标系下:")
    print(f"  位置: {ee_pos_robot}")
    print(f"  四元数: {ee_quat_robot}")
    
    # 转换到 AprilTag 坐标系
    april_ee_pos, april_ee_quat = robot_pose_to_april_pose(ee_pos_robot, ee_quat_robot, device)
    
    print(f"\nEE 在 AprilTag 坐标系下:")
    print(f"  位置: {april_ee_pos}")
    print(f"  四元数: {april_ee_quat}")
    
    print("\n预期：AprilTag 坐标系下的位置应该更接近零件位置（约 0-0.3 范围）")
