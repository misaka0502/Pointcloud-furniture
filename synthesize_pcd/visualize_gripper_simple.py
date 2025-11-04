#!/usr/bin/env python3
"""简单可视化夹爪"""
import sys
sys.path.insert(0, '/home2/zxp/Projects/Juicer_ws/juicer')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载 finger 点云
finger_0 = np.load("assets/furniture_bench/mesh/square_table/finger_0.npy")
finger_1 = np.load("assets/furniture_bench/mesh/square_table/finger_1.npy")

# 中心化 Z 坐标
def center_z(pcd):
    pcd = pcd.copy()
    z_min = pcd[:, 2].min()
    z_max = pcd[:, 2].max()
    z_center = (z_min + z_max) / 2
    pcd[:, 2] -= z_center
    return pcd

finger_0_centered = center_z(finger_0)
finger_1_centered = center_z(finger_1)

print("finger_0 范围:")
print(f"  X: [{finger_0_centered[:, 0].min():.4f}, {finger_0_centered[:, 0].max():.4f}]")
print(f"  Y: [{finger_0_centered[:, 1].min():.4f}, {finger_0_centered[:, 1].max():.4f}]")
print(f"  Z: [{finger_0_centered[:, 2].min():.4f}, {finger_0_centered[:, 2].max():.4f}]")

print("\nfinger_1 范围:")
print(f"  X: [{finger_1_centered[:, 0].min():.4f}, {finger_1_centered[:, 1].max():.4f}]")
print(f"  Y: [{finger_1_centered[:, 1].min():.4f}, {finger_1_centered[:, 1].max():.4f}]")
print(f"  Z: [{finger_1_centered[:, 2].min():.4f}, {finger_1_centered[:, 2].max():.4f}]")

# 模拟左右手指（沿 Y 轴偏移）
offset = 0.0325  # 一半的夹爪宽度
left_finger_0 = finger_0_centered.copy()
left_finger_0[:, 1] += offset
left_finger_1 = finger_1_centered.copy()
left_finger_1[:, 1] += offset

right_finger_0 = finger_0_centered.copy()
right_finger_0[:, 1] -= offset
right_finger_1 = finger_1_centered.copy()
right_finger_1[:, 1] -= offset

# 可视化
fig = plt.figure(figsize=(15, 5))

# 视图1：从前方看 (X-Z 平面)
ax1 = fig.add_subplot(131)
ax1.scatter(left_finger_0[:, 0], left_finger_0[:, 2], c='r', s=5, alpha=0.5, label='left finger_0')
ax1.scatter(left_finger_1[:, 0], left_finger_1[:, 2], c='darkred', s=5, alpha=0.5, label='left finger_1')
ax1.scatter(right_finger_0[:, 0], right_finger_0[:, 2], c='b', s=5, alpha=0.5, label='right finger_0')
ax1.scatter(right_finger_1[:, 0], right_finger_1[:, 2], c='darkblue', s=5, alpha=0.5, label='right finger_1')
ax1.set_xlabel('X')
ax1.set_ylabel('Z')
ax1.set_title('从前方看 (X-Z平面)')
ax1.legend()
ax1.grid(True)
ax1.axis('equal')

# 视图2：从上方看 (X-Y 平面)
ax2 = fig.add_subplot(132)
ax2.scatter(left_finger_0[:, 0], left_finger_0[:, 1], c='r', s=5, alpha=0.5, label='left finger_0')
ax2.scatter(left_finger_1[:, 0], left_finger_1[:, 1], c='darkred', s=5, alpha=0.5, label='left finger_1')
ax2.scatter(right_finger_0[:, 0], right_finger_0[:, 1], c='b', s=5, alpha=0.5, label='right finger_0')
ax2.scatter(right_finger_1[:, 0], right_finger_1[:, 1], c='darkblue', s=5, alpha=0.5, label='right finger_1')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('从上方看 (X-Y平面)')
ax2.legend()
ax2.grid(True)
ax2.axis('equal')

# 视图3：3D
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(left_finger_0[:, 0], left_finger_0[:, 1], left_finger_0[:, 2], c='r', s=1, alpha=0.5, label='left')
ax3.scatter(left_finger_1[:, 0], left_finger_1[:, 1], left_finger_1[:, 2], c='darkred', s=1, alpha=0.5)
ax3.scatter(right_finger_0[:, 0], right_finger_0[:, 1], right_finger_0[:, 2], c='b', s=1, alpha=0.5, label='right')
ax3.scatter(right_finger_1[:, 0], right_finger_1[:, 1], right_finger_1[:, 2], c='darkblue', s=1, alpha=0.5)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('3D 视图')
ax3.legend()

plt.tight_layout()
plt.savefig('gripper_visualization.png', dpi=150)
print(f"\n图像已保存到: gripper_visualization.png")
print("\n如果左右手指看起来不像夹子，可能需要对右手指进行镜像变换")
