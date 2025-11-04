#!/usr/bin/env python3
"""检查 robot_state 中的 gripper_width"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'furniture-bench'))
sys.path.insert(0, '/home2/zxp/Projects/Juicer_ws/juicer')

from utils.read_dataset import read_zarr

DATA_PATH = "/home2/zxp/Projects/Juicer_ws/juicer_dataset/processed/diffik/sim/one_leg/teleop/low/success.zarr"
data = read_zarr(DATA_PATH)

print("=" * 70)
print("检查 robot_state 中的 gripper_width")
print("=" * 70)

# robot_state 的维度
print(f"\nrobot_state shape: {data['robot_state'].shape}")
print(f"  → 每个 state 有 {data['robot_state'].shape[1]} 个值")

# 检查前几帧
print(f"\n前5帧的数据:")
for i in range(5):
    robot_state = data['robot_state'][i]
    action_pos = data['action/pos'][i]
    
    print(f"\n第 {i+1} 帧:")
    print(f"  robot_state: {robot_state}")
    print(f"  action/pos: {action_pos}")
    print(f"  action gripper: {action_pos[9]}")
    
    # 根据 furniture_rl_sim_env.py 的 _read_robot_state 顺序
    # robot_state 包含: ee_pos(3) + ee_quat(4) + ee_pos_vel(3) + ee_ori_vel(3) + gripper_width(1)
    # 或者可能是其他顺序
    
# 尝试查找 gripper_width 的位置
print("\n" + "=" * 70)
print("分析 robot_state 的结构")
print("=" * 70)

# 查看数值范围，gripper_width 应该在 0 到 max_gripper_width (0.065) 之间
for i in range(data['robot_state'].shape[1]):
    values = data['robot_state'][:100, i]  # 前100帧
    print(f"\nrobot_state[{i}]:")
    print(f"  范围: [{values.min():.4f}, {values.max():.4f}]")
    print(f"  平均: {values.mean():.4f}")
    
    # gripper_width 应该在 0-0.065 范围内
    if 0 <= values.min() and values.max() <= 0.1:
        print(f"  → 可能是 gripper_width! ✓")
