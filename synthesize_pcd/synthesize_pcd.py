import sys
import os
# 添加 furniture-bench 和 juicer 到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'furniture-bench'))
sys.path.insert(0, '/home2/zxp/Projects/Juicer_ws/juicer')

from utils.read_dataset import read_zarr
from utils.get_pcd_from_npy import get_pcd_from_offline_data
import utils.fb_control_utils as C
import torch
from utils.furniture import Furniture, sample_points, draw_point_cloud, record_point_cloud_animation_imageio
import time
import numpy as np
from utils.visualizer import PointCloudVisualizer
from tqdm import tqdm

DATA_PATH = "/home2/zxp/Projects/Juicer_ws/juicer_dataset/processed/diffik/sim/one_leg/teleop/low/success.zarr"
# 使用绝对路径，确保从任何位置运行都能找到资源
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_PATH = os.path.join(SCRIPT_DIR, 'assets/furniture_bench/mesh/square_table')
MAX_GRIPPER_WIDTH = 0.065  # 最大夹爪宽度（米）

def print_gpu_memory_usage(device, message=""):
    """
    打印指定 GPU 的显存使用情况。
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GiB
        reserved = torch.cuda.memory_reserved(device) / (1024**3)   # GiB
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3) # GiB
        print(
            f"[{message}] "
            f"Allocated: {allocated:.2f} GiB | "
            f"Reserved: {reserved:.2f} GiB | "
            f"Peak Allocated: {peak_allocated:.2f} GiB"
        )

def rotation_matrix_to_quaternion_simple(rot_mat):
    """
    将旋转矩阵转换为四元数 (w, x, y, z)
    rot_mat: [3, 3] 旋转矩阵
    返回: [4] 四元数 (w, x, y, z)
    """
    trace = rot_mat[0, 0] + rot_mat[1, 1] + rot_mat[2, 2]
    
    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot_mat[2, 1] - rot_mat[1, 2]) * s
        y = (rot_mat[0, 2] - rot_mat[2, 0]) * s
        z = (rot_mat[1, 0] - rot_mat[0, 1]) * s
    elif rot_mat[0, 0] > rot_mat[1, 1] and rot_mat[0, 0] > rot_mat[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2])
        w = (rot_mat[2, 1] - rot_mat[1, 2]) / s
        x = 0.25 * s
        y = (rot_mat[0, 1] + rot_mat[1, 0]) / s
        z = (rot_mat[0, 2] + rot_mat[2, 0]) / s
    elif rot_mat[1, 1] > rot_mat[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2])
        w = (rot_mat[0, 2] - rot_mat[2, 0]) / s
        x = (rot_mat[0, 1] + rot_mat[1, 0]) / s
        y = 0.25 * s
        z = (rot_mat[1, 2] + rot_mat[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1])
        w = (rot_mat[1, 0] - rot_mat[0, 1]) / s
        x = (rot_mat[0, 2] + rot_mat[2, 0]) / s
        y = (rot_mat[1, 2] + rot_mat[2, 1]) / s
        z = 0.25 * s
    
    return torch.tensor([w, x, y, z], device=rot_mat.device, dtype=rot_mat.dtype)

def quat_apply_simple(quat, vec):
    """
    使用四元数旋转向量
    quat: [4] (w, x, y, z)
    vec: [3] (x, y, z)
    返回: [3] 旋转后的向量
    """
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    x, y, z = vec[0], vec[1], vec[2]
    
    # 四元数旋转公式
    ix = qw * x + qy * z - qz * y
    iy = qw * y + qz * x - qx * z
    iz = qw * z + qx * y - qy * x
    iw = -qx * x - qy * y - qz * z
    
    rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
    ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
    rz = iz * qw + iw * -qz + ix * -qy - iy * -qx
    
    return torch.tensor([rx, ry, rz], device=quat.device, dtype=quat.dtype)

def quat_mul_simple(q1, q2):
    """
    四元数乘法 (w, x, y, z)
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    
    return torch.tensor([w, x, y, z], device=q1.device, dtype=q1.dtype)

def compute_gripper_poses(ee_pos, ee_rot_6d, gripper_action, device):
    """
    根据 EE 位置、旋转和夹爪动作计算两个夹爪（left和right）的位姿
    
    Args:
        ee_pos: EE位置 [3] (x, y, z)
        ee_rot_6d: EE旋转（6D表示）[6]
        gripper_action: 夹爪动作 [-1: 开, 1: 闭]
        device: torch device
        
    Returns:
        left_finger_pose: [7] (pos[3] + quat[4])
        right_finger_pose: [7] (pos[3] + quat[4])
    """
    # 将6D旋转转换为旋转矩阵
    rot_6d = ee_rot_6d.reshape(2, 3)
    x = rot_6d[0] / torch.norm(rot_6d[0])
    y = rot_6d[1] - torch.dot(rot_6d[1], x) * x
    y = y / torch.norm(y)
    z = torch.cross(x, y)
    ee_rot_mat = torch.stack([x, y, z], dim=1)  # [3, 3]
    
    # 将旋转矩阵转换为四元数 (w, x, y, z)
    ee_quat = rotation_matrix_to_quaternion_simple(ee_rot_mat)  # [4]
    
    # EE 到 hand 的偏移（在hand坐标系下，EE在hand上方0.1m，根据mjx_panda.xml）
    # hand 的朝向需要考虑 quat="0.9238795 0 0 -0.3826834"
    hand_quat_offset = torch.tensor([0.9238795, 0.0, 0.0, -0.3826834], device=device, dtype=torch.float32)
    hand_quat = quat_mul_simple(ee_quat, hand_quat_offset)
    
    # hand位置 = ee位置 - ee_to_hand_offset（在EE坐标系下）
    ee_to_hand_in_ee_frame = torch.tensor([0.0, 0.0, -0.1], device=device, dtype=torch.float32)
    ee_to_hand_in_world = quat_apply_simple(ee_quat, ee_to_hand_in_ee_frame)
    hand_pos = ee_pos + ee_to_hand_in_world
    
    # finger base 到 hand 的偏移（在hand坐标系下，finger在hand上方0.0584m）
    hand_to_finger_in_hand_frame = torch.tensor([0.0, 0.0, 0.0584], device=device, dtype=torch.float32)
    hand_to_finger_in_world = quat_apply_simple(hand_quat, hand_to_finger_in_hand_frame)
    finger_base_pos = hand_pos + hand_to_finger_in_world
    
    # 计算夹爪开合程度
    # gripper_action: -1 表示开（max_width），1 表示闭（0）
    if gripper_action < 0:
        gripper_width = MAX_GRIPPER_WIDTH
    else:
        gripper_width = 0.0
    
    # 每个手指的偏移量是 gripper_width / 2
    finger_offset = gripper_width / 2
    
    # left finger 沿着 +Y 方向（在hand坐标系下）
    left_offset_in_hand = torch.tensor([0.0, finger_offset, 0.0], device=device, dtype=torch.float32)
    left_offset_in_world = quat_apply_simple(hand_quat, left_offset_in_hand)
    left_finger_pos = finger_base_pos + left_offset_in_world
    
    # right finger 沿着 -Y 方向（在hand坐标系下），同时旋转180度
    right_offset_in_hand = torch.tensor([0.0, -finger_offset, 0.0], device=device, dtype=torch.float32)
    right_offset_in_world = quat_apply_simple(hand_quat, right_offset_in_hand)
    right_finger_pos = finger_base_pos + right_offset_in_world
    
    # right finger 的朝向：相对于 hand 旋转180度（绕Z轴）
    right_quat_offset = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32)  # 180度绕Z轴
    right_finger_quat = quat_mul_simple(hand_quat, right_quat_offset)
    
    # 返回两个finger的位姿
    left_finger_pose = torch.cat([left_finger_pos, hand_quat])  # [7]
    right_finger_pose = torch.cat([right_finger_pos, right_finger_quat])  # [7]
    
    return left_finger_pose, right_finger_pose


def main():
    data = read_zarr(DATA_PATH)
    

    furniture = Furniture(ASSET_PATH, device='cuda:0', downsample_voxel_size=0.001)
    
    # 加载夹爪点云数据
    gripper_pcds = get_pcd_from_offline_data(
        ASSET_PATH, device='cuda:0'
    )
    print(f"加载了夹爪点云: {list(gripper_pcds.keys())}")
    if not RENDER:
        visualizer = None
    else:
        visualizer = PointCloudVisualizer()
    print_gpu_memory_usage(furniture.device, "Initial State")
    part_pose = {}
    pcd_animation_sequence = []
    timings = []  # 用于存储每一帧的处理时间
    warmup_frames = 20  # 前几帧可能较慢，不计入最终平均值
    for i in tqdm(range(10000)):
        # 获取家具部件位姿
        part_pose['square_table_top'] = torch.tensor(data['parts_poses'][i, 0:7], device=furniture.device).unsqueeze(0).expand(N_ENVS, -1)
        part_pose['square_table_leg1'] = torch.tensor(data['parts_poses'][i, 7:14], device=furniture.device).unsqueeze(0).expand(N_ENVS, -1)
        part_pose['square_table_leg2'] = torch.tensor(data['parts_poses'][i, 14:21], device=furniture.device).unsqueeze(0).expand(N_ENVS, -1)
        part_pose['square_table_leg3'] = torch.tensor(data['parts_poses'][i, 21:28], device=furniture.device).unsqueeze(0).expand(N_ENVS, -1)
        part_pose['square_table_leg4'] = torch.tensor(data['parts_poses'][i, 28:35], device=furniture.device).unsqueeze(0).expand(N_ENVS, -1)
        
        # 获取夹爪位姿（从 action/pos）
        action_pos = data['action/pos'][i]  # [10]
        ee_pos = torch.tensor(action_pos[:3], device=furniture.device, dtype=torch.float32)
        ee_rot_6d = torch.tensor(action_pos[3:9], device=furniture.device, dtype=torch.float32)
        gripper_action = action_pos[9]
        
        # 计算两个手指的位姿
        left_finger_pose, right_finger_pose = compute_gripper_poses(
            ee_pos, ee_rot_6d, gripper_action, furniture.device
        )
        
        # 添加夹爪位姿到part_pose字典
        part_pose['finger_0_left'] = left_finger_pose.unsqueeze(0).expand(N_ENVS, -1)
        part_pose['finger_0_right'] = right_finger_pose.unsqueeze(0).expand(N_ENVS, -1)
        if i == 0:
            print_gpu_memory_usage(furniture.device, "begin to synthesize point cloud")
        
        if COMPUTE_FPS:
            # 强制 CPU 等待 GPU 完成上一帧的所有工作
            torch.cuda.synchronize()
            start_time = time.perf_counter()

        # 处理家具点云合成
        furniture.get_pcd_from_offline_data(part_pose)
        
        # 处理夹爪点云
        gripper_pcds_world = {}
        # Left finger (两个部分：finger_0 和 finger_1)
        if 'finger_0' in gripper_pcds and 'finger_1' in gripper_pcds:
            # Transform left finger parts
            left_pose = part_pose['finger_0_left']
            left_pose_mat = C.batched_pose2mat(left_pose[:, :3], left_pose[:, 3:7], furniture.device)  # [N_env, 4, 4]
            
            # finger_0: [N_points, 4] 齐次坐标
            # 变换: [N_env, N_points, 4] = [N_points, 4] @ [N_env, 4, 4].T
            # 扩展gripper_pcds以匹配环境数量
            n_envs = left_pose_mat.shape[0]
            finger_0_expanded = gripper_pcds['finger_0'].unsqueeze(0).expand(n_envs, -1, -1)  # [N_env, N_points, 4]
            finger_1_expanded = gripper_pcds['finger_1'].unsqueeze(0).expand(n_envs, -1, -1)  # [N_env, N_points, 4]
            
            gripper_pcds_world['finger_0_left'] = torch.matmul(
                finger_0_expanded,  # [N_env, N_points, 4]
                left_pose_mat.transpose(1, 2)  # [N_env, 4, 4]
            )[:, :, :3]  # [N_env, N_points, 3] - 只取xyz坐标
            
            gripper_pcds_world['finger_1_left'] = torch.matmul(
                finger_1_expanded,  # [N_env, N_points, 4]
                left_pose_mat.transpose(1, 2)  # [N_env, 4, 4]
            )[:, :, :3]  # [N_env, N_points, 3]
            
            # Transform right finger parts
            right_pose = part_pose['finger_0_right']
            right_pose_mat = C.batched_pose2mat(right_pose[:, :3], right_pose[:, 3:7], furniture.device)  # [N_env, 4, 4]
            
            gripper_pcds_world['finger_0_right'] = torch.matmul(
                finger_0_expanded,  # [N_env, N_points, 4]
                right_pose_mat.transpose(1, 2)  # [N_env, 4, 4]
            )[:, :, :3]  # [N_env, N_points, 3]
            
            gripper_pcds_world['finger_1_right'] = torch.matmul(
                finger_1_expanded,  # [N_env, N_points, 4]
                right_pose_mat.transpose(1, 2)  # [N_env, 4, 4]
            )[:, :, :3]  # [N_env, N_points, 3]
        
        # 合并家具和夹爪点云
        all_pcds = list(furniture.parts_pcds_world.values()) + list(gripper_pcds_world.values())
        # 在 dim=1 (点的维度) 上拼接，shape: [N_env, total_points, 3]
        pcds_sampled = sample_points(torch.cat(all_pcds, dim=1), sample_num=4096)
        if i == 0:
            print_gpu_memory_usage(furniture.device, "finish synthesize point cloud")

        if COMPUTE_FPS:
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            if i >= warmup_frames:
                timings.append(elapsed_time)

        # 提取第一个环境的点云（家具 + 夹爪）
        first_env_pcds_parts = {
            part_name: batched_pcd[0]  # [N_points, 3]
            for part_name, batched_pcd in furniture.parts_pcds_world.items()
        }
        first_env_gripper_pcds = {
            part_name: batched_pcd[0]  # [N_points, 3]
            for part_name, batched_pcd in gripper_pcds_world.items()
        }
        all_first_env_pcds = list(first_env_pcds_parts.values()) + list(first_env_gripper_pcds.values())
        # 在 dim=0 (点的维度) 上拼接，shape: [total_points, 3]
        pcd_to_sample_single_env = torch.cat(all_first_env_pcds, dim=0).unsqueeze(0)  # [1, total_points, 3]
        pcds_sampled = sample_points(pcd_to_sample_single_env, sample_num=4096)
        pcd_animation_sequence.append(pcds_sampled)
        # draw_point_cloud(pcds_sampled) # 阻塞式，需要关掉窗口才能显示下一个点云
        if visualizer is not None:
            if visualizer.update_point_cloud(pcds_sampled): # 非阻塞式，循环更新点云
                time.sleep(0.01)
            else: 
                break
    
    print("\n--- Final Memory Usage ---")
    print_gpu_memory_usage(furniture.device, "Final State")

    if COMPUTE_FPS:
        timings_np = np.array(timings)
        avg_time_per_frame = np.mean(timings_np)
        std_dev_time = np.std(timings_np)
        fps = 1.0 / avg_time_per_frame
        print("\n--- 点云处理速率分析 ---")
        print(f"总计测量的有效帧数: {len(timings)}")
        print(f"平均处理时间/帧: {avg_time_per_frame * 1000:.2f} ms")
        print(f"时间标准差: {std_dev_time * 1000:.2f} ms")
        print(f"平均处理速率 (FPS): {fps:.2f} 帧/秒")
        print("--------------------------\n")

    if RENDER:
        if pcd_animation_sequence:
            record_point_cloud_animation_imageio(pcd_animation_sequence, output_path="synthesize_pcd/videos/synthesize_pcd.mp4", fps=30)
        else:
            print("没有生成任何点云数据用于播放。")

if __name__ == '__main__':
    RENDER = False
    N_ENVS = 1
    COMPUTE_FPS = True
    main()