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
from utils.coordinate_transform import robot_pose_to_april_pose
from utils.gripper_pcd_utils import synthesize_gripper_pcd, build_gripper_pcd_in_hand_frame

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
        
        # 获取夹爪位姿（从 action/pos 和 robot_state）
        action_pos = data['action/pos'][i]  # [10]
        robot_state = data['robot_state'][i]  # [16]
        ee_pos_robot = torch.tensor(action_pos[:3], device=furniture.device, dtype=torch.float32)
        ee_rot_6d_robot = torch.tensor(action_pos[3:9], device=furniture.device, dtype=torch.float32)
        # 使用 robot_state 中的实际夹爪宽度，而不是 action 中的归一化值
        gripper_width = robot_state[15]  # 实际夹爪宽度（米）
        
        # ✅ 使用高级函数生成夹爪点云（封装了完整流程）
        gripper_pcd_world = synthesize_gripper_pcd(
            ee_pos_robot=ee_pos_robot,
            ee_rot_6d_robot=ee_rot_6d_robot,
            gripper_width=gripper_width,
            gripper_pcds=gripper_pcds,
            device=furniture.device,
            batch_size=N_ENVS
        )  # [N_env, N_total, 3]
        
        # 转换到 AprilTag 坐标系用于存储 ee_pose（用于可视化等）
        # ee_rot_mat = rotation_6d_to_matrix_simple(ee_rot_6d_robot)
        # ee_quat_robot = rotation_matrix_to_quaternion_simple(ee_rot_mat)
        # ee_pos_april, ee_quat_april = robot_pose_to_april_pose(
        #     ee_pos_robot, ee_quat_robot, furniture.device
        # )
        # ee_pose = torch.cat([ee_pos_april, ee_quat_april])  # [7]
        # part_pose['gripper'] = ee_pose.unsqueeze(0).expand(N_ENVS, -1)  # [N_env, 7]
        
        if i == 0:
            print_gpu_memory_usage(furniture.device, "begin to synthesize point cloud")
        
        if COMPUTE_FPS:
            torch.cuda.synchronize()
            start_time = time.perf_counter()

        # 处理家具点云合成
        furniture.get_pcd_from_offline_data(part_pose)
        
        if i == 0:
            print_gpu_memory_usage(furniture.device, "finish synthesize point cloud")

        if COMPUTE_FPS:
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            if i >= warmup_frames:
                timings.append(elapsed_time)

        # 提取第一个环境的点云（家具 + 夹爪）- 分别采样
        first_env_furniture = torch.cat(
            [batched_pcd[0] for batched_pcd in furniture.parts_pcds_world.values()],
            dim=0
        ).unsqueeze(0)  # [1, ~214k, 3]
        
        # gripper_pcd_world 现在是张量 [N_env, N_total, 3]
        first_env_gripper = gripper_pcd_world[0].unsqueeze(0)  # [1, N_total, 3]
        
        # 分别采样
        gripper_vis_sample = 256
        furniture_vis_sample = 4096 - gripper_vis_sample
        
        furniture_vis_sampled = sample_points(first_env_furniture, sample_num=furniture_vis_sample)
        gripper_vis_sampled = sample_points(first_env_gripper, sample_num=gripper_vis_sample)
        
        # 合并用于可视化：家具采样 + 夹爪采样
        pcds_sampled_vis = torch.cat([furniture_vis_sampled, gripper_vis_sampled], dim=1)
        pcd_animation_sequence.append(pcds_sampled_vis)
        # draw_point_cloud(pcds_sampled_vis) # 阻塞式，需要关掉窗口才能显示下一个点云
        if visualizer is not None:
            # 持续更新当前帧，直到不再暂停或窗口关闭
            while True:
                if not visualizer.update_point_cloud(pcds_sampled_vis):
                    # 窗口关闭
                    break
                if not visualizer.paused:
                    time.sleep(0.01)
                    # 未暂停，推进到下一帧
                    break
            
            # 如果窗口已关闭，退出主循环
            if not visualizer.keep_running:
                break
            
            time.sleep(0.01)
    
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