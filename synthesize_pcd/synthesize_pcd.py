from utils.read_dataset import read_zarr
from utils.get_pcd_from_npy import get_pcd_from_offline_data
import utils.fb_control_utils as C
import torch
from utils.furniture import Furniture, sample_points, draw_point_cloud, record_point_cloud_animation_imageio
import time
import numpy as np
from utils.visualizer import PointCloudVisualizer

DATA_PATH = "/home/rlg3/projects/6D-Manipulation/data/processed/diffik/sim/one_leg/teleop/low/success.zarr"
ASEET_PATH = 'synthesize_pcd/assets/furniture_bench/mesh/square_table'

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
    

    furniture = Furniture(ASEET_PATH, device='cuda:0', downsample_voxel_size=0.001)
    visualizer = PointCloudVisualizer()
    print_gpu_memory_usage(furniture.device, "Initial State")
    part_pose = {}
    pcd_animation_sequence = []
    timings = []  # 用于存储每一帧的处理时间
    warmup_frames = 20  # 前几帧可能较慢，不计入最终平均值
    for i in range(10000):

        # part_pose['square_table_top'] = torch.tensor(data['parts_poses'][i, 0:7], device=furniture.device)
        # part_pose['square_table_leg1'] = torch.tensor(data['parts_poses'][i, 7:14], device=furniture.device)
        # part_pose['square_table_leg2'] = torch.tensor(data['parts_poses'][i, 14:21], device=furniture.device)
        # part_pose['square_table_leg3'] = torch.tensor(data['parts_poses'][i, 21:28], device=furniture.device)
        # part_pose['square_table_leg4'] = torch.tensor(data['parts_poses'][i, 28:35], device=furniture.device)
        part_pose['square_table_top'] = torch.tensor(data['parts_poses'][i, 0:7], device=furniture.device).unsqueeze(0).expand(N_ENVS, -1)
        part_pose['square_table_leg1'] = torch.tensor(data['parts_poses'][i, 7:14], device=furniture.device).unsqueeze(0).expand(N_ENVS, -1)
        part_pose['square_table_leg2'] = torch.tensor(data['parts_poses'][i, 14:21], device=furniture.device).unsqueeze(0).expand(N_ENVS, -1)
        part_pose['square_table_leg3'] = torch.tensor(data['parts_poses'][i, 21:28], device=furniture.device).unsqueeze(0).expand(N_ENVS, -1)
        part_pose['square_table_leg4'] = torch.tensor(data['parts_poses'][i, 28:35], device=furniture.device).unsqueeze(0).expand(N_ENVS, -1)
        if i == 0:
            print_gpu_memory_usage(furniture.device, "begin to synthesize point cloud")
        
        if COMPUTE_FPS:
            # 强制 CPU 等待 GPU 完成上一帧的所有工作
            torch.cuda.synchronize()
            start_time = time.perf_counter()

        # 处理点云合成
        furniture.get_pcd_from_offline_data(part_pose)
        # 降采样
        pcds_sampled = sample_points(torch.cat(list(furniture.parts_pcds_world.values())), sample_num=4096)
        if i == 0:
            print_gpu_memory_usage(furniture.device, "finish synthesize point cloud")

        if COMPUTE_FPS:
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            if i >= warmup_frames:
                timings.append(elapsed_time)

        first_env_pcds_parts = {
            part_name: batched_pcd[0].unsqueeze(0)  # 使用索引 [0] 来选择第一个环境
            for part_name, batched_pcd in furniture.parts_pcds_world.items()
        }
        pcd_to_sample_single_env = torch.cat(list(first_env_pcds_parts.values()), dim=0)
        pcds_sampled = sample_points(pcd_to_sample_single_env, sample_num=1024)
        pcd_animation_sequence.append(pcds_sampled)
        # draw_point_cloud(pcds_sampled) # 阻塞式，需要关掉窗口才能显示下一个点云
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