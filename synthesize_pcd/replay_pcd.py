import sys
import os
# 添加 furniture-bench 和 juicer 到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'furniture-bench'))
sys.path.insert(0, '/home2/zxp/Projects/Juicer_ws/juicer')

from utils.read_dataset import read_zarr
import torch
from utils.furniture import sample_points
from utils.visualizer import PointCloudVisualizer
from tqdm import tqdm
import time
import argparse
import numpy as np

# 默认数据路径
DEFAULT_DATA_PATH = "/home/rlg3/projects/6D-Manipulation/data/one_leg_seperate_part_with_gripper.zarr"

def replay_point_cloud_animation(zarr_path, num_frames=None, fps=30, device='cuda:0', skip_frames=1):
    """
    从 zarr 文件中读取数据并逐帧可视化家具和夹爪点云。
    
    Args:
        zarr_path (str): zarr 数据文件的路径
        num_frames (int): 要播放的帧数，如果为 None 则播放所有帧
        fps (int): 播放帧率（用于控制播放速度）
        device (str): 使用的设备 ('cuda:0' 或 'cpu')
        skip_frames (int): 跳帧数，1 表示不跳帧，2 表示每隔一帧播放一次
    """
    print(f"正在加载数据从: {zarr_path}")
    data = read_zarr(zarr_path)
    
    # 检查数据集中是否有预生成的点云
    has_pregenerated_pcds = 'furniture_pcds' in data and 'gripper_pcds' in data
    
    if has_pregenerated_pcds:
        print("✓ 检测到预生成的点云数据，将直接读取")
        # 获取总帧数
        total_frames = len(data['gripper_pcds'])
    else:
        print("✗ 未检测到预生成的点云数据")
        print("请使用包含 'furniture_pcds' 和 'gripper_pcds' 的数据集")
        return
    
    if num_frames is None:
        num_frames = total_frames
    else:
        num_frames = min(num_frames, total_frames)
    
    print(f"总帧数: {total_frames}, 将播放: {num_frames} 帧")
    
    # 打印点云数据信息
    print("\n点云数据信息:")
    print(f"  - 夹爪点云: {data['gripper_pcds'].shape}")
    if 'furniture_pcds' in data:
        for part_name in data['furniture_pcds'].keys():
            print(f"  - {part_name}: {data['furniture_pcds'][part_name].shape}")
    
    # 初始化可视化器
    print("正在初始化可视化器...")
    visualizer = PointCloudVisualizer(window_name="Point Cloud Replay")
    
    # 计算帧间延迟（用于控制播放速度）
    frame_delay = 1.0 / fps
    
    print("\n开始播放点云动画...")
    print("  - 按 'S' 键暂停/继续")
    print("  - 按 'ESC' 键退出\n")
    
    # 逐帧处理和可视化
    frame_indices = range(0, num_frames, skip_frames)
    for i in tqdm(frame_indices, desc="播放进度"):
        start_time = time.time()
        
        # 直接从数据集读取预生成的点云
        # 读取夹爪点云 [636, 3]
        gripper_pcd = torch.tensor(data['gripper_pcds'][i], device=device, dtype=torch.float32)
        
        # 读取并合并所有家具部件点云
        furniture_pcds_list = []
        for part_name in ['square_table_top', 'square_table_leg1', 'square_table_leg2', 
                          'square_table_leg3', 'square_table_leg4']:
            if part_name in data['furniture_pcds']:
                part_pcd = torch.tensor(data['furniture_pcds'][part_name][i], 
                                       device=device, dtype=torch.float32)
                furniture_pcds_list.append(part_pcd)
        
        # 合并所有家具点云 [N_furniture, 3]
        furniture_pcd = torch.cat(furniture_pcds_list, dim=0)
        
        # 添加 batch 维度
        furniture_pcd = furniture_pcd.unsqueeze(0)  # [1, N_furniture, 3]
        gripper_pcd = gripper_pcd.unsqueeze(0)  # [1, N_gripper, 3]
        
        # 采样以控制点云密度
        gripper_vis_sample = 512
        furniture_vis_sample = 4096 - gripper_vis_sample
        
        furniture_vis_sampled = sample_points(furniture_pcd, sample_num=furniture_vis_sample)
        gripper_vis_sampled = sample_points(gripper_pcd, sample_num=gripper_vis_sample)
        
        # 合并用于可视化：家具采样 + 夹爪采样
        pcds_sampled_vis = torch.cat([furniture_vis_sampled, gripper_vis_sampled], dim=1)
        
        # 更新可视化器
        while True:
            if not visualizer.update_point_cloud(pcds_sampled_vis):
                # 窗口关闭
                break
            if not visualizer.paused:
                # 未暂停，推进到下一帧
                break
        
        # 如果窗口已关闭，退出主循环
        if not visualizer.keep_running:
            print("\n可视化器已关闭，停止播放。")
            break
        
        # 控制帧率
        elapsed = time.time() - start_time
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)
    
    print("\n播放完成！")
    visualizer.close()

def main():
    parser = argparse.ArgumentParser(description='播放点云动画')
    parser.add_argument('--zarr_path', type=str, default=DEFAULT_DATA_PATH,
                        help='zarr 数据文件的路径')
    parser.add_argument('--num_frames', type=int, default=None,
                        help='要播放的帧数（默认全部）')
    parser.add_argument('--fps', type=int, default=30,
                        help='播放帧率')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='使用的设备 (cuda:0 或 cpu)')
    parser.add_argument('--skip_frames', type=int, default=1,
                        help='跳帧数，1表示不跳帧，2表示每隔一帧播放')
    
    args = parser.parse_args()
    
    replay_point_cloud_animation(
        zarr_path=args.zarr_path,
        num_frames=args.num_frames,
        fps=args.fps,
        device=args.device,
        skip_frames=args.skip_frames
    )

if __name__ == '__main__':
    main()