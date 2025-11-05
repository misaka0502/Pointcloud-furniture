import sys
import os
# 添加 furniture-bench 和 juicer 到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'furniture-bench'))
sys.path.insert(0, '/home2/zxp/Projects/Juicer_ws/juicer')

from utils.read_dataset import read_zarr
from utils.get_pcd_from_npy import get_pcd_from_offline_data
import utils.fb_control_utils as C
import torch
from utils.furniture import Furniture, sample_points
from utils.visualizer import PointCloudVisualizer
from tqdm import tqdm
from utils.gripper_pcd_utils import synthesize_gripper_pcd
import time
import argparse

# 默认数据路径
DEFAULT_DATA_PATH = "/home/rlg3/projects/6D-Manipulation/data/processed/diffik/sim/one_leg/teleop/low/success.zarr"
# 使用绝对路径，确保从任何位置运行都能找到资源
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_PATH = os.path.join(SCRIPT_DIR, 'assets/furniture_bench/mesh/square_table')

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
    
    # 获取总帧数
    total_frames = len(data['parts_poses'])
    if num_frames is None:
        num_frames = total_frames
    else:
        num_frames = min(num_frames, total_frames)
    
    print(f"总帧数: {total_frames}, 将播放: {num_frames} 帧")
    
    # 初始化家具点云加载器
    print(f"正在加载家具资产从: {ASSET_PATH}")
    furniture = Furniture(ASSET_PATH, device=device, downsample_voxel_size=0.001)
    
    # 加载夹爪点云数据
    print("正在加载夹爪点云...")
    gripper_pcds = get_pcd_from_offline_data(ASSET_PATH, device=device)
    print(f"加载了夹爪点云: {list(gripper_pcds.keys())}")
    
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
        
        # 准备家具部件位姿字典
        part_pose = {}
        part_pose['square_table_top'] = torch.tensor(data['parts_poses'][i, 0:7], device=device).unsqueeze(0)
        part_pose['square_table_leg1'] = torch.tensor(data['parts_poses'][i, 7:14], device=device).unsqueeze(0)
        part_pose['square_table_leg2'] = torch.tensor(data['parts_poses'][i, 14:21], device=device).unsqueeze(0)
        part_pose['square_table_leg3'] = torch.tensor(data['parts_poses'][i, 21:28], device=device).unsqueeze(0)
        part_pose['square_table_leg4'] = torch.tensor(data['parts_poses'][i, 28:35], device=device).unsqueeze(0)
        
        # 获取夹爪位姿
        action_pos = data['action/pos'][i]
        robot_state = data['robot_state'][i]
        ee_pos_robot = torch.tensor(robot_state[:3], device=device, dtype=torch.float32)
        ee_rot_6d_robot = torch.tensor(robot_state[3:9], device=device, dtype=torch.float32)
        gripper_width = robot_state[15]  # 实际夹爪宽度（米）
        
        # 生成家具点云
        furniture.get_pcd_from_offline_data(part_pose)
        
        # 生成夹爪点云
        gripper_pcd_world = synthesize_gripper_pcd(
            ee_pos_robot=ee_pos_robot,
            ee_rot_6d_robot=ee_rot_6d_robot,
            gripper_width=gripper_width,
            gripper_pcds=gripper_pcds,
            device=device,
            batch_size=1
        )  # [1, N_total, 3]
        
        # 提取第一个环境的点云（家具 + 夹爪）
        first_env_furniture = torch.cat(
            [batched_pcd[0] for batched_pcd in furniture.parts_pcds_world.values()],
            dim=0
        ).unsqueeze(0)  # [1, ~214k, 3]
        
        first_env_gripper = gripper_pcd_world[0].unsqueeze(0)  # [1, N_total, 3]
        
        # 分别采样以控制点云密度
        gripper_vis_sample = 512
        furniture_vis_sample = 4096 - gripper_vis_sample
        
        furniture_vis_sampled = sample_points(first_env_furniture, sample_num=furniture_vis_sample)
        gripper_vis_sampled = sample_points(first_env_gripper, sample_num=gripper_vis_sample)
        
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