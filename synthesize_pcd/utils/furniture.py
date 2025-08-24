import utils.fb_control_utils as C
import torch
import os
import glob
import numpy as np
import open3d as o3d
import time
import imageio.v2 as imageio

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

class Furniture:
    def __init__(self, asset_path: str, device: str='cpu', downsample_voxel_size=0.01):
        if not os.path.isdir(asset_path):
            print(f"Error: directory '{asset_path}' do not exist。")
            return {}
        
        pattern = os.path.join(asset_path, '*.npy')
        npy_file_paths = glob.glob(pattern)
        if not npy_file_paths:
            print(f"Warning: in directory '{asset_path}' do not find .npy files。")
            return {}

        self.downsample_voxel_size = downsample_voxel_size
        self.local_point_clouds_dict = {} # init a dict to storage pcd
        num_points = 0
        for file_path in npy_file_paths:
            filename_with_ext = os.path.basename(file_path)
            
            part_name = os.path.splitext(filename_with_ext)[0]
            
            point_cloud_data = np.load(file_path)

            # --- 下采样 ---
            if self.downsample_voxel_size > 0:
                pcd_o3d = o3d.geometry.PointCloud()
                pcd_o3d.points = o3d.utility.Vector3dVector(point_cloud_data)
                
                # 使用体素下采样来降低密度
                pcd_o3d_down = pcd_o3d.voxel_down_sample(self.downsample_voxel_size)
                point_cloud_data = np.asarray(pcd_o3d_down.points)
                print(f"  - {part_name} downsampled to {len(point_cloud_data)} points.")
            
            self.local_point_clouds_dict[part_name] = C.xyz_to_homogeneous(
                torch.tensor(
                    point_cloud_data, device=device, dtype=torch.float32
                ),
                device=device,
            )
            num_points += len(point_cloud_data)
            print(f"  - load: {part_name} (shape: {self.local_point_clouds_dict[part_name].shape}, homogeneous form)")
            print(f"total num of points: {num_points}")
        
        self.parts_pcds_world = {}
        self.device = device
    
    def get_pcd_from_offline_data(self, parts_poses: dict):
        furniture_pcds = []
        for part_name, local_pcd in self.local_point_clouds_dict.items():
            part_poses = parts_poses[part_name]
            part_pos, part_quat = part_poses[:, :3], part_poses[:, 3:]
            # quaternion_to_matrix assumes real part first
            part_quat = part_quat[..., [3, 0, 1, 2]]
            part_tf = C.batched_pose2mat(
                part_pos, part_quat, device=self.device
            )  # (num_envs, 4, 4)
            part_pcd_transformed = part_tf @ local_pcd.T  # (num_envs, 4, n_points)
            part_pcd_transformed = part_pcd_transformed.transpose(1, 2)[
               :, :, :3
            ]  # (num_envs, n_points, 3)
            furniture_pcds.append(part_pcd_transformed)
        furniture_pcds = torch.cat(furniture_pcds, dim=1)  # (num_envs, n_points, 3)
        self.parts_pcds_world[part_name] = furniture_pcds

        # # transform pcd coordinate into Franka base frame
        # base_state = state_dict["franka_base"][:, :3].unsqueeze(1)  # (B, 1, 3)
        # pcds_full = pcds_full - base_state

def sample_points(points, sample_num: int):
    """
    points: (num_envs, n_points, 3)
    """
    sampling_idx = torch.randperm(points.shape[1])[:sample_num]
    sampled_points = points[:, sampling_idx, :]
    return sampled_points

def draw_point_cloud(pcd_tensor, window_name="Point Cloud"):
    """
    使用 Open3D 可视化 PyTorch 点云张量。
    
    :param pcd_tensor: PyTorch 张量，形状为 (N, 3)。
    :param window_name: 可视化窗口的标题。
    """
    print("\n正在准备可视化...")
    # 1. 将数据从 GPU 移至 CPU，并转换为 NumPy 数组
    points_np = pcd_tensor.cpu().numpy()
    
    # 2. 创建 Open3D 的 PointCloud 对象
    pcd_o3d = o3d.geometry.PointCloud()
    
    # 3. 将 NumPy 点云数据赋值给 Open3D 对象
    pcd_o3d.points = o3d.utility.Vector3dVector(points_np)
    
    # 4. (可选) 为点云上色以获得更好的视觉效果
    # pcd_o3d.paint_uniform_color([0.1, 0.7, 0.3]) # 绿色
    
    # 5. 创建坐标系并显示
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    print("按 'q' 关闭可视化窗口。")
    o3d.visualization.draw_geometries([pcd_o3d, coord_frame], window_name=window_name)

def record_point_cloud_animation_imageio(pcd_sequence, output_path="output_animation.mp4", fps=30):
    """
    将点云序列使用 imageio 录制成视频文件。

    :param pcd_sequence: 一个列表，每个元素都是 (N, 3) 的 PyTorch 点云张量。
    :param output_path: 输出视频文件的路径 (如 'animation.mp4')。
    :param fps: 视频的帧率。
    """
    print(f"\n准备使用 imageio 录制视频到: {output_path}")
    
    # --- 第 1 步: 捕获每一帧为图片 (这部分和之前完全相同) ---
    
    temp_frame_dir = "temp_frames"
    if not os.path.exists(temp_frame_dir):
        os.makedirs(temp_frame_dir)
    else:
        files = glob.glob(os.path.join(temp_frame_dir, '*.png'))
        for f in files: os.remove(f)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Video Recording", width=1280, height=720)
    
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_sequence[0].squeeze(0).cpu().numpy())
    vis.add_geometry(pcd_o3d)

    print(f"正在捕获 {len(pcd_sequence)} 帧图片...")
    for i, pcd_tensor in enumerate(pcd_sequence):
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_tensor.squeeze(0).cpu().numpy())
        vis.update_geometry(pcd_o3d)
        vis.poll_events()
        vis.update_renderer()
        frame_filename = os.path.join(temp_frame_dir, f"frame_{i:05d}.png")
        vis.capture_screen_image(frame_filename, do_render=True)

    vis.destroy_window()
    print("帧图片捕获完成。")

    # --- 第 2 步: 使用 imageio 将图片序列合成为视频 (替换 cv2 的部分) ---
    
    print("正在使用 imageio 将帧合成为视频...")
    
    frame_files = sorted(glob.glob(os.path.join(temp_frame_dir, '*.png')))
    if not frame_files:
        print("错误：没有找到任何帧图片。")
        return

    # 使用 imageio.get_writer，'with' 语句会自动处理关闭
    with imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8) as writer:
        for i, frame_file in enumerate(frame_files):
            # 使用 imageio 读取图片
            image = imageio.imread(frame_file)
            # 将图片数据添加到视频写入器
            writer.append_data(image)
            
            # (可选) 打印进度
            if (i + 1) % 50 == 0:
                print(f"  ...已合成 {i + 1}/{len(frame_files)} 帧")
    
    print(f"视频成功保存到: {output_path}")

    # --- 第 3 步: 清理临时文件 (这部分和之前完全相同) ---
    
    print("正在清理临时帧文件...")
    for frame_file in frame_files:
        os.remove(frame_file)
    os.rmdir(temp_frame_dir)
    print("清理完成。")