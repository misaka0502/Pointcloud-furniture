import furniture_bench.controllers.control_utils as C
import utils.fb_control_utils as CC
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
            part_tf = CC.batched_pose2mat(
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
    points_np = pcd_tensor.squeeze(0).cpu().numpy()
    
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

def get_wrist_camera_pose(
    ee_pos: torch.Tensor, 
    ee_quat: torch.Tensor,
    camera_local_mat: torch.Tensor
) -> torch.Tensor:
    """
    根据末端执行器(EE)的位姿，计算附加在其上的腕部相机的世界位姿。

    此函数假设相机相对于EE的局部变换是固定的，
    其值为 p=(-0.04, 0, -0.05) 和 r=绕Y轴旋转-70度。

    Args:
        ee_pos (torch.Tensor): 末端执行器在世界坐标系中的位置，形状为 (3,)。
        ee_quat (torch.Tensor): 末端执行器在世界坐标系中的旋转四元数，
                                形状为 (4,)，顺序为 (x, y, z, w)。

    Returns:
        torch.Tensor: 腕部相机在世界坐标系下的4x4位姿矩阵。
    """
    
    # 将输入的EE位姿转换为一个 4x4 的世界变换矩阵
    
    ee_world_mat = torch.eye(4, device=ee_pos.device, dtype=torch.float32)
    ee_world_rot_mat = CC.quat2mat(ee_quat.squeeze(0))

    ee_world_mat[:3, :3] = ee_world_rot_mat
    # 填充平移部分
    ee_world_mat[:3, 3] = ee_pos

    # 矩阵相乘，得到最终的相机世界位姿
    # 公式: Camera_World_Pose = EE_World_Pose * Camera_Local_Transform
    wrist_camera_pose = ee_world_mat @ camera_local_mat

    # 将相机的位姿绕其本地坐标系Y轴旋转90度修正，虽然不知道为什么，但是得这样做
    angle_rad = np.radians(90)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    
    device = wrist_camera_pose.device
    
    # 创建一个绕本地Y轴旋转的4x4变换矩阵
    rotation_y_local = torch.tensor([
        [c,  0,  s,  0],
        [0,  1,  0,  0],
        [-s, 0,  c,  0],
        [0,  0,  0,  1]
    ], device=device, dtype=torch.float32)
    # 创建一个绕本地Z轴旋转的4x4变换矩阵
    rotation_z_local = torch.tensor([
        [c, -s,  0,  0],
        [s,  c,  0,  0],
        [0,  0,  1,  0],
        [0,  0,  0,  1]
    ], device=device, dtype=torch.float32)
    
    # 右乘
    # PyTorch的广播机制会自动处理 (B, 4, 4) @ (4, 4) -> (B, 4, 4)
    wrist_camera_pose = wrist_camera_pose @ rotation_y_local @ rotation_z_local
    
    return wrist_camera_pose

def get_wrist_camera_pcd(full_pcd_world: torch.Tensor, camera_pose_world: torch.Tensor, camera_intrinsics: dict, z_near: float=0.07, z_far: float=0.5):
    """
    根据腕部相机的位姿和视野，从世界点云中获取其视野内的点云。

    Args:
        world_pcd (torch.Tensor): 世界坐标系下的完整点云，形状为 (num_envs, N, 3)。
        camera_pose_world (torch.Tensor): 腕部相机在世界坐标系下的位姿，
                                          表示从相机坐标系到世界坐标系的变换。
                                          形状为 (num_envs, 4, 4)。
        intrinsics (dict): 相机内参，包含 'width', 'height', 'fx', 'fy', 'cx', 'cy'。
        z_near (float): 相机视锥体的近裁剪平面距离。
        z_far (float): 相机视锥体的远裁剪平面距离。

    Returns:
        torch.Tensor: 经过腕部相机视野裁剪后，仍在世界坐标系下的点云。
                      形状为 (num_envs, M, 3)，M 是可见点的数量。
    """
    device = full_pcd_world.device
    
    # transform world pcd to camera frame
    # camera_pose_world 是从相机坐标系 -> 世界坐标系的变换
    world_to_camera_tf = torch.inverse(camera_pose_world)
    # 将点云转换为齐次坐标 (N, 4)
    world_pcd_homo = C.xyz_to_homogeneous(full_pcd_world, device)
    # 应用变换 (4, 4) @ (4, N) -> (4, N)
    pcd_camera_frame_homo = world_to_camera_tf @ world_pcd_homo.T.squeeze(-1)

    # 转换回非齐次坐标 (N, 3)
    pcd_camera_frame = pcd_camera_frame_homo.T[:, :3]

    # 视椎体裁剪 Frustum Culling
    X, Y, Z = pcd_camera_frame[..., 0], pcd_camera_frame[..., 1], pcd_camera_frame[..., 2]
    # 深度裁剪: 只保留在相机近裁平面和远裁平面之间的点
    depth_mask = (Z > z_near) & (Z < z_far)
    # 视野裁剪: 将3D点云投影到2D像素平面
    u = camera_intrinsics['fx'] * X / Z + camera_intrinsics['cx']
    v = camera_intrinsics['fy'] * Y / Z + camera_intrinsics['cy']
    # 只保留投影后再图像范围内的点
    fov_mask = (u >= 0) & (u < camera_intrinsics['width']) & (v >= 0) & (v < camera_intrinsics['height'])
    # 合并所有掩码
    visible_mask = depth_mask & fov_mask

    # 提取所有可见的点
    visible_pcd_camera_frame = pcd_camera_frame[visible_mask]

    # 将可见点云从相机坐标系转换回世界坐标系
    # 确保张量是二维的，以便后续处理
    if visible_pcd_camera_frame.ndim == 1:
        visible_pcd_camera_frame = visible_pcd_camera_frame.unsqueeze(0)

    # 如果没有可见点，则返回空张量
    if visible_pcd_camera_frame.shape[0] == 0:
        return torch.empty((0, 3), device=device, dtype=torch.float32)
    
    # 转换回齐次坐标
    visible_pcd_camera_frame_homo = C.xyz_to_homogeneous(visible_pcd_camera_frame, device)

    # 应用原始的相机位姿矩阵，将其变换回世界坐标系
    visible_pcd_world_homo = camera_pose_world @ visible_pcd_camera_frame_homo.T

    # 转换回非齐次坐标
    visible_pcd_world = visible_pcd_world_homo.T[:, :3]

    return visible_pcd_world

def get_wrist_camera_pcd_from_view_matrix(
    world_pcd: torch.Tensor,             # 形状: (B, N, 3)
    camera_view_matrix: torch.Tensor,    # 形状: (B, 4, 4) <-- 直接使用视图矩阵
    intrinsics: dict,
    z_near: float = 0.07,
    z_far: float = 0.5,
) -> torch.Tensor:
    """
    根据腕部相机的【视图矩阵】和视野，从世界点云中获取其视野内的点云。
    
    Args:
        world_pcd (torch.Tensor): 世界坐标系下的完整点云。
        camera_view_matrix (torch.Tensor): 腕部相机在世界坐标系下的【视图矩阵】。
        intrinsics (dict): 相机内参。
        z_near (float): 相机视锥体的近裁剪平面距离。
        z_far (float): 相机视锥体的远裁剪平面距离。

    Returns:
        torch.Tensor: 裁剪后，仍在【世界坐标系】下的点云。
    """
    B, N, _ = world_pcd.shape
    device = world_pcd.device

    # --- 步骤 1: 将世界点云变换到相机坐标系 ---
    
    # 准备齐次坐标的世界点云，形状 (B, N, 4) -> (B, 4, N)
    ones = torch.ones((B, N, 1), device=device, dtype=torch.float32)
    world_pcd_homo = torch.cat([world_pcd, ones], dim=-1).transpose(1, 2)

    # 【核心改动】直接使用视图矩阵进行变换，无需再求逆
    # (B, 4, 4) @ (B, 4, N) -> (B, 4, N)
    points_in_camera_frame_homo = camera_view_matrix @ world_pcd_homo
    
    # 转换回非齐次坐标 (B, N, 3)
    points_in_camera_frame = points_in_camera_frame_homo.transpose(1, 2)[:, :, :3]

    # --- 步骤 2: 视锥体裁剪 (Frustum Culling) - 这部分完全不变 ---
    
    X, Y, Z = points_in_camera_frame[..., 0], points_in_camera_frame[..., 1], points_in_camera_frame[..., 2]

    # a. 深度裁剪
    depth_mask = (Z > z_near) & (Z < z_far)

    # b. 视野裁剪
    u = intrinsics['fx'] * X / Z + intrinsics['cx']
    v = intrinsics['fy'] * Y / Z + intrinsics['cy']
    fov_mask = (u >= 0) & (u < intrinsics['width']) & (v >= 0) & (v < intrinsics['height'])

    # 合并掩码，形状 (B, N)
    visible_mask = depth_mask & fov_mask
    
    # --- 步骤 3: 将可见点云变换回世界坐标系 ---
    # 为了将结果返回到世界坐标系（方便与其他数据一起使用），我们仍然需要位姿矩阵。
    # 位姿矩阵 = 视图矩阵的逆
    
    camera_pose_matrix = torch.inverse(camera_view_matrix)
    # 我们需要一种高效的方式来只拾取可见点并进行反向变换。
    # 使用 advanced indexing 会使代码复杂化。一个更简洁的方法是：
    # 1. 创建一个完整的、变换回世界坐标系的点云。
    # 2. 使用掩码来筛选它。
    
    # (B, 4, 4) @ (B, 4, N) -> (B, 4, N)
    visible_points_world_homo_all = camera_pose_matrix @ points_in_camera_frame_homo
    visible_points_world_all = visible_points_world_homo_all.transpose(1, 2)[:, :, :3]
    
    # 使用掩码筛选出最终的点云
    # 注意：这里的 visible_mask 是 (B, N)，visible_points_world_all 是 (B, N, 3)
    # 筛选后的结果是一个列表，每个元素是每个环境中的可见点云
    visible_pcd_world = visible_points_world_all[visible_mask]

    return visible_pcd_world, camera_pose_matrix