import open3d as o3d
import torch
import numpy as np
import time
import matplotlib.cm as cm

def colorize_pcd_by_height(points_np, colormap='jet'):
    """
    根据点云的高度（Z值）为其着色。
    
    :param points_np: NumPy 数组，形状为 (N, 3)。
    :param colormap: Matplotlib 的颜色图谱名称，'jet' 或 'viridis' 效果很好。
    :return: NumPy 数组，形状为 (N, 3)，代表 RGB 颜色。
    """
    z_coords = points_np[:, 2]
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    
    if z_max == z_min:
        normalized_z = np.full_like(z_coords, 0.5)
    else:
        # 归一化 Z 坐标到 [0, 1]
        normalized_z = (z_coords - z_min) / (z_max - z_min)
    
    # 使用指定的颜色图谱进行映射
    mapper = cm.get_cmap(colormap)
    colors = mapper(normalized_z)[:, :3] # 取 RGB 通道
    return colors

class PointCloudVisualizer:
    def __init__(self, window_name="Point Cloud Animation"):
        """
        初始化非阻塞式可视化器。
        """
        self.visualizer = o3d.visualization.VisualizerWithKeyCallback()
        self.visualizer.create_window(window_name=window_name)
        
        # 创建一个空的 PointCloud 对象和一个坐标系
        self.pcd_o3d = o3d.geometry.PointCloud()
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        
        # 将几何体添加到场景中（只需要添加一次）
        self.visualizer.add_geometry(self.pcd_o3d)
        self.visualizer.add_geometry(self.coord_frame)
        
        # 优化渲染设置
        opt = self.visualizer.get_render_option()
        opt.background_color = np.asarray([1.0, 1.0, 1.0]) # 设置纯白背景
        opt.point_size = 3.5                               # 调整点的大小
        opt.light_on = False                               # 关闭光照效果，实现简约风格
        
        self.is_initialized = True
        self.keep_running = True
        self.visualizer.register_key_callback(27, self._exit_callback)
        print("Visualizer initialized. Starting animation loop...")

    def _exit_callback(self, vis):
        print("Exit signal received. Shutting down.")
        self.keep_running = False

    def update_point_cloud(self, pcd_tensor):
        """
        在循环中更新点云数据并渲染新的一帧。
        
        :param pcd_tensor: PyTorch 张量，形状为 (1, N, 3) 或 (N, 3)。
        """
        if not self.is_initialized:
            print("Error: Visualizer not initialized.")
            return

        if not self.keep_running:
            self.close()
            return False

        # 1. 将 PyTorch 张量转换为 NumPy 数组
        points_np = pcd_tensor.squeeze().cpu().numpy()

        # 2. 更新 PointCloud 对象的点
        self.pcd_o3d.points = o3d.utility.Vector3dVector(points_np.astype(np.float64))
        
        # 3. (可选) 如果点云颜色也逐帧变化，可以在这里更新
        # self.pcd_o3d.colors = o3d.utility.Vector3dVector(colors_np)
        colors_np = colorize_pcd_by_height(points_np, colormap='jet') # 使用 'jet' 图谱
        self.pcd_o3d.colors = o3d.utility.Vector3dVector(colors_np.astype(np.float64))
        
        # 4. 通知可视化器几何体已更新
        self.visualizer.update_geometry(self.pcd_o3d)
        
        # 5. 处理窗口事件并渲染场景
        self.visualizer.poll_events()
        self.visualizer.update_renderer()

        return self.keep_running
    
    def crop_pcd_with_camera_view(self, pcd: torch.Tensor, camera_T: np.ndarray, camera_intrinsic_matrix: np.ndarray, 
                                  img_size: tuple, depth_min: float=0.05, depth_max: float=2.0):
        """
        根据腕部相机视角裁剪点云，实现腕部相机点云合成
        :param pcd: PyTorch 张量，世界坐标系下的完整点云 (N, 3)。
        :param camera_T: 世界坐标系到相机坐标系的变换 T_world_to_camera (4x4 NumPy 数组)。
        :param camera_intrinsic_matrix: 相机内参矩阵 K (3x3 NumPy 数组)。
        :param img_size: 相机图像尺寸 (W, H)。
        :param depth_min: 相机可检测的最小深度 (米)。
        :param depth_max: 相机可检测的最大深度 (米)。
        :return: PyTorch 张量，裁剪后的点云 (M, 3)。
        """
        pcd_np = pcd.squeeze().cpu().numpy()
        img_width, img_height = img_size[0], img_size[1]
        points_homo = np.hstack((pcd_np, np.ones((pcd_np.shape[0], 1)))) # 添加 W 
        points_camera_coord = (camera_T @ points_homo.T).T[:, :3] # 将点从世界坐标系转换到相机坐标系
        # 深度裁剪（Z 轴裁剪）。注意方向，一般相机Z轴指向前方
        depth_mask = (points_camera_coord[:, 2] > depth_min) & \
                     (points_camera_coord[:, 2] < depth_max)
        points_camera_coord_depth_filtered = points_camera_coord[depth_mask]
        points_world_coord_depth_filtered = pcd_np[depth_mask] # 
        if points_camera_coord_depth_filtered.shape[0] == 0:
            print("Warning: No points remain after depth filtering.")
            return torch.empty((0, 3), dtype=pcd.dtype)
        
        # 视锥体裁剪 (Frustum Culling)
        # 将相机坐标系中的点投影到图像平面
        # 齐次坐标投影
        proj_points_homo = (camera_intrinsic_matrix @ points_camera_coord_depth_filtered.T).T
        
        # 归一化齐次坐标，得到像素坐标 (u, v)
        # 避免除以零
        z_coords_proj = proj_points_homo[:, 2]
        # 使用一个小的 epsilon 防止除以零，并处理 z_coords_proj 为负的情况 (通常不应该发生，除非点在相机后面)
        epsilon = 1e-6 
        valid_z_mask = z_coords_proj > epsilon
        
        # 对于 z <= epsilon 的点，它们在相机后面或在近裁剪面附近，不进行投影
        proj_points_homo = proj_points_homo[valid_z_mask]
        points_world_coord_frustum_filtered = points_world_coord_depth_filtered[valid_z_mask]

        if proj_points_homo.shape[0] == 0:
            print("Warning: No points remain after valid Z projection filtering.")
            return torch.empty((0, 3), dtype=pcd.dtype)

        pixel_u = proj_points_homo[:, 0] / proj_points_homo[:, 2]
        pixel_v = proj_points_homo[:, 1] / proj_points_homo[:, 2]
        
        # 检查像素坐标是否在图像范围内
        frustum_mask = (pixel_u >= 0) & (pixel_u < img_width) & \
                       (pixel_v >= 0) & (pixel_v < img_height)
        
        cropped_points_world_coord = points_world_coord_frustum_filtered[frustum_mask]

        # 将结果转换回 PyTorch 张量
        return torch.tensor(cropped_points_world_coord, dtype=pcd.dtype)



    def close(self):
        """
        关闭可视化窗口。
        """
        self.visualizer.destroy_window()
        self.is_initialized = False
        print("Visualizer closed.")