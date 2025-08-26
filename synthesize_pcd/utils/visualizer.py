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
        opt.background_color = np.asarray([1.0, 1.0, 1.0]) # 1. 设置纯白背景
        opt.point_size = 3.5                               # 2. 调整点的大小
        opt.light_on = False                               # 3. 关闭光照效果，实现简约风格
        
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
    def close(self):
        """
        关闭可视化窗口。
        """
        self.visualizer.destroy_window()
        self.is_initialized = False
        print("Visualizer closed.")