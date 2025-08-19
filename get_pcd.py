import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
import os

def rgbd_to_point_cloud(rgb_image, depth_image, K, cam_pos, cam_target, up_axis=np.array([0., 0., 1.])):
    """
    将RGB-D图像、相机内外惨转换为世界坐标系下的彩色点云
    Args:
        rgb_image (np.array): HxWx3 的RGB图像，像素值范围[0, 255]， uint8类型。
        depth_image (np.array): HxW 的深度图，每个像素值是其在相机坐标系下的z轴距离(米)
        K (np.array): 3x3 的相机内参矩阵。
        cam_pos (np.array): 1x3 的相机在世界坐标系下的位置。
        cam_target (np.array): 1x3 的相机在世界坐标系下朝向的目标点。
        up_axis (np.array): 1x3 的世界坐标系的上方向向量，通常是[0, 0, 1](Z-up)或[0, 1, 0](Y-up)

    Returns:
        o3d.geometry.PointCloud: Open3D格式的彩色点云对象。
    """
    # 获取图像尺寸和相机内参
    height, width = depth_image.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 创建像素坐标网格 (u, v)
    # np.meshgrid 生成一个坐标矩阵
    u = np.arange(width)
    v = np.arange(height)
    u_grid, v_grid = np.meshgrid(u, v)

    # 反投影：将像素坐标（u, v）和深度d转换为相机坐标系下的(Xc, Yc, Zc)
    # Zc就是深度值
    Zc = depth_image.astype(float)

    # 过滤掉无效的深度点（通常为0或者非常大的值）
    valid_mask = Zc > 0

    # 应用公式计算 Xc 和 Yc
    Xc = (u_grid - cx) * Zc / fx
    Yc = -(v_grid - cy) * Zc / fy # 由于2D图像和3D相机模型中对坐标轴方向的定义有差异，这里取反能够得到正确的点云结果

    # 将点云和颜色数据展平，并值保留有效点
    # (H, W) → （N,）其中N是有效点的数量
    points_camera_frame = np.stack((Xc, Yc, Zc), axis=-1)[valid_mask]
    colors = rgb_image[valid_mask] / 255.0  # 将颜色值归一化到[0, 1]

    # 构建相机到世界坐标系的变换矩阵（外参）
    # 这是一个标准的"look-at"变换矩阵
    
    # 确定相机的Z轴（朝向的反方向）
    forward = np.array(cam_target) - np.array(cam_pos)
    if np.linalg.norm(forward) == 0:
        raise ValueError("相机位置和相机朝向不能是一样的")
    forward = forward / np.linalg.norm(forward)

    # 确定相机的X轴（右方向）
    right = np.cross(forward, up_axis)
    if np.linalg.norm(right) == 0:
        # 如果相机直上或直下看，up_axis和forward平行
        # 此时可以用一个备用的up向量，例如世界X轴
        if np.allclose(forward, up_axis):
            right = np.cross(forward, np.array([1.0, 0., 0.]))
        else: # 直下看
            right = np.cross(forward, np.array([-1., 0., 0.]))

    right = right / np.linalg.norm(right)

    # 确定相机的Y轴（上方向）
    up = np.cross(right, forward)

    # 创建4x4变换矩阵
    cam_to_world_mat = np.eye(4)
    cam_to_world_mat[:3, 0] = right
    cam_to_world_mat[:3, 1] = up
    cam_to_world_mat[:3, 2] = forward
    cam_to_world_mat[:3, 3] = cam_pos

    # 将点云从相机坐标系变换到世界坐标系
    # 将相机坐标系点转换为齐次坐标 (x, y, z, 1)
    num_points = points_camera_frame.shape[0]
    points_homogeneous = np.hstack((points_camera_frame, np.ones((num_points, 1))))

    # 应用变换矩阵
    point_world_frame_homogeneous = (cam_to_world_mat @ points_homogeneous.T).T

    # 转换回非齐次坐标
    points_world_frame = point_world_frame_homogeneous[:, :3]
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world_frame)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


if __name__ == '__main__':
    # 创建虚拟输入数据
    H, W = 720, 1080

    # 读取rgb和深度图
    rgb_path = 'images/color/raw/0000000000000000000.png'
    depth_path = 'images/depth/raw/0000000000000000000.png'

    depth_scale_factor = 1000.0

    # 相机内参矩阵
    K_mat = np.array([
        [9.242773437500000000e+02, 0.000000000000000000e+00, 6.400000000000000000e+02],
        [0.000000000000000000e+00, 9.242773818969726562e+02, 3.600000000000000000e+02],
        [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
    ])

    # 相机外参
    cam_pos = np.array([0.90, -0.00, 0.65])
    cam_target = np.array([-1, -0.00, 0.3])

    # 读取图像
    print("\n从硬盘中读取图像")
    try:
        rgb_pil = Image.open(rgb_path)
        depth_pil = Image.open(depth_path)
    except FileNotFoundError as e:
        print(f"错误，无法找到图像，请确保'{rgb_path}'和'{depth_path}'存在")
    
    # 将pillow图像对象转换为numpy数组
    rgb_np = np.array(rgb_pil)
    depth_np = np.array(depth_pil)

    # 将深度图的值从整数（mm，uint16）转换为浮点数（m，float）
    depth_np = depth_np.astype(np.float32) / depth_scale_factor

    print(f"RGB图像尺寸: {rgb_np.shape}")
    print(f"深度图尺寸: {depth_np.shape}")

    # 显示加载的图像
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(rgb_np)
    axs[0].set_title('rgb')
    axs[1].imshow(depth_np)
    axs[1].set_title('depth')
    plt.show()

    # rgb 转换 point cloud
    print('将RGB-D图像转换为点云...')
    point_cloud = rgbd_to_point_cloud(
        rgb_np, depth_np, K_mat, cam_pos, cam_target
    )
    print(f"成功生成点云， 包含{len(point_cloud.points)}个点")

    # 可视化和保存
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    print("正在显示点云...按‘q’关闭窗口")
    o3d.visualization.draw_geometries([point_cloud, world_frame])