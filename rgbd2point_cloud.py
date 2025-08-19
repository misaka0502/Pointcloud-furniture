import open3d as o3d
import numpy as np
import imageio.v2 as imageio

# # 在脚本的最开始，导入 open3d 之后，马上初始化无头渲染
# # 确保在创建任何 o3d 对象之前调用
# try:
#     o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
#     o3d.visualization.webrtc_server.enable_webrtc()
#     # 如果上面这行导致错误，可能你的Open3D版本较旧或编译时未包含WebRTC
#     # 对于纯粹的离屏渲染，你可以只保留下面的初始化
#     o3d.visualization.RenderOption.line_width = 0.0 # 只是一个示例调用来确保初始化
# except Exception as e:
#     print(f"Could not initialize WebRTC for headless rendering: {e}")
#     # 在较新版本的Open3D中，可以直接设置渲染器
#     # o3d.visualization.rendering.RenderToImage.render_to_image(...) 可能是另一种方式

# ----------------------------------------------------------------------------
# 辅助函数 (与之前相同)
# ----------------------------------------------------------------------------

def build_view_matrix_from_pos_target(cam_pos, cam_target, up_vector=np.array([0, 0, 1])):
    """从相机位置和朝向目标构建视图矩阵 (LookAt)。"""
    if not isinstance(cam_pos, np.ndarray):
        cam_pos = np.array([cam_pos.x, cam_pos.y, cam_pos.z])
    if not isinstance(cam_target, np.ndarray):
        cam_target = np.array([cam_target.x, cam_target.y, cam_target.z])
    z_axis = cam_pos - cam_target
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up_vector, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.array([x_axis, y_axis, z_axis])
    translation = -np.dot(rotation, cam_pos)
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = rotation
    view_matrix[:3, 3] = translation
    return view_matrix

# ----------------------------------------------------------------------------
# 主函数：整合了点云创建和离线渲染
# ----------------------------------------------------------------------------

def render_point_cloud_from_files(
    rgb_path: str,
    depth_path: str,
    output_image_path: str, # 新增：输出图像的文件路径
    k_matrix: np.ndarray,
    cam_pos: np.array,
    cam_target: np.array,
    width: int,
    height: int,
    near_plane: float = 0.001,
    far_plane: float = 2.0,
    depth_scale: float = 1000.0
):
    """
    加载图像，重建相机参数，并在后台渲染点云视图，最后保存为图像文件。
    """
    # === 第 1 部分：创建点云 (与之前脚本的核心逻辑相同) ===
    
    # 1. 加载图像
    print(f"Loading images: {rgb_path}, {depth_path}")
    rgb_image = imageio.imread(rgb_path)
    depth_image = imageio.imread(depth_path)
    if rgb_image.shape[2] == 4:
        rgb_image = rgb_image[..., :3]

    # 2. 从 K 矩阵创建 Open3D 相机内参
    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    # 3. 创建 RGBD 图像
    o3d_color = o3d.geometry.Image(rgb_image)
    o3d_depth = o3d.geometry.Image(depth_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth, depth_scale=depth_scale,
        depth_trunc=far_plane, convert_rgb_to_intensity=False
    )
    
    # 4. 创建点云 (仍在相机坐标系)
    pcd_camera_frame = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsics
    )
    
    # 5. 重建视图矩阵，并计算相机外参 (Extrinsics)
    view_matrix = build_view_matrix_from_pos_target(cam_pos, cam_target)
    camera_extrinsics = np.linalg.inv(view_matrix)
    
    # 6. 将点云变换到世界坐标系
    pcd_world_frame = pcd_camera_frame.transform(camera_extrinsics)

    # === 第 2 部分：离线渲染并保存图像 ===

    print("Starting offline rendering...")
    
    # 1. 创建离屏渲染器
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    # (可选) 设置渲染风格
    renderer.scene.set_background(np.array([0.15, 0.15, 0.15, 1.0])) # RGBA
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit" # 对于点云，Unlit通常效果更好
    mat.point_size = 3.0

    # 2. 将点云添加到场景中
    renderer.scene.add_geometry("point_cloud", pcd_world_frame, mat)

    # 3. 设置相机
    # OffscreenRenderer 直接使用内外参矩阵
    # 我们已经有了 K 矩阵和外参矩阵
    renderer.scene.camera.set_projection(
        intrinsics.intrinsic_matrix, near_plane, far_plane, width, height
    )
    renderer.scene.camera.look_at(
        camera_extrinsics[0:3, 3], # 相机位置 (从外参矩阵提取)
        camera_extrinsics[0:3, 3] + camera_extrinsics[0:3, 2] * -1.0, # 目标点 (位置 + 向前的方向)
        -camera_extrinsics[0:3, 1]  # 上方向 由于Isaacgym中定义的相机的上方向是Z-up，而一般的图形库（如Open3D）默认的上方向是Y-up，所以这里需要反转一下
    )

    # 4. 渲染图像
    img_o3d = renderer.render_to_image()

    # 5. 保存图像
    # 将 Open3D 图像转换为 NumPy 数组并使用 imageio 保存
    img_np = np.asarray(img_o3d)
    # img_np_flipped = np.flipud(img_np)
    imageio.imwrite(output_image_path+"color_point_cloud.png", img_np)

    print(f"Point cloud view successfully rendered and saved to: {output_image_path}")

def create_and_render_xyz_point_cloud(
    depth_path: str,
    output_image_path: str,
    k_matrix: np.ndarray,
    cam_pos: np.array,
    cam_target: np.array,
    width: int,
    height: int,
    depth_scale: float = 1000.0,
    near_plane: float = 0.001,
    far_plane: float = 2.0
):
    """
    仅从深度图加载数据，创建XYZ点云，并在后台渲染视图，最后保存为图像文件。
    """
    # === 第 1 部分：创建点云 ===
    
    # 1. 加载深度图像
    print(f"Loading depth image: {depth_path}")
    depth_image = imageio.imread(depth_path)

    # 2. 从 K 矩阵创建 Open3D 相机内参
    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    # 3. 将NumPy深度图转换为Open3D图像对象
    o3d_depth = o3d.geometry.Image(depth_image)
    
    # 4. 【核心变化】直接从深度图创建点云 (在相机坐标系中)
    # 这个函数不需要RGB信息，只根据深度值和内参计算每个像素的XYZ坐标
    pcd_camera_frame = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_depth,
        intrinsics,
        depth_scale=depth_scale,
        depth_trunc=far_plane
    )
    
    # 5. 重建视图矩阵和相机外参
    view_matrix = build_view_matrix_from_pos_target(cam_pos, cam_target)
    camera_extrinsics = np.linalg.inv(view_matrix)
    
    # 6. 将点云变换到世界坐标系
    pcd_world_frame = pcd_camera_frame.transform(camera_extrinsics)

    # === 第 2 部分：离线渲染 (逻辑不变) ===
    print("Starting offline rendering of XYZ point cloud...")
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 3.0
    # 由于没有颜色信息，点云会以默认的灰色或单色显示
    renderer.scene.set_background(np.array([0.15, 0.15, 0.15, 1.0]))
    renderer.scene.add_geometry("point_cloud", pcd_world_frame, mat)

    renderer.scene.camera.set_projection(
        intrinsics.intrinsic_matrix, near_plane, far_plane, width, height
    )
    renderer.scene.camera.look_at(
        camera_extrinsics[0:3, 3], # 相机位置 (从外参矩阵提取)
        camera_extrinsics[0:3, 3] + camera_extrinsics[0:3, 2] * -1.0, # 目标点 (位置 + 向前的方向)
        -camera_extrinsics[0:3, 1]
    )

    img_o3d = renderer.render_to_image()
    img_np = np.asarray(img_o3d)
    imageio.imwrite(output_image_path+"depth_point_cloud.png", img_np)

    print(f"XYZ point cloud view successfully saved to: {output_image_path}")

if __name__ == '__main__':
    # --- 你的已知参数 ---
    # K 矩阵 (相机内参)
    K = np.array([
        [9.242773437500000000e+02, 0.000000000000000000e+00, 6.400000000000000000e+02],
        [0.000000000000000000e+00, 9.242773818969726562e+02, 3.600000000000000000e+02],
        [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
    ])

    # 相机位置和朝向
    cam_pos = np.array([0.90, -0.00, 0.65])
    cam_target = np.array([-1, -0.00, 0.3])
    
    # 图像尺寸
    IMG_WIDTH = 1280
    IMG_HEIGHT = 720

    # --- 文件路径 ---
    rgb_file = "images/color/raw/0000000000000000000.png"
    depth_file = "images/depth/raw/0000000000000000000.png"
    
    # 新增：定义输出图像的路径
    output_file = "images/"

    # --- 调用渲染函数 ---
    # 渲染彩色点云
    render_point_cloud_from_files(
        rgb_path=rgb_file,
        depth_path=depth_file,
        output_image_path=output_file, # 传入输出路径
        k_matrix=K,
        cam_pos=cam_pos,
        cam_target=cam_target,
        width=IMG_WIDTH,
        height=IMG_HEIGHT,
        depth_scale=1000.0
    )
    # 渲染深度点云
    # create_and_render_xyz_point_cloud(
    #     depth_path=depth_file,
    #     output_image_path=output_file, # 传入输出路径
    #     k_matrix=K,
    #     cam_pos=cam_pos,
    #     cam_target=cam_target,
    #     width=IMG_WIDTH,
    #     height=IMG_HEIGHT,
    #     depth_scale=1000.0,
    #     near_plane=0.001,
    #     far_plane=2.0
    # )