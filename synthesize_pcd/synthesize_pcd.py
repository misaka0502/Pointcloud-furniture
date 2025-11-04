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
from coordinate_transform import robot_pose_to_april_pose

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

def rotation_6d_to_matrix_simple(d6):
    """
    将 6D 旋转表示转换为旋转矩阵 (Gram-Schmidt 正交化)
    
    参考：furniture_bench/controllers/control_utils.py::rotation_6d_to_matrix
    
    Args:
        d6: [6] 6D 旋转向量
    Returns:
        rot_mat: [3, 3] 旋转矩阵（行向量形式）
    """
    a1, a2 = d6[:3], d6[3:]
    
    # 正交化
    b1 = torch.nn.functional.normalize(a1, dim=0)
    b2 = a2 - (b1 * a2).sum() * b1
    b2 = torch.nn.functional.normalize(b2, dim=0)
    b3 = torch.cross(b1, b2)
    
    # 关键修复：按行堆叠，而不是按列！
    # dim=0 表示在第0维（行）堆叠，得到 [3, 3] 矩阵
    # 其中第1行是 b1，第2行是 b2，第3行是 b3
    return torch.stack([b1, b2, b3], dim=0)

def rotation_matrix_to_quaternion_simple(rot_mat):
    """
    将旋转矩阵转换为四元数
    Args:
        rot_mat: [3, 3] 旋转矩阵
    Returns:
        quat: [4] 四元数 (w, x, y, z)
    """
    trace = rot_mat[0, 0] + rot_mat[1, 1] + rot_mat[2, 2]
    
    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot_mat[2, 1] - rot_mat[1, 2]) * s
        y = (rot_mat[0, 2] - rot_mat[2, 0]) * s
        z = (rot_mat[1, 0] - rot_mat[0, 1]) * s
    elif rot_mat[0, 0] > rot_mat[1, 1] and rot_mat[0, 0] > rot_mat[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2])
        w = (rot_mat[2, 1] - rot_mat[1, 2]) / s
        x = 0.25 * s
        y = (rot_mat[0, 1] + rot_mat[1, 0]) / s
        z = (rot_mat[0, 2] + rot_mat[2, 0]) / s
    elif rot_mat[1, 1] > rot_mat[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2])
        w = (rot_mat[0, 2] - rot_mat[2, 0]) / s
        x = (rot_mat[0, 1] + rot_mat[1, 0]) / s
        y = 0.25 * s
        z = (rot_mat[1, 2] + rot_mat[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1])
        w = (rot_mat[1, 0] - rot_mat[0, 1]) / s
        x = (rot_mat[0, 2] + rot_mat[2, 0]) / s
        y = (rot_mat[1, 2] + rot_mat[2, 1]) / s
        z = 0.25 * s
    
    return torch.tensor([w, x, y, z], device=rot_mat.device, dtype=rot_mat.dtype)

def quat_apply_simple(quat, vec):
    """
    使用四元数旋转向量
    quat: [4] (w, x, y, z)
    vec: [3] (x, y, z)
    返回: [3] 旋转后的向量
    """
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    x, y, z = vec[0], vec[1], vec[2]
    
    # 四元数旋转公式
    ix = qw * x + qy * z - qz * y
    iy = qw * y + qz * x - qx * z
    iz = qw * z + qx * y - qy * x
    iw = -qx * x - qy * y - qz * z
    
    rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
    ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
    rz = iz * qw + iw * -qz + ix * -qy - iy * -qx
    
    return torch.tensor([rx, ry, rz], device=quat.device, dtype=quat.dtype)

def quat_mul_simple(q1, q2):
    """
    四元数乘法 (w, x, y, z)
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    
    return torch.tensor([w, x, y, z], device=q1.device, dtype=q1.dtype)

def build_gripper_pcd_in_hand_frame(gripper_pcds, gripper_width, device):
    """
    在 hand 局部坐标系中构建完整的夹爪点云
    
    关键设计：
    - finger_0 是外壳/主体
    - finger_1 是内侧夹持面（接触面）
    - gripper_width 是两个 finger_1 **内侧**之间的距离
    - finger_0 和 finger_1 保持原始相对位置关系
    
    Args:
        gripper_pcds: dict, 包含 'finger_0' 和 'finger_1' 的齐次坐标 [N, 4]
        gripper_width: float, 两个 finger_1 内侧之间的距离
        device: torch device
    
    Returns:
        gripper_pcd_local: [N_total, 4] 在 hand 局部坐标系中的完整夹爪点云
    """
    finger_0_original = gripper_pcds['finger_0'].clone()
    finger_1_original = gripper_pcds['finger_1'].clone()
    
    # ✅ 关键：找到 finger_1 的内侧位置（Y 最小值）
    finger_1_inner_y = finger_1_original[:, 1].min().item()  # 内侧位置
    
    # finger base 在 hand 坐标系中的位置
    finger_base_z = 0.066  # hand 顶部
    
    # ✅ 构建右 finger：finger_1 的内侧应该在 -gripper_width/2
    # 计算需要的 Y 偏移量
    right_finger_target_inner_y = -gripper_width / 2
    right_y_shift = right_finger_target_inner_y - finger_1_inner_y
    
    right_finger_0 = finger_0_original.clone()
    right_finger_0[:, 1] += right_y_shift
    right_finger_0[:, 2] += finger_base_z
    
    right_finger_1 = finger_1_original.clone()
    right_finger_1[:, 1] += right_y_shift
    right_finger_1[:, 2] += finger_base_z
    
    # ✅ 构建左 finger：Y 轴镜像，然后 finger_1 的内侧应该在 +gripper_width/2
    left_finger_0 = finger_0_original.clone()
    left_finger_0[:, 1] = -left_finger_0[:, 1]  # Y 镜像
    
    # ⚠️ 关键：镜像后，原来的 Y_min 变成新的 Y_max，原来的 Y_max 变成新的 Y_min
    # 镜像后 finger_1 的内侧（新的 Y_min）= -(原来的 Y_max)
    finger_1_max_y = finger_1_original[:, 1].max().item()
    left_finger_mirrored_inner_y = -finger_1_max_y
    
    left_finger_target_inner_y = gripper_width / 2
    left_y_shift = left_finger_target_inner_y - left_finger_mirrored_inner_y
    left_finger_0[:, 1] += left_y_shift
    left_finger_0[:, 2] += finger_base_z
    
    left_finger_1 = finger_1_original.clone()
    left_finger_1[:, 1] = -left_finger_1[:, 1]  # Y 镜像
    left_finger_1[:, 1] += left_y_shift
    left_finger_1[:, 2] += finger_base_z
    
    # 合并所有点云
    gripper_pcd_local = torch.cat([
        left_finger_0,
        left_finger_1,
        right_finger_0,
        right_finger_1
    ], dim=0)  # [N_total, 4]
    
    return gripper_pcd_local

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
        
        # 将 EE 位姿从机器人基座坐标系转换到 AprilTag 坐标系
        # 1. 先将 6D 旋转转换为四元数
        ee_rot_mat = rotation_6d_to_matrix_simple(ee_rot_6d_robot)
        ee_quat_robot = rotation_matrix_to_quaternion_simple(ee_rot_mat)  # [4] (w,x,y,z)
        
        # 2. 坐标系转换：Robot -> AprilTag
        ee_pos_april, ee_quat_april = robot_pose_to_april_pose(
            ee_pos_robot, ee_quat_robot, furniture.device
        )
        
        # 3. 将四元数转回旋转矩阵
        # 注意：C.quaternion_to_matrix 期望 (x,y,z,w) 格式
        ee_quat_april_xyzw = torch.cat([ee_quat_april[1:], ee_quat_april[0:1]])  # (w,x,y,z) -> (x,y,z,w)
        ee_rot_mat_april = C.quaternion_to_matrix(ee_quat_april_xyzw.unsqueeze(0)).squeeze(0)
        
        # ✅ 使用新方法：直接存储 hand 的位姿，不需要单独计算左右 finger
        # ee_pose = (pos[3] + quat[4])  quat 格式为 (w,x,y,z)
        ee_pose = torch.cat([ee_pos_april, ee_quat_april])  # [7]
        part_pose['gripper'] = ee_pose.unsqueeze(0).expand(N_ENVS, -1)  # [N_env, 7]
        if i == 0:
            print_gpu_memory_usage(furniture.device, "begin to synthesize point cloud")
        
        if COMPUTE_FPS:
            # 强制 CPU 等待 GPU 完成上一帧的所有工作
            torch.cuda.synchronize()
            start_time = time.perf_counter()

        # 处理家具点云合成
        furniture.get_pcd_from_offline_data(part_pose)
        
        # ✅ 处理夹爪点云：先在局部坐标系构建，再整体变换
        if 'finger_0' in gripper_pcds and 'finger_1' in gripper_pcds:
            # 1. 在 hand 局部坐标系中构建完整的夹爪点云
            gripper_pcd_local = build_gripper_pcd_in_hand_frame(
                gripper_pcds, gripper_width, furniture.device
            )  # [N_total, 4]
            
            # 2. 用 ee_pose 将整个夹爪变换到世界坐标系
            gripper_pose = part_pose['gripper']  # [N_env, 7] (pos[3] + quat[4], quat 为 w,x,y,z)
            
            # ✅ batched_pose2mat 接受 (w,x,y,z) 格式，不需要转换
            gripper_pose_mat = C.batched_pose2mat(
                gripper_pose[:, :3],      # [N_env, 3] 位置
                gripper_pose[:, 3:7],     # [N_env, 4] 四元数 (w,x,y,z)
                furniture.device
            )  # [N_env, 4, 4]
            
            # 3. 广播 gripper_pcd_local 到所有环境
            n_envs = gripper_pose_mat.shape[0]
            gripper_pcd_local_expanded = gripper_pcd_local.unsqueeze(0).expand(n_envs, -1, -1)  # [N_env, N_total, 4]
            
            # 4. 应用变换
            gripper_pcd_world = torch.matmul(
                gripper_pcd_local_expanded,
                gripper_pose_mat.transpose(1, 2)
            )[:, :, :3]  # [N_env, N_total, 3]
        
        # 分别采样家具和夹爪点云，避免夹爪被过度稀疏化
        furniture_pcd = torch.cat(list(furniture.parts_pcds_world.values()), dim=1)  # [N_env, ~214k, 3]
        # gripper_pcd_world 现在是一个整体的张量，不是字典
        gripper_pcd = gripper_pcd_world  # [N_env, N_total, 3]
        
        # 为夹爪分配固定采样点数
        gripper_sample_num = 512
        furniture_sample_num = 4096 - gripper_sample_num  # 3584 个点
        
        furniture_sampled = sample_points(furniture_pcd, sample_num=furniture_sample_num)
        gripper_sampled = sample_points(gripper_pcd, sample_num=gripper_sample_num)
        
        # 合并：家具采样 + 夹爪采样
        pcds_sampled = torch.cat([furniture_sampled, gripper_sampled], dim=1)  # [N_env, 4096, 3]
        
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
        gripper_vis_sample = 512
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
                    # 未暂停，推进到下一帧
                    break
                time.sleep(0.01)  # 暂停时降低 CPU 使用
            
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