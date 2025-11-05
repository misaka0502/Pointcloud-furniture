import torch
from utils.coordinate_transform import robot_pose_to_april_pose
from utils.fb_control_utils import rotation_6d_to_matrix_simple, rotation_matrix_to_quaternion_simple, batched_pose2mat

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
    
    # finger base 在 EE (gripper site) 坐标系中的位置
    # 根据 mjx_panda.xml:
    #   - gripper site 在 hand 上方: pos="0 0 0.1"
    #   - finger_base 在 hand 上方: pos="0 0 0.0584"
    #   - 所以 finger_base 相对于 gripper site: 0.0584 - 0.1 = -0.0416
    finger_base_z = -0.0416
    
    # ✅ 构建右 finger（Y < 0）
    # 原始 finger 内侧朝向 -Y，需要绕 Z 轴旋转 180° 让内侧朝向 +Y（朝向中心）
    # 旋转 180°: (x, y, z) → (-x, -y, z)
    right_finger_0 = finger_0_original.clone()
    right_finger_0[:, 0] = -right_finger_0[:, 0]  # X 取反
    right_finger_0[:, 1] = -right_finger_0[:, 1]  # Y 取反
    
    # 旋转后，finger_1 的内侧从 finger_1_inner_y 变成 -finger_1_inner_y
    right_finger_target_inner_y = -gripper_width / 2
    right_y_shift = right_finger_target_inner_y - (-finger_1_inner_y)
    right_finger_0[:, 1] += right_y_shift
    right_finger_0[:, 2] += finger_base_z
    
    right_finger_1 = finger_1_original.clone()
    right_finger_1[:, 0] = -right_finger_1[:, 0]  # X 取反
    right_finger_1[:, 1] = -right_finger_1[:, 1]  # Y 取反
    right_finger_1[:, 1] += right_y_shift
    right_finger_1[:, 2] += finger_base_z
    
    # ✅ 构建左 finger（Y > 0）
    # 需要内侧朝向 -Y（朝向中心）
    # 方法：旋转 180° + Y 镜像 = X 取反
    # 证明：旋转 (x,y)→(-x,-y), 然后 Y 镜像 (-x,-y)→(-x,y)
    left_finger_0 = finger_0_original.clone()
    left_finger_0[:, 0] = -left_finger_0[:, 0]  # X 取反
    
    # X 取反后，finger_1 的内侧还是 finger_1_inner_y，且仍朝向 -Y
    left_finger_target_inner_y = gripper_width / 2
    left_y_shift = left_finger_target_inner_y - finger_1_inner_y
    left_finger_0[:, 1] += left_y_shift
    left_finger_0[:, 2] += finger_base_z
    
    left_finger_1 = finger_1_original.clone()
    left_finger_1[:, 0] = -left_finger_1[:, 0]  # X 取反
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


def synthesize_gripper_pcd(
    ee_pos_robot,
    ee_rot_6d_robot,
    gripper_width,
    gripper_pcds,
    device,
    batch_size=1,
):
    """
    从 EE 状态合成完整的夹爪点云（世界坐标系）
    
    这是一个高级接口，封装了完整的夹爪点云生成流程：
    1. 坐标系转换（Robot Frame → April Tag Frame）
    2. 在 EE 局部坐标系构建夹爪点云
    3. 变换到世界坐标系
    
    Args:
        ee_pos_robot: [N, 3] or [3], EE 位置（Robot Frame）
        ee_rot_6d_robot: [N, 6] or [6], EE 姿态 6D rotation（Robot Frame）
        gripper_width: [N] or float, 夹爪宽度
        gripper_pcds: dict, 包含 'finger_0' 和 'finger_1' 的齐次坐标 [N_points, 4]
        device: torch device
        batch_size: int, 批次大小（默认 1）
    
    Returns:
        gripper_pcd_world: [N, N_total, 3], 世界坐标系中的夹爪点云
        
    Examples:
        >>> # 单帧处理
        >>> gripper_pcd = synthesize_gripper_pcd(
        ...     ee_pos_robot=torch.tensor([0.5, 0.2, 0.3]),
        ...     ee_rot_6d_robot=torch.tensor([1, 0, 0, 0, 1, 0]),
        ...     gripper_width=0.065,
        ...     gripper_pcds=gripper_pcds,
        ...     device='cuda'
        ... )
        >>> print(gripper_pcd.shape)  # [1, 636, 3]
        
        >>> # 批量处理
        >>> gripper_pcds_batch = synthesize_gripper_pcd(
        ...     ee_pos_robot=ee_positions,  # [N, 3]
        ...     ee_rot_6d_robot=ee_rotations,  # [N, 6]
        ...     gripper_width=gripper_widths,  # [N]
        ...     gripper_pcds=gripper_pcds,
        ...     device='cuda',
        ...     batch_size=N
        ... )
    """
    # 确保输入是 tensor
    if not isinstance(ee_pos_robot, torch.Tensor):
        ee_pos_robot = torch.tensor(ee_pos_robot, device=device, dtype=torch.float32)
    if not isinstance(ee_rot_6d_robot, torch.Tensor):
        ee_rot_6d_robot = torch.tensor(ee_rot_6d_robot, device=device, dtype=torch.float32)
    if not isinstance(gripper_width, torch.Tensor):
        gripper_width = torch.tensor(
            [gripper_width] if isinstance(gripper_width, (int, float)) else gripper_width,
            device=device,
            dtype=torch.float32
        )
    
    # 确保是批次形式
    if ee_pos_robot.dim() == 1:
        ee_pos_robot = ee_pos_robot.unsqueeze(0)  # [1, 3]
    if ee_rot_6d_robot.dim() == 1:
        ee_rot_6d_robot = ee_rot_6d_robot.unsqueeze(0)  # [1, 6]
    if gripper_width.dim() == 0:
        gripper_width = gripper_width.unsqueeze(0)  # [1]
    
    N = ee_pos_robot.shape[0]
    
    # 存储所有批次的结果
    gripper_pcds_world_list = []
    
    for i in range(N):
        # 1. 转换旋转表示: 6D → 旋转矩阵 → 四元数
        ee_rot_mat_robot = rotation_6d_to_matrix_simple(ee_rot_6d_robot[i])
        ee_quat_robot = rotation_matrix_to_quaternion_simple(ee_rot_mat_robot)
        
        # 2. 坐标系转换: Robot Frame → April Tag Frame
        ee_pos_april, ee_quat_april = robot_pose_to_april_pose(
            ee_pos_robot[i], ee_quat_robot, device
        )
        
        # 3. 在 EE 局部坐标系中构建夹爪点云
        gripper_pcd_local = build_gripper_pcd_in_hand_frame(
            gripper_pcds, gripper_width[i].item(), device
        )
        
        # 4. 构建位姿变换矩阵
        gripper_pose_mat = batched_pose2mat(
            ee_pos_april.unsqueeze(0),
            ee_quat_april.unsqueeze(0),
            device
        )  # [1, 4, 4]
        
        # 5. 变换到世界坐标系
        gripper_pcd_world = torch.matmul(
            gripper_pcd_local.unsqueeze(0),
            gripper_pose_mat.transpose(1, 2)
        )[:, :, :3]  # [1, N_total, 3]
        
        gripper_pcds_world_list.append(gripper_pcd_world)
    
    # 合并所有批次
    gripper_pcd_world = torch.cat(gripper_pcds_world_list, dim=0)  # [N, N_total, 3]
    
    return gripper_pcd_world