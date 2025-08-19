import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation

class PointNet2EncoderGlobal(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(PointNet2EncoderGlobal, self).__init__()
        # 初始化是否使用法向量的标志
        if normal_channel:
            additional_channel = 3  # 如果使用法向量，则增加3个通道
        else:
            additional_channel = 0  # 不使用法向量，通道数增加0
        self.normal_channel = normal_channel
        # 第一层集合抽象模块，使用多尺度分组
        # 输入点数512，使用3种不同尺度的半径和采样点数
        # 输入通道数为3(坐标) + additional_channel(法向量)
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # 第二层集合抽象模块，进一步抽象特征
        # 输入点数128，使用2种尺度的半径和采样点数
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        # 第三层集合抽象模块，全局特征聚合
        # 使用group_all=True将所有点作为一个组处理
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)

    def forward(self, xyz):
        if self.normal_channel:
            xyz = xyz[:, :, :3]  # 只保留坐标信息
        # 第一层集合抽象模块
        # l1_xyz：这是经过第一层集合抽象模块（sa1）后，从原始点云中采样出的512个点的三维坐标。它代表了点云的一个更稀疏的骨架结构。
        # l1_points：这是与 l1_xyz 中的每个点相对应的特征向量。由于 sa1 采用了多尺度分组（MSG），l1_points 是由三个不同尺度（半径0.1, 0.2, 0.4）下学习到的特征拼接而成的。因此，它包含了丰富的局部几何信息。
        l1_xyz, l1_points = self.sa1(xyz)
        # 第二层集合抽象模块
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # 第三层集合抽象模块
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        return l3_points, l3_xyz, l2_points, l2_xyz, l1_points, l1_xyz

class PointNet2EncoderLocal(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(PointNet2EncoderLocal, self).__init__()
        # 初始化是否使用法向量的标志
        if normal_channel:
            additional_channel = 3  # 如果使用法向量，则增加3个通道
        else:
            additional_channel = 0  # 不使用法向量，通道数增加0
        self.normal_channel = normal_channel
        # 第一层集合抽象模块，使用多尺度分组
        # 输入点数512，使用3种不同尺度的半径和采样点数
        # 输入通道数为3(坐标) + additional_channel(法向量)
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # 第二层集合抽象模块，进一步抽象特征
        # 输入点数128，使用2种尺度的半径和采样点数
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        # 第三层集合抽象模块，全局特征聚合
        # 使用group_all=True将所有点作为一个组处理
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        # 特征传播模块，从高层到底层传播特征
        # fp3: 从第三层到第二层的特征传播
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        # fp2: 从第二层到第一层的特征传播
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        # fp1: 从第一层到原始输入的特征传播
        self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])

    def forward(self, xyz):
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz
        # 第一层集合抽象模块
        l1_xyz, l1_points = self.sa1(xyz)
        # 第二层集合抽象模块
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # 第三层集合抽象模块
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # 特征传播（解码过程）
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)
        # 返回每个点的特征
        return l0_points  # [B, 128, N]