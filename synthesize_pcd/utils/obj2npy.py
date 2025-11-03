"""
OBJ to NPY converter
将 OBJ 文件中的顶点坐标转换为 NPY 格式，保存表面点云数据
"""

import numpy as np
import os
from typing import Optional


def read_obj_vertices(obj_path: str) -> np.ndarray:
    """
    从 OBJ 文件中读取所有顶点坐标
    
    Args:
        obj_path: OBJ 文件路径
        
    Returns:
        顶点坐标数组，shape: (N, 3)
    """
    vertices = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):  # 顶点坐标行
                parts = line.split()
                if len(parts) >= 4:
                    # 提取 x, y, z 坐标
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append([x, y, z])
    
    return np.array(vertices, dtype=np.float64)


def obj_to_npy(obj_path: str, 
               output_path: Optional[str] = None,
               remove_duplicates: bool = True) -> str:
    """
    将 OBJ 文件转换为 NPY 文件
    
    Args:
        obj_path: 输入的 OBJ 文件路径
        output_path: 输出的 NPY 文件路径，如果为 None 则自动生成
        remove_duplicates: 是否移除重复的顶点
        
    Returns:
        输出文件的路径
    """
    # 读取顶点
    vertices = read_obj_vertices(obj_path)
    print(f"从 {os.path.basename(obj_path)} 读取了 {len(vertices)} 个顶点")
    
    # 可选：移除重复顶点
    if remove_duplicates:
        vertices_unique = np.unique(vertices, axis=0)
        print(f"去重后剩余 {len(vertices_unique)} 个唯一顶点")
        vertices = vertices_unique
    
    # 生成输出路径
    if output_path is None:
        base_name = os.path.splitext(obj_path)[0]
        output_path = base_name + '.npy'
    
    # 保存为 NPY 文件
    np.save(output_path, vertices)
    print(f"已保存到: {output_path}")
    print(f"数据形状: {vertices.shape}, 数据类型: {vertices.dtype}")
    
    return output_path


def convert_finger_objs():
    """
    转换 finger_0.obj 和 finger_1.obj 为 NPY 格式
    """
    # 获取脚本所在目录的父目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # 构建 OBJ 文件路径
    mesh_dir = os.path.join(project_dir, 'assets', 'furniture_bench', 
                            'mesh', 'square_table')
    
    obj_files = ['finger_0.obj', 'finger_1.obj']
    
    print("=" * 60)
    print("开始转换 OBJ 文件到 NPY 格式")
    print("=" * 60)
    
    for obj_file in obj_files:
        obj_path = os.path.join(mesh_dir, obj_file)
        
        if not os.path.exists(obj_path):
            print(f"警告: 文件不存在: {obj_path}")
            continue
        
        print(f"\n处理: {obj_file}")
        print("-" * 60)
        
        try:
            output_path = obj_to_npy(obj_path, remove_duplicates=True)
            
            # 验证生成的文件
            data = np.load(output_path)
            print(f"验证: 成功加载 {os.path.basename(output_path)}")
            print(f"点云范围:")
            print(f"  X: [{data[:, 0].min():.6f}, {data[:, 0].max():.6f}]")
            print(f"  Y: [{data[:, 1].min():.6f}, {data[:, 1].max():.6f}]")
            print(f"  Z: [{data[:, 2].min():.6f}, {data[:, 2].max():.6f}]")
            
        except Exception as e:
            print(f"错误: 处理 {obj_file} 时出错: {e}")
    
    print("\n" + "=" * 60)
    print("转换完成！")
    print("=" * 60)


if __name__ == '__main__':
    convert_finger_objs()
