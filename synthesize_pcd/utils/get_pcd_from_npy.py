import numpy as np
import utils.fb_control_utils as C
import os
import glob
import torch

def get_pcd_from_offline_data(asset_path: str, device: int = 'cuda'):
    if not os.path.isdir(asset_path):
        print(f"Error: directory '{asset_path}' do not exist。")
        return {}
     
    pattern = os.path.join(asset_path, '*.npy')
    npy_file_paths = glob.glob(pattern)
    if not npy_file_paths:
        print(f"Warning: in directory '{asset_path}' do not find .npy files。")
        return {}

    point_clouds_dict = {} # init a dict to storage pcd

    for file_path in npy_file_paths:
        filename_with_ext = os.path.basename(file_path)
        
        part_name = os.path.splitext(filename_with_ext)[0]
        
        point_cloud_data = np.load(file_path)
        
        point_clouds_dict[part_name] = C.xyz_to_homogeneous(
            torch.tensor(
                point_cloud_data, device=device, dtype=torch.float32
            ),
            device=device,
        )

        print(f"  - load: {part_name} (shape: {point_clouds_dict[part_name].shape}, homogeneous form)")

    return point_clouds_dict