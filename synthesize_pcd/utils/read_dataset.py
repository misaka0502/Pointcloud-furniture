import zarr
import torch


def read_zarr(path: str):
    data = zarr.open(path, mode='r')
    # show the hierarchical structure of zarr
    print(data.tree())
    return data

def prepare_data(path, len, device, num_envs=1):
    data = read_zarr(path)
    part_pose = {}
    for i in range(len):
        # part_pose['square_table_top'] = torch.tensor(data['parts_poses'][i, 0:7], device=device)
        # part_pose['square_table_leg1'] = torch.tensor(data['parts_poses'][i, 7:14], device=device)
        # part_pose['square_table_leg2'] = torch.tensor(data['parts_poses'][i, 14:21], device=device)
        # part_pose['square_table_leg3'] = torch.tensor(data['parts_poses'][i, 21:28], device=device)
        # part_pose['square_table_leg4'] = torch.tensor(data['parts_poses'][i, 28:35], device=device)
        part_pose['square_table_top'] = torch.tensor(data['parts_poses'][i, 0:7], device=device).unsqueeze(0).expand(num_envs, -1)
        part_pose['square_table_leg1'] = torch.tensor(data['parts_poses'][i, 7:14], device=device).unsqueeze(0).expand(num_envs, -1)
        part_pose['square_table_leg2'] = torch.tensor(data['parts_poses'][i, 14:21], device=device).unsqueeze(0).expand(num_envs, -1)
        part_pose['square_table_leg3'] = torch.tensor(data['parts_poses'][i, 21:28], device=device).unsqueeze(0).expand(num_envs, -1)
        part_pose['square_table_leg4'] = torch.tensor(data['parts_poses'][i, 28:35], device=device).unsqueeze(0).expand(num_envs, -1)
        action_pos = torch.tensor(data['action']['pos'][i], device=device).unsqueeze(0).expand(num_envs, -1)

    return part_pose, action_pos

if __name__ == '__main__':
    path = '/home2/zxp/Projects/Juicer_ws/juicer_dataset/processed/diffik/sim/one_leg/teleop/low/success.zarr'
    part_pose, action_pos = prepare_data(path, 10, 'cuda')