"""Instantiate FurnitureSim-v0 and test various functionalities."""

import argparse
import pickle

import furniture_bench
import furniture_bench.device

import gym
import cv2
import torch
import numpy as np
import imageio
from synthesize_pcd.utils.furniture import Furniture, sample_points, draw_point_cloud, record_point_cloud_animation_imageio
from synthesize_pcd.utils.visualizer import PointCloudVisualizer
import open3d as o3d
import time
from scripts.observation import FULL_OBS

ASEET_PATH = 'synthesize_pcd/assets/furniture_bench/mesh/square_table'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--furniture", default="square_table")
    parser.add_argument(
        "--file-path", help="Demo path to replay (data directory or pickle)"
    )
    parser.add_argument(
        "--scripted", action="store_true", help="Execute hard-coded assembly script."
    )
    parser.add_argument("--no-action", action="store_true")
    parser.add_argument("--random-action", action="store_true")
    parser.add_argument(
        "--input-device",
        help="Device to control the robot.",
        choices=["keyboard", "oculus", "keyboard-oculus", "omega7"],
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--init-assembled",
        action="store_true",
        help="Initialize the environment with the assembled furniture.",
    )
    parser.add_argument(
        "--save-camera-input",
        action="store_true",
        help="Save camera input of the simulator at the beginning of the episode.",
    )
    parser.add_argument(
        "--record", action="store_true", help="Record the video of the simulator."
    )
    parser.add_argument(
        "--high-res",
        action="store_true",
        help="Use high resolution images for the camera input.",
    )
    parser.add_argument(
        "--randomness",
        default="low",
        help="Randomness level of the environment.",
    )
    parser.add_argument(
        "--high-random-idx",
        default=0,
        type=int,
        help="The index of high_randomness.",
    )
    parser.add_argument(
        "--env-id",
        default="FurnitureSim-v0",
        help="Environment id of FurnitureSim",
    )
    parser.add_argument(
        "--replay-path", type=str, help="Path to the saved data to replay action."
    )

    parser.add_argument(
        "--act-rot-repr",
        type=str,
        help="Rotation representation for action space.",
        choices=["quat", "axis", "rot_6d"],
        default="quat",
    )

    parser.add_argument(
        "--compute-device-id",
        type=int,
        default=0,
        help="GPU device ID used for simulation.",
    )

    parser.add_argument(
        "--graphics-device-id",
        type=int,
        default=0,
        help="GPU device ID used for rendering.",
    )

    parser.add_argument(
        "--display-pcd",
        action="store_true",
        default=False,
        help="Whether to display point cloud",
    )

    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()

    # Create FurnitureSim environment.
    env = gym.make(
        args.env_id,
        furniture=args.furniture,
        num_envs=args.num_envs,
        resize_img=not args.high_res,
        obs_keys=FULL_OBS,
        init_assembled=args.init_assembled,
        record=args.record,
        headless=args.headless,
        save_camera_input=args.save_camera_input,
        randomness=args.randomness,
        high_random_idx=args.high_random_idx,
        act_rot_repr=args.act_rot_repr,
        compute_device_id=args.compute_device_id,
        graphics_device_id=args.graphics_device_id,
        action_type='pos',
        ctrl_mode='diffik'
    )

    if args.display_pcd:
        furniture = Furniture(ASEET_PATH, device='cuda:0', downsample_voxel_size=0.001)
        visualizer = PointCloudVisualizer()
        part_pose = {}
    
    # create device interface
    if args.input_device is not None:
        # Teleoperation.
        device_interface = furniture_bench.device.make_device(args.input_device, control_mode=env.action_type, 
                                                              robot_workspace_center=env.init_ee_pos[0].cpu().numpy(), robot_init_quat=env.init_ee_quat[0].cpu().numpy())

    # Initialize FurnitureSim.
    ob = env.reset()
    done = False

    def action_tensor(ac):
        if isinstance(ac, (list, np.ndarray)):
            return torch.tensor(ac).float().to(env.device)

        ac = ac.clone()
        if len(ac.shape) == 1:
            ac = ac[None]
        return ac.tile(args.num_envs, 1).float().to(env.device)

    # Rollout one episode with a selected policy:
    if args.input_device is not None:
        while not done:
            if args.input_device == "omega7":
                device_interface.update()
            action, _ = device_interface.get_action()
            action = action_tensor(action)
            ob, rew, done, _ = env.step(action)
            ee_pos, ee_quat = env.get_ee_pose()
            img_wrist_np = ob['color_image1'].squeeze(0).cpu().numpy().astype(np.uint8)
            img_front_np = ob['color_image2'].squeeze(0).cpu().numpy().astype(np.uint8)
            img_wrist_bgr = cv2.cvtColor(img_wrist_np, cv2.COLOR_RGB2BGR)
            img_front_bgr = cv2.cvtColor(img_front_np, cv2.COLOR_RGB2BGR)
            cv2.putText(img_wrist_bgr, 'Wrist Camera', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img_front_bgr, 'Front Camera', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            combined_image = np.hstack((img_wrist_bgr, img_front_bgr))
            cv2.imshow("camera", combined_image)
            # if args.display_pcd:
            #     part_pose['square_table_top'] = ob['parts_poses'][:, :7]
            #     part_pose['square_table_leg1'] = ob['parts_poses'][:, 7:14]
            #     part_pose['square_table_leg2'] = ob['parts_poses'][:, 14:21]
            #     part_pose['square_table_leg3'] = ob['parts_poses'][:, 21:28]
            #     part_pose['square_table_leg4'] = ob['parts_poses'][:, 28:35]
            #     furniture.get_pcd_from_offline_data(part_pose)
            #     # pcds_sampled = sample_points(torch.cat(list(furniture.parts_pcds_world.values())), sample_num=4096)
            #     first_env_pcds_parts = {
            #         part_name: batched_pcd[0].unsqueeze(0)  # 使用索引 [0] 来选择第一个环境
            #         for part_name, batched_pcd in furniture.parts_pcds_world.items()
            #     }
            #     pcd_to_sample_single_env = torch.cat(list(first_env_pcds_parts.values()), dim=0)
            #     pcds_sampled = sample_points(pcd_to_sample_single_env, sample_num=4096)
            #     pcds_sampled = sample_points(pcd_to_sample_single_env, sample_num=4096)
            #     if visualizer.update_point_cloud(pcds_sampled): # 非阻塞式，循环更新点云
            #         time.sleep(0.01)
            #     else: 
            #         break
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27: # 如果按下 'esc' 键
                print("'esc' key pressed, exiting.")
                break

            if key == ord('r'):  # 如果按下 'r' 键
                print("'r' key pressed, resetting environment.")
                if args.input_device is not None and args.input_device == "omega7":
                    device_interface.reset()
                ob = env.reset()
                if env.ctrl_mode == "osc":
                    # 如果控制器是osc，需要发送一个和初始位置有偏差的动作，机械臂才能迅速回到初始位置,否则会“弹射”到reset前的状态缓慢回到初始位置（osc控制器的实现细节导致的，需要进一步debug）
                    # 使用dikkik控制器能够避免这个问题
                    action = action_tensor([env.init_ee_pos[0, 0], env.init_ee_pos[0, 1], env.init_ee_pos[0, 2], env.init_ee_quat[0, 0], env.init_ee_quat[0, 1], env.init_ee_quat[0, 2], env.init_ee_quat[0, 3], -1])
                    action_ = action * 1.05
                    ob, rew, done, _ = env.step(action_)
                    ob, rew, done, _ = env.step(action)
                done = False
            
            # 检查窗口是否被手动关闭
            if cv2.getWindowProperty("camera", cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed, exiting.")
                break

    elif args.no_action or args.init_assembled:
        # Execute 0 actions.
        while True:
            if env.action_type == 'delta':
                if args.act_rot_repr == "quat":
                    ac = action_tensor([0, 0, 0, 0, 0, 0, 1, -1])
                else:
                    ac = action_tensor([0, 0, 0, 0, 0, 0, -1])
                ob, rew, done, _ = env.step(ac)
            elif env.action_type == 'pos':
                # ee_pos, ee_quat = env.get_ee_pose()
                if args.act_rot_repr == "quat":
                    ac = action_tensor([env.init_ee_pos[0, 0], env.init_ee_pos[0, 1], env.init_ee_pos[0, 2], env.init_ee_quat[0, 0], env.init_ee_quat[0, 1], env.init_ee_quat[0, 2], env.init_ee_quat[0, 3], -1])
                else:
                    raise NotImplementedError
                ob, rew, done, _ = env.step(ac)
            else:
                raise NotImplementedError

            img_wrist_np = ob['color_image1'].squeeze(0).cpu().numpy().astype(np.uint8)
            img_front_np = ob['color_image2'].squeeze(0).cpu().numpy().astype(np.uint8)
            img_wrist_bgr = cv2.cvtColor(img_wrist_np, cv2.COLOR_RGB2BGR)
            img_front_bgr = cv2.cvtColor(img_front_np, cv2.COLOR_RGB2BGR)
            cv2.putText(img_wrist_bgr, 'Wrist Camera', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img_front_bgr, 'Front Camera', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            combined_image = np.hstack((img_wrist_bgr, img_front_bgr))
            cv2.imshow("camera", combined_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):  # 如果按下 'r' 键
                print("'r' key pressed, resetting environment.")
                ob = env.reset()
                done = False
                env.step()

    elif args.random_action:
        # Execute randomly sampled actions.
        import tqdm

        pbar = tqdm.tqdm()
        while True:
            ac = action_tensor(env.action_space.sample())
            ob, rew, done, _ = env.step(ac)
            pbar.update(args.num_envs)

    elif args.file_path is not None:
        # Play actions in the demo.
        with open(args.file_path, "rb") as f:
            data = pickle.load(f)
        for ac in data["actions"]:
            ac = action_tensor(ac)
            env.step(ac)
    elif args.scripted:
        # Execute hard-coded assembly script.
        while not done:
            action, skill_complete = env.get_assembly_action()
            action = action_tensor(action)
            ob, rew, done, _ = env.step(action)
    elif args.replay_path:
        # Replay the trajectory.
        with open(args.replay_path, "rb") as f:
            data = pickle.load(f)
        env.reset_to([data["observations"][0]])  # reset to the first observation.
        for ac in data["actions"]:
            ac = action_tensor(ac)
            ob, rew, done, _ = env.step(ac)
    else:
        raise ValueError(f"No action specified")

    print("done")


if __name__ == "__main__":
    main()