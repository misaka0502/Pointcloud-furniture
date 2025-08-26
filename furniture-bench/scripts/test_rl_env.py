from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv
import contextlib
import os
from pathlib import Path
from scripts.observation import FULL_OBS
import pygame
import torch
import numpy as np
import time

@contextlib.contextmanager
def suppress_all_output(disable=True):
    if disable:
        null_fd = os.open(os.devnull, os.O_RDWR)
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
    try:
        yield
    finally:
        if disable:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(null_fd)

def get_rl_env(
    gpu_id,
    task: str = "one_leg",
    num_envs=1,
    randomness="low",
    max_env_steps=5_000,
    resize_img=True,
    observation_space="image",  # Observation space for the robot. Options are 'image' and 'state'.
    act_rot_repr="quat",
    action_type="pos",  # Action type for the robot. Options are 'delta' and 'pos'.
    april_tags=False,
    verbose=False,
    headless=False,
    record=False,
    concat_robot_state=False,
    ctrl_mode="diffik",
    obs_keys=None,
    **kwargs,
):
    from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv

    if not april_tags:
        from furniture_bench.envs import furniture_sim_env

        furniture_sim_env.ASSET_ROOT = str(
            Path(__file__).parent.parent.absolute() / "assets"
        )

    # To ensure we can replay the rollouts, we need to (1) include all robot states in the observation space
    # and (2) ensure that the robot state is stored as a dict for compatibility with the teleop data
    
    if obs_keys is None:
        obs_keys = FULL_OBS
        if observation_space == "state":
            # Filter out keys with `image` in them
            obs_keys = [key for key in obs_keys if "image" not in key]
            if num_envs == 1:
                obs_keys = obs_keys + ["color_image1"] + ["depth_image1"] + ["color_image2"] + ["depth_image2"]

    if action_type == "relative":
        print(
            "[INFO] Using relative actions. This keeps the environment using position actions."
        )
    action_type = "pos" if action_type == "relative" else action_type

    with suppress_all_output(False):
        env = FurnitureRLSimEnv(
            furniture=task,  # Specifies the type of furniture [lamp | square_table | desk | drawer | cabinet | round_table | stool | chair | one_leg].
            num_envs=num_envs,  # Number of parallel environments.
            resize_img=resize_img,  # If true, images are resized to 224 x 224.
            concat_robot_state=concat_robot_state,  # If true, robot state is concatenated to the observation.
            headless=headless,  # If true, simulation runs without GUI.
            obs_keys=obs_keys,
            compute_device_id=gpu_id,
            graphics_device_id=gpu_id,
            init_assembled=False,  # If true, the environment is initialized with assembled furniture.
            np_step_out=False,  # If true, env.step() returns Numpy arrays.
            channel_first=False,  # If true, images are returned in channel first format.
            randomness=randomness,  # Level of randomness in the environment [low | med | high].
            high_random_idx=-1,  # Index of the high randomness level (range: [0-2]). Default -1 will randomly select the index within the range.
            save_camera_input=False,  # If true, the initial camera inputs are saved.
            record=record,  # If true, videos of the wrist and front cameras' RGB inputs are recorded.
            max_env_steps=max_env_steps,  # Maximum number of steps per episode.
            act_rot_repr=act_rot_repr,  # Representation of rotation for action space. Options are 'quat' and 'axis'.
            ctrl_mode="diffik",  # Control mode for the robot. Options are 'osc' and 'diffik'.
            action_type=action_type,  # Action type for the robot. Options are 'delta' and 'pos'.
            verbose=verbose,  # If true, prints debug information.
            # **kwargs,
        )

    return env

        
def main():
    env = get_rl_env(gpu_id=0)

    def action_tensor(ac):
        if isinstance(ac, (list, np.ndarray)):
            return torch.tensor(ac).float().to(env.device)

        ac = ac.clone()
        if len(ac.shape) == 1:
            ac = ac[None]
        return ac.tile(env.num_envs, 1).float().to(env.device)

    obs = env.reset()
    while True:
        action = action_tensor([0, 0, 0, 0, 0, 0, 1, -1])
        obs, reward, done, _ = env.step(action, sample_perturbations=False)
        time.sleep(0.01)

if __name__ == '__main__':
    main()