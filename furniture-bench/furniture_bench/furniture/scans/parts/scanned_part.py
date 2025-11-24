import numpy as np

from furniture_bench.config import config
from furniture_bench.furniture.parts.part import Part


class ScannedPart(Part):
    def __init__(self, part_config: dict, part_idx: int):
        super().__init__(part_config, part_idx)

        reset_pos_base = part_config["reset_pos"].copy()
        reset_ori_base = part_config.get("reset_ori").copy()

        # These are expressed in the robot base frame for scans - re-build in tag frame
        self.reset_pos = []
        self.reset_ori = []
        for i in range(len(reset_ori_base)):
            reset_pose_base = reset_ori_base[i].copy()
            reset_pose_base[:-1, -1] = np.asarray(reset_pos_base)

            reset_pose_tag = (
                np.linalg.inv(config["robot"]["tag_base_from_robot_base"])
                @ reset_pose_base
            )
            ori = np.eye(4)
            ori[:-1, :-1] = reset_pose_tag[:-1, :-1]

            self.reset_pos.append(reset_pose_tag[:-1, -1].tolist())
            self.reset_ori.append(ori)
