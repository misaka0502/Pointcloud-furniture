import gym
import numpy as np
from furniture_bench.device.device_interface import DeviceInterface
from furniture_bench.data.collect_enum import CollectEnum
import forcedimension_core.containers as containers
import forcedimension_core.dhd as dhd
import forcedimension_core.drd as drd
from scipy.spatial.transform import Rotation as R
import time
from ctypes import c_double
from enum import Enum, auto
import furniture_bench.utils.transform as T

class ControlState(Enum):
    STARTUP = auto()
    DRD_HOMING = auto()
    DRD_HOLDING = auto()
    DHD_MANUAL_CONTROL = auto()

# 设定一个阈值来检测用户的意图
FORCE_TAKEOVER_THRESHOLD = 3.0 # 1.0 牛顿的力

class Omega7Interface(DeviceInterface):
    def __init__(self, control_mode='delta', robot_workspace_center: np.array=np.array([0.5673, 0.0554, 0.1239]), 
                 robot_init_quat: np.array=np.array([0.8933, 0.4494, -0.0080, 0.0027]), pos_sensitivity: float=5.0,
                 rot_sensitivity: float=10.0, hybrid_control=True, workspace_radius: float=0.2, dead_zone_ratio: float=0.9):
        """
        Initializes the Omega.7 device.

        Args:
            control_mode (str): The control mode of the device. 'delta' or 'pos' (default: 'delta').
            robot_workspace_center (np.array): The center of the robot's workspace (default: [0.5673, 0.0554, 0.1239]).
            robot_init_quat (np.array): The initial quaternion of the robot (default: [0.8933, 0.4494, -0.0080, 0.0027]).
            pos_sensitivity (float): The position sensitivity of the device (default: 5.0).
            rot_sensitivity (float): The rotation sensitivity of the device (default: 10.0).
            hybrid_control (bool): Whether to use hybrid control (default: True).
            workspace_radius (float): The radius of the robot's workspace, unit: m (default: 0.2).
            dead_zone_ratio (float): The dead zone ratio (default: 0.9).
        """
        print("Initializing Omega.7 device...")

        self.control_mode = control_mode
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.enum = CollectEnum.DONE_FALSE
        self.state = ControlState.STARTUP # initial state
        self.pos = np.zeros(3)
        self.euler = np.array([0.0, 0.0, 52]) # in radian
        self.matrix = np.zeros((3, 3))
        self.gripper_angle = 0.0
        self.gripper_angle_c_double = c_double(self.gripper_angle)
        self.grasp = np.array([-1]) # -1 for open, 1 for close
        self.last_pos = np.zeros(3)
        self.last_euler = np.zeros(3)

        # --- Positional control specific variables ---
        self.robot_workspace_center = robot_workspace_center
        self.robot_init_quat = robot_init_quat
        self.robot_inti_rot = T.quat2mat(self.robot_init_quat)
        self.is_clutched = False
        self.robot_target_pos = np.copy(self.robot_workspace_center)
        self.robot_target_rot = np.identity(3)
        self.clutch_device_pos = np.zeros(3)
        self.clutch_device_rot = np.identity(3)
        self.coord_swap_rot = R.from_euler('zy', [-90, -90], degrees=True).as_matrix()

        # --- hybrid control mode ---
        self.hybrid_control = hybrid_control
        if self.hybrid_control:
            # 边界半径 (米)
            self.center_zone_radius = workspace_radius * dead_zone_ratio
            self.edge_zone_width = workspace_radius * (1.0 - dead_zone_ratio)
            
            # 增量控制的最大速度 (米/秒), 在omega7操作空间的边缘触发
            # 这个值需要根据 pos_sensitivity 进行调整
            self.max_delta_speed = 0.1 
            
            print("Hybrid control enabled.")
            print(f"  - Center zone radius: {self.center_zone_radius * 100:.1f} cm")
            print(f"  - Edge zone width: {self.edge_zone_width * 100:.1f} cm")

        self.reset()

    def reset(self):
        """Resets the internal state of the interface."""
        # # init proprioception
        if self.state == ControlState.DHD_MANUAL_CONTROL:
            if(dhd.close() < 0):
                print("Failed to close the device")
            dhd.os_independent.sleep(2)
        self.pos = np.zeros(3)
        self.euler = np.array([0.0, 0.0, 50.0]) * np.pi / 180 # in radian
        self.matrix = np.zeros((3, 3))
        self.gripper_angle = 0.0
        self.gripper_angle_c_double = c_double(self.gripper_angle)
        self.grasp = np.array([-1]) # -1 for open, 1 for close
        self.state = ControlState.STARTUP

    def update(self):
        """
        Updates the state machine based on the current state and inputs from the device.
        """
        if self.state == ControlState.STARTUP:
            print("State: STARTUP -> DRD_HOMING")
            self.state = ControlState.DRD_HOMING
            # 进入drd模式
            if drd.open() < 0:
                print("Failed to open device")
                dhd.os_independent.sleep(2)
            if not drd.isInitialized() and drd.autoInit() < 0:
                print("Failed to init device")
                dhd.os_independent.sleep(2)
            if drd.start() < 0:
                print("Failed to start device")
                dhd.os_independent.sleep(2)
            if drd.moveToPos(self.pos, block=True) < 0:
                print("Failed to move to target position")
                dhd.os_independent.sleep(2)
            if drd.moveToRot(self.euler, block=True) < 0:
                print("Failed to rotate to target rotation matrix")
                dhd.os_independent.sleep(2)
            if drd.moveToGrip(2.0, block=True) < 0:
                print("Failed to move to target gripper distance")
                dhd.os_independent.sleep(2)
        
        elif self.state == ControlState.DRD_HOMING:
            force = np.zeros(3)
            dhd.getForce(force)
            force_magnitude = np.linalg.norm(force)
            # 检查是否已到达目标位置
            # drd.isMoving() 会在 moveTo* 完成后返回 False
            # 停止移动并且用户施加了一定的力之后退出循环，否则维持在当前位置
            if not drd.isMoving() and (force_magnitude > FORCE_TAKEOVER_THRESHOLD):
                print("State: DRD_HOMING -> DHD_MANUAL_CONTROL")
                self.state = ControlState.DHD_MANUAL_CONTROL
                self.is_clutched = True
                # 退出drd模式
                if drd.stop(True) < 0:
                    print("Failed to stop drd")
                    dhd.os_independent.sleep(2)
                if drd.close() < 0:
                    print("Failed to close drd")
                    dhd.os_independent.sleep(2)
                # 进入dhd模式
                # open the device
                if(dhd.open() < 0):
                    raise RuntimeError("Failed to open the device: ")

                # enable force 
                if(dhd.enableForce(True) < 0):
                    print("Failed to enable force: ")

                # enable the button
                if(dhd.emulateButton(True) < 0):
                    print("Failed to emulate button: ")

                # dhd.expert.setTimeGuard(-1) # for debug
                dhd.getPosition(self.pos) # get position
                dhd.getOrientationRad(self.euler)
                dhd.getGripperAngleDeg(self.gripper_angle_c_double)
                self.euler *= np.array([1., -1., -1.])
                self.matrix = T.euler2mat(self.euler)
                self.gripper_angle = self.gripper_angle_c_double.value


                self.last_pos = self.pos.copy()
                self.last_euler = self.euler.copy()
                
                if self.control_mode == 'pos':
                    self.clutch_device_pos = self.pos.copy()
                    self.clutch_device_rot = self.matrix.copy()
                    print("self.clutch_device_pos: ", self.clutch_device_pos)
            else:
                print(f"\rHolding... Applied Force: {force_magnitude:.2f} N", end="")
                # print(f"\rHolding... Squeeze the jaws to activate", end="")
                drd.hold()

    def get_action(self, use_quat=True):
        """
        Retrieves the current state, calculates the delta from the previous state,
        and returns an incremental action.

        Returns:
            np.ndarray: A 7D vector [pos_x, pos_y, pos_z, roll, pitch, yaw, grasp] or 8D action vector [pos_x, pos_y, pos_z, qx, qy, qz, qw, grasp].
            dict: An empty dictionary for compatibility with some interfaces.
        """
        if self.state != ControlState.DHD_MANUAL_CONTROL:
            if self.control_mode == 'delta':
                return np.array([0, 0, 0, 0, 0, 0, 1, -1]) if use_quat else np.array([0, 0, 0, 0, 0, 0, -1]), self.enum
            elif self.control_mode == 'pos':
                return np.concatenate([self.robot_workspace_center, self.robot_init_quat, np.array([-1])]) if use_quat else np.concatenate([self.robot_workspace_center, R.from_quat(self.robot_init_quat).as_euler('xyz'), np.array([-1])]), self.enum
            else:
                raise NotImplementedError

        if self.control_mode == 'delta':
            # get current state from the device
            dhd.getPosition(self.pos)
            dhd.getOrientationRad(self.euler)
            self.euler *= np.array([1., -1., -1.])
            self.matrix = T.euler2mat(self.euler)
            
            dpos = (self.pos - self.last_pos) * self.pos_sensitivity
            deuler = (self.euler - self.last_euler) * self.rot_sensitivity
            dquat = T.mat2quat(T.euler2mat(deuler))

            self.get_gripper_action()

            if use_quat:
                if dquat[0] < 0:
                    dquat = -dquat
                action = np.concatenate([dpos, dquat, self.grasp])
            else:
                action = np.concatenate([dpos, deuler, self.grasp])

            self.last_pos = self.pos.copy()
            self.last_euler = self.euler.copy()
        elif self.control_mode == 'pos':

            if self.is_clutched:
                dhd.getPosition(self.pos)
                dhd.getOrientationRad(self.euler)
                current_device_pos = self.pos
                self.euler *= np.array([1., -1., -1.])
                current_device_rot = T.euler2mat(self.euler)
                relative_pos = current_device_pos - self.clutch_device_pos
                relative_rot = np.linalg.inv(self.clutch_device_rot) @ current_device_rot
                
                robot_pos_offset = np.array([relative_pos[0], relative_pos[1], relative_pos[2]]) * self.pos_sensitivity
                target_pos = self.robot_workspace_center + robot_pos_offset
                
                target_rot = self.robot_inti_rot @ relative_rot
            
            self.get_gripper_action()

            if use_quat:
                target_quat = T.mat2quat(target_rot)
                action = np.concatenate([target_pos, target_quat, self.grasp])
            else:
                target_euler = T.mat2euler(target_rot)
                action = np.concatenate([target_pos, target_euler, self.grasp])
        else:
            raise NotImplementedError

        return action, self.enum
    
    def get_gripper_action(self):
        dhd.getGripperAngleDeg(self.gripper_angle_c_double)
        self.gripper_angle = self.gripper_angle_c_double.value
        if self.gripper_angle < 2.0:
            self.grasp = np.array([1]) # close
        else:
            self.grasp = np.array([-1]) # open

    def print_usage(self):
        
        print("==============Omega.7 Usage=================")
        print("Move the Omega.7 device to control the robot's end-effector position and orientation.")
        print("Use the gripper to open/close the gripper.")
        print("============================================")