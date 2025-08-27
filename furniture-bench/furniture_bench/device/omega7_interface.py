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
    def __init__(self, pos_sensitivity: float=8.0, rot_sensitivity: float=1.0,):
        """
        Initializes the Omega.7 device.

        Args:
            pos_sensitivity: Scaling factor for positional movement. Higher means more robot movement for less device movement.
            rot_sensitivity: Scaling factor for rotational movement.
        """
        print("Initializing Omega.7 device...")

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.enum = CollectEnum.DONE_FALSE
        self.state = ControlState.STARTUP # initial state
        self.reset()

    def reset(self):
        """Resets the internal state of the interface."""
        # # init proprioception
        self.pos = np.zeros(3)
        self.euler = np.zeros(3)
        self.matrix = np.zeros((3, 3))
        self.gripper_angle = 29.0 # unit: degree
        self.gripper_angle_c_double = c_double(self.gripper_angle)
        self.grasp = np.array([-1]) # -1 for open, 1 for close
        self.state = ControlState.STARTUP

        # # reset the device state
        # if drd.open() < 0:
        #     print("Failed to open device")
        #     dhd.os_independent.sleep(2)
        # if not drd.isInitialized() and drd.autoInit() < 0:
        #     print("Failed to init device")
        #     dhd.os_independent.sleep(2)
        # if drd.start() < 0:
        #     print("Failed to start device")
        #     dhd.os_independent.sleep(2)
        # if drd.moveToPos(self.pos, block=True) < 0:
        #     print("Failed to move to target position")
        #     dhd.os_independent.sleep(2)
        # if drd.moveToRot(self.euler, block=True) < 0:
        #     print("Failed to rotate to target rotation matrix")
        #     dhd.os_independent.sleep(2)
        
        # force = np.zeros(3)
        # while True:
        #     # 检查是否已到达目标位置
        #     # drd.isMoving() 会在 moveTo* 完成后返回 False
        #     dhd.getForce(force)
        #     force_magnitude = np.linalg.norm(force)
        #     # 停止移动并且用户施加了一定的力之后退出循环，否则维持在当前位置
        #     if not drd.isMoving() and (force_magnitude > FORCE_TAKEOVER_THRESHOLD):
        #         break
        #     drd.hold()
        #     time.sleep(0.01)

        # # 退出drd模式
        # if drd.stop(True) < 0:
        #     print("Failed to stop drd")
        #     dhd.os_independent.sleep(2)
        # if drd.close() < 0:
        #     print("Failed to close drd")
        #     dhd.os_independent.sleep(2)

        # # 进入dhd模式
        # # open the device
        # if(dhd.open() < 0):
        #     raise RuntimeError("Failed to open the device: ")

        # # enable force 
        # if(dhd.enableForce(True) < 0):
        #     print("Failed to enable force: ")

        # # enable the button
        # if(dhd.emulateButton(True) < 0):
        #     print("Failed to emulate button: ")

        # # dhd.expert.setTimeGuard(-1) # for debug

        # dhd.getPosition(self.pos) # get position
        # dhd.getOrientationRad(self.matrix)
        # dhd.getGripperAngleDeg(self.gripper_angle_c_double)
        # self.euler = R.from_matrix(np.array(self.matrix)).as_euler('xyz')
        # self.gripper_angle = self.gripper_angle_c_double.value

        # self.last_pos = self.pos.copy()
        # self.last_euler = self.euler.copy()

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
                dhd.getOrientationDeg(self.euler)
                dhd.getGripperAngleDeg(self.gripper_angle_c_double)
                # self.euler = R.from_matrix(np.array(self.matrix)).as_euler('xyz')
                # self.euler[0]=(self.euler[0]+2.16+3.14-1.8)*240/320#原来是270
                # self.euler[1]=-(self.euler[1]+0.145)/140*120#原来是180
                # self.euler[2]=(self.euler[2]-1.5)*1.95
                self.matrix = T.euler2mat(self.euler)
                self.gripper_angle = self.gripper_angle_c_double.value


                self.last_pos = self.pos.copy()
                self.last_euler = self.euler.copy()

    def get_action(self, use_quat=True):
        """
        Retrieves the current state, calculates the delta from the previous state,
        and returns an incremental action.

        Returns:
            np.ndarray: A 7D vector [pos_x, pos_y, pos_z, roll, pitch, yaw, grasp] or 8D action vector [pos_x, pos_y, pos_z, qx, qy, qz, qw, grasp].
            dict: An empty dictionary for compatibility with some interfaces.
        """
        if self.state != ControlState.DHD_MANUAL_CONTROL:
            return np.array([0, 0, 0, 0, 0, 0, 1, -1]) if use_quat else np.array([0, 0, 0, 0, 0, 0, -1]), self.enum

        # get current state from the device
        dhd.getPosition(self.pos)
        dhd.getOrientationDeg(self.euler)
        dhd.getGripperAngleDeg(self.gripper_angle_c_double)
        # self.euler[0]=(self.euler[0]+2.16+3.14-1.8)*240/320#原来是270
        # self.euler[1]=-(self.euler[1]+0.145)/140*120#原来是180
        # self.euler[2]=(self.euler[2]-1.5)*1.95
        self.matrix = T.euler2mat(self.euler)
        self.gripper_angle = self.gripper_angle_c_double.value

        
        dpos = (self.pos - self.last_pos) * self.pos_sensitivity
        deuler = (self.euler - self.last_euler) * self.rot_sensitivity
        dquat = T.mat2quat(T.euler2mat(deuler))

        if self.gripper_angle < 2.0:
            self.grasp = np.array([1]) # close
        else:
            self.grasp = np.array([-1])
        
        if use_quat:
            if dquat[0] < 0:
                dquat = -dquat
            action = np.concatenate([dpos, dquat, self.grasp])
        else:
            action = np.concatenate([dpos, deuler, self.grasp])

        self.last_pos = self.pos.copy()
        self.last_euler = self.euler.copy()

        return action, self.enum
    
    def print_usage(self):
        print("==============Omega.7 Usage=================")
        print("Move the Omega.7 device to control the robot's end-effector position and orientation.")
        print("Use the gripper to open/close the gripper.")
        print("============================================")