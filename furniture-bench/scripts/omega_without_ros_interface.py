
import math
import numpy as np
from threading import Lock
from ctypes import cast, POINTER, c_double,byref
import time
import torch

import forcedimension_core
from forcedimension_core import dhd as dhd
from furniture_bench.device.device_interface import DeviceInterface
from furniture_bench.data.collect_enum import CollectEnum
import furniture_bench.utils.transform as T

class OmegaDeviceInterface(DeviceInterface):
    
    def __init__(self,):
        if(dhd.open()<0):
            print("failed to open the device")
        
        print("Copyright (C) 2001-2023 Force Dimension")
        print("All Rights Reserved.")
        time.sleep(2.0)
        #enable force 
        dhd.enableForce(True)

        #enable the button
        dhd.emulateButton(True)
        
        dhd.expert.setTimeGuard(-1)

        self.last_button=0
        # 线程锁用于保护共享数据
        self.lock = Lock()
        self.collect_enum=0
        self.reset()

 
        
    def reset(self):
        with self.lock:
            self.current_action = np.zeros(8)  # 默认8维动作数组: [dx, dy, dz, droll, dpitch, dyaw, gripper, command]
            self.last_pos = np.zeros(3)
            self.last_wrist = np.zeros(3)
            self.lastr_gripper=0.
            self.collect_enum = CollectEnum.DONE_FALSE
            self.containers_of_pos=dhd.containers.Vec3()
            self.containers_of_wrist=dhd.containers.Vec3()
            self.gripper_action=np.array([1.0])


    
    # def whether_to_use_velocity_mode(self)->bool:
    #     flagIfComeToEdge=False
    #     x,y,z=self.containers_of_pos
    #     raw,pitch,yaw=self.containers_of_wrist
    #     if(x<)

    #     return flagIfComeToEdge


    
    def get_action(self, use_quat=True):
        """
        获取当前动作和状态枚举
        
        Args:
            use_quat: 是否使用四元数表示旋转
        
        Returns:
            tuple: (action, key_enum)
                - action: 包含位置增量、旋转增量和夹爪状态的数组
                - key_enum: 当前状态枚举
        """
        with self.lock:
            gripper=c_double()
            dhd.getPosition(out=self.containers_of_pos)
            dhd.getOrientationRad(out=self.containers_of_wrist)
            dhd.getGripperGap(byref(gripper))

        #由于omega的角度和长度不够大，所以我决定将他的范围乘以5
        
            self.containers_of_pos=np.array(self.containers_of_pos)*5.+np.array([0.5,0,0.])#原本的世界坐标是0.3,但是要拉出来太痛苦了
            
            self.containers_of_wrist=np.array(self.containers_of_wrist)
            #由于franka和omega设备的参数对不上，现根据二者相关的映射去调整参数，让参数能符合行为
            self.containers_of_wrist[0]=(self.containers_of_wrist[0]+2.16+3.14-1.8)*240/320#原来是270
            self.containers_of_wrist[1]=-(self.containers_of_wrist[1]+0.145)/140*120#原来是180
            self.containers_of_wrist[2]=(self.containers_of_wrist[2]-1.5)*1.95
            
            self.containers_of_wrist=self.containers_of_wrist
            self.containers_of_pos=self.containers_of_pos


            dpos = self.containers_of_pos
            dori = self.containers_of_wrist

            self.last_pos=self.containers_of_pos.copy()
            self.last_wrist=self.containers_of_wrist.copy()



            if gripper is None:
                print("Warning,gripper is None")

            dgripper=(np.array([gripper.value])-self.lastr_gripper)
            self.lastr_gripper=np.array([gripper.value])
    
            # if abs(dgripper[0])<0.001:
            #     dgripper==dgripper

            if dgripper<0. and dgripper<-0.001:
                dgripper=np.array([-1.])
            elif dgripper>0 and dgripper > 0.001:
                dgripper=np.array([1.])
            else:
                dgripper=self.gripper_action
           
            self.gripper_action=dgripper
            self.lastr_gripper=np.array([gripper.value]).copy()
            

            self.collect_enum=dhd.getButton(0)

            if self.last_button==0 and self.collect_enum==1:
                self.enum=3
            else:
                self.enum=0

            self.last_button=self.collect_enum
            # 构建动作数组
            if use_quat:
                dquat = T.mat2quat(T.euler2mat(dori))
                # 确保四元数的第一个元素为正
                if dquat[0] < 0:
                    dquat = -dquat
                action = np.concatenate([dpos, dquat, -dgripper])
            else:
                action = np.concatenate([dpos, dori, -dgripper])
            
            # 保存当前枚举值并重置
            action=torch.from_numpy(action).to('cuda:0').to(torch.float32)
            return action, self.enum
    
    def print_usage(self):
        print("yes")

    def get_force(self,*args):
        force =args

    
    def close(self):
        """关闭接口并释放资源"""
        print("closing omega")
        dhd.expert.setTimeGuard(-1)
        dhd.stop()
