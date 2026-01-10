"""
Main interface for controlling the training environment
"""

import pydirectinput as pdi
import time

import numpy as np
import training_env.send_message as send_message
from torchvision import transforms
import cv2
from mss import mss
import win32gui
pdi.PAUSE = 0.005

class EnvironmentControl:
    def __init__(self):
       self.rust_window_title = "Rust"  # Adjust if needed
       self.sct = mss()
    def reset(self):
        send_message.reset_env()
        time.sleep(3) # wait for environment to reset
        send_message.give_pickaxe()
        time.sleep(2)
        pdi.press("3") # select pickaxe after respawn
        time.sleep(0.5)
    
    def is_rust_focused(self):
        """Check if Rust window is in focus"""
        try:
            foreground_window = win32gui.GetForegroundWindow()
            window_title = win32gui.GetWindowText(foreground_window)
            return self.rust_window_title in window_title
        except:
            return False

    def get_state(self):
        # check to make sure the main window is still active
        
        monitor = self.sct.monitors[1]  # primary monitor
        screenshot = np.array(self.sct.grab(monitor))

        # Match pretraining pipeline: BGRA → BGR → RGB
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)   # Then to RGB (like .convert("RGB") does)
        screenshot = cv2.resize(screenshot, (320, 320))  # resize to 320x320
        
        # Convert to tensor - transforms.ToTensor() does permute + divide by 255
        transform = transforms.ToTensor()
        screenshot_tensor = transform(screenshot)  # Now (C, H, W) and normalized
        # now downsize to 320x320
        game_state = send_message.get_state()
        
        return screenshot_tensor, game_state

    def step(self, action):
        """
        Execute action in the environment and return next_state, reward, done, info
        action: tensor of shape [num_action_dims]
        """
        # Execute actions based on the action indices
        # Action 0: Swing pickaxe
        if not self.is_rust_focused():
            raise Exception("Rust window out of focus)")
        if action[0] == 1: pdi.keyDown('w')
        else: pdi.keyUp('w')
        if action[1] == 1: pdi.keyDown('a')
        else: pdi.keyUp('a')
        if action[2] == 1: pdi.keyDown('s')
        else: pdi.keyUp('s')
        if action[3] == 1: pdi.keyDown('d')
        else: pdi.keyUp('d')
        if action[4] == 1: pdi.keyDown('space')
        else: pdi.keyUp('space')
        if action[5] == 1: pdi.click()
        
        mouse_movement_scale = [-50, -10, 0, 10, 50] # scale for mouse movement
        mouse_dx = mouse_movement_scale[action[6]]*4
        mouse_dy = mouse_movement_scale[action[7]]*2
        pdi.moveRel(mouse_dx, mouse_dy, relative=True)
        # after executing action, get new state and return it with reward info
        next_state, extra_info = self.get_state()
        info = extra_info

        return next_state, info
    
    def pause(self):
        # lift up all keys
        pdi.keyUp('w')
        pdi.keyUp('a')
        pdi.keyUp('s')
        pdi.keyUp('d')
        pdi.keyUp('space')
