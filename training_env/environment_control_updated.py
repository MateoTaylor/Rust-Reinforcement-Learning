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
        time.sleep(2) # wait for environment to reset
        send_message.give_pickaxe()
        time.sleep(3)
        pdi.press("3", duration=1) # select pickaxe after respawn
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
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2RGB)
        screenshot = cv2.resize(screenshot, (640, 640))  # resize to 640x640
        
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
        if action[1] == 1: pdi.mouseDown()
        else: pdi.mouseUp()
        mouse_movement_conversion = {1: -150, 2: 150, 0: 0} # convert from binned movement back to actual movement
        mouse_dx = mouse_movement_conversion[action[2]]
        mouse_dy = mouse_movement_conversion[action[3]]
        pdi.moveRel(mouse_dx, mouse_dy, relative=True)
        # after executing action, get new state and return it with reward info
        next_state, extra_info = self.get_state()
        info = extra_info

        return next_state, info
    
    def pause(self):
        # lift up all keys
        pdi.keyUp('w')
