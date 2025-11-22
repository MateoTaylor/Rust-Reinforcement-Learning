"""
Main interface for controlling the training environment
"""

import pydirectinput as pdi
import time
from conf.conf import GameConfig
from PIL import ImageGrab
import torch
import numpy as np
import training_env.send_message as send_message

class EnvironmentControl:
    def __init__(self):
       self.movement_speed = GameConfig.MOVEMENT_SPEED
       
        # 0. 2 Swing pickaxe, [No, Yes]
        # 1. 2 Move vs Look, [Move, Look]
        # 2. 4 Movement dir, [None, Forward, Left, Right]
        # 3. 2 Jump, [No, Yes]
        # 4. 3 Movement scale, [1x, 2x, 3x]
        # 5. 4 Mouse Movement, [Up, Down, Left, Right]
        # 6. 3 Mouse movement scale, [15deg, 30deg, 60deg]        

    def reset(self):
        send_message.reset_env()
        time.sleep(3) # wait for environment to reset
        send_message.give_pickaxe()
        time.sleep(2)
        pdi.press("3") # select pickaxe after respawn
        time.sleep(0.5)
    
    def get_state(self):
        screenshot = ImageGrab.grab() # our main feature is a screenshot of the game.

        # translate screenshot into tensor and normalize
        screenshot_array = np.array(screenshot)  # PIL → numpy
        screenshot_tensor = torch.from_numpy(screenshot_array).permute(2, 0, 1).float() / 255.0
        
        # shrink screenshot to desired size
        # screenshot_tensor = torch.nn.functional.interpolate(
        #     screenshot_tensor.unsqueeze(0), size=(360, 640), mode='bilinear', align_corners=False
        # ).squeeze(0)

        screenshot_tensor = torch.nn.functional.interpolate(
            screenshot_tensor.unsqueeze(0), size=(320, 320), mode='bilinear', align_corners=False
        ).squeeze(0)

        # (C, H, W)
        #assert screenshot_tensor.shape == (3, 360, 640), f"Unexpected screenshot shape: {screenshot_tensor.shape}"
        game_state = send_message.get_state()  # get additional game info to calculate reward

        return screenshot_tensor, game_state

    def step(self, action):
        """
        Execute action in the environment and return next_state, reward, done, info
        action: tensor of shape [num_action_dims]
        """
        # Execute actions based on the action indices

        # Action 0: Swing pickaxe
        if action[0] == 1:
            pdi.click()  # left mouse click
        
        # action 1: Move vs Look
        if action[1] == 0:
            # Movew 
            direction = action[2]
            jump = action[3]    
            scale = action[4]
            self.move(direction, jump, scale)
        else:
            if action[0] == 1:
                time.sleep(0.25)  # wait a bit for swing to process
            direction = action[5]
            scale = action[6]
            self.look(direction, scale)

        # after executing action, get new state and return it with reward info
        next_state, extra_info = self.get_state()
        done = False # hardcoded as always false for the time being.
        info = extra_info

        return next_state, done, info
    
    
    def move(self, direction, jump, scale):
        # press movement keys based on direction, jump, and scale
        scale_factors = [1.0, 3.0, 5.0]  # scale multipliers for movement duration
        scale = scale_factors[scale]  # convert scale index to multiplier
        duration = self.movement_speed * scale
        if direction == 1:  # Forward
            pdi.keyDown('w')
        elif direction == 2:  # Left
            pdi.keyDown('a')
        elif direction == 3:  # Right
            pdi.keyDown('d')
        time.sleep(0.1) # nowte we sleep even if movement = None
        if jump == 1:
            pdi.press('space')
        time.sleep(duration - 0.1)
        # release keys
        if direction == 1:
            pdi.keyUp('w')
        elif direction == 2:
            pdi.keyUp('a')
        elif direction == 3:
            pdi.keyUp('d')
    
    def look(self, mouse_move, scale):
        # Convert tensors to scalars
        mouse_move = int(mouse_move.item()) if torch.is_tensor(mouse_move) else int(mouse_move)
        scale = int(scale.item()) if torch.is_tensor(scale) else int(scale)
        
        # Get screen size
        screen_width, screen_height = 2560, 1440 # hardcoded for now
        # Define movement as percentage of screen
        movement_map = {
            0: (0, -screen_height),  # Up
            1: (0, screen_height),   # Down
            2: (-screen_width, 0),   # Left
            3: (screen_width, 0)     # Right
        } 

        # Scale factors (15%, 30%, 60% based on scale value)
        scale_factors = [0.15 , 0.30, 0.60]

        if mouse_move in movement_map:
            dx, dy = movement_map[mouse_move]
            # Apply scale multiplier
            dx *= scale_factors[scale] * 0.4
            dy *= scale_factors[scale] * 0.4
            pdi.moveRel(int(dx), int(dy), relative=True, duration=0.1)
            time.sleep(0.1)
