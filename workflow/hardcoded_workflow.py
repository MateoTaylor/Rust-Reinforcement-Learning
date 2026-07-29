"""
trying to hardcode farming workflow to generate frames for imitation learning

"""
import time
import cv2
import numpy as np
from ultralytics import YOLO
from feature.reward_calculation import calculate_reward
from training_env.environment_control_updated import EnvironmentControl
import torch
from conf.conf import Config
import os
import pandas as pd

SAVE_DIR = "pretraining/"
FRAME_DIR = "pretraining/imitation_frames/"

os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

if __name__ == "__main__":
    env = EnvironmentControl()
    INTERVAL = 0.25  # (4 FPS)
    yolo = YOLO("model/best.pt")
    print("Starting hardcoded workflow in 2s...")
    frame_count = 40000
    data_log = []
    time.sleep(2)
    swung_previously = False
    
    try:
        for episode in range(1000):
            print(f"Starting episode {episode+1}/1000")
            env.reset()
            state, info = env.step([0] * len(Config.ACTION_DIM))
            for step in range(80):
                start_time = time.time()
                state, next_info = env.get_state()
                reward, reward_info = calculate_reward(info, next_info)

                # state is normalized C,H,W, so convert back to uint8 H,W,C for Ultralytics
                frame = (state.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                
                results = yolo(frame, verbose=False, classes=[0], conf=0.30)  # only look for rocks, confidence threshold 0.30
                
                # convert from rgb to bgr and then save frame to frame dir 
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_filename = os.path.join(FRAME_DIR, f"frame_{frame_count:06d}.jpg")
                success = cv2.imwrite(frame_filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if not success:
                    # kill the thread if writing fails
                    raise IOError(f"Failed to write frame to {frame_filename}")

                detections = []
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    largest_height = 0
                    largest_center_x = 0
                    largest_center_y = 0
                    for x1, y1, x2, y2 in boxes.xyxy.cpu().tolist():
                        height = y2 - y1
                        center_x = (x1 + x2) / 2.0
                        center_y = (y1 + y2) / 2.0
                        if height > largest_height:
                            largest_height = height
                            largest_center_x = center_x
                            largest_center_y = center_y
                        
                    # try to center the largest rock in the middle of the screen
                    action = np.array([1, 0, 0, 0, reward])  # default to moving forward
                    if largest_center_x < 160 - 10:
                        action[2] = 1  # turn left
                    elif largest_center_x > 160 + 10:
                        action[2] = 2  # turn right
                    if largest_center_y < 160 - 10:
                        action[3] = 1  # look up
                    elif largest_center_y > 160 + 10:
                        action[3] = 2  # look down

                    # if we just swung, 50% to continue the motion and stop moving
                    if swung_previously:
                        if np.random.rand() < 0.50:
                            action[1] = 1  # swing pickaxe again

                    # if largest rock is more than 200 pixels tall, or we just tried to swing pickaxe: 4/5 chance to not move forward
                    if largest_height > 85:
                        action[1] = 1  # swing pickaxe
                        swung_previously = True
                        if np.random.rand() < 0.80:
                            action[0] = 0  # stop moving forward
                else:
                    # if no detections, spin around on x, while forward movement and y are random
                    action = np.array([np.random.choice([0, 1]), 0, 1, np.random.choice([0, 1, 2]), reward]) 
                
                data_log.append(action.tolist())
                env.step(action)
                elapsed = time.time() - start_time
                time_to_wait = INTERVAL - elapsed
                if time_to_wait > 0:
                    time.sleep(time_to_wait)
                frame_count += 1
            env.pause()  # pause between episodes
        
        # once all episodes are done, save data log to csv
        df = pd.DataFrame(data_log, columns=['forward','swing','mouse_dx_bin','mouse_dy_bin', 'reward'])
        log_filename = os.path.join(SAVE_DIR, "movement_log.csv")
        df.to_csv(log_filename, index=False)
        print(f"Saved log to {log_filename}")
    except KeyboardInterrupt:
        print("Stopping recording...")
        df = pd.DataFrame(data_log, columns=['forward','swing','mouse_dx_bin','mouse_dy_bin', 'reward'])
        log_filename = os.path.join(SAVE_DIR, "movement_log.csv")
        df.to_csv(log_filename, index=False)
        print(f"Saved log to {log_filename}")
    except Exception as e:
        print(f"Error occurred: {e}")
        df = pd.DataFrame(data_log, columns=['forward','swing','mouse_dx_bin','mouse_dy_bin', 'reward'])
        log_filename = os.path.join(SAVE_DIR, "movement_log.csv")
        df.to_csv(log_filename, index=False)
        print(f"Saved log to {log_filename}")
        raise