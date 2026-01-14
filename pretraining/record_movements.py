'''
record movements in rust and then use them for pretraining rl agent.
'''

import time
import os
import cv2
import pandas as pd
from mss import mss
from pynput import mouse, keyboard
from PIL import Image
import numpy as np
from feature.reward_calculation import calculate_reward
import training_env.send_message as send_message

SAVE_DIR = "pretraining/"
FRAME_DIR = "pretraining/imitation_frames/"
INTERVAL = 0.2 # 5 fps

current_keys = {"w": 0, "a": 0, "s": 0, "d": 0, "space": 0, "left_click": 0}
raw_mouse_dx = 0  # Move to global scope
raw_mouse_dy = 0  # Move to global scope

# listen to keyboard events
def on_press(key):
    try:
        if key.char in current_keys:
            current_keys[key.char] = 1
    except AttributeError:
        if key == keyboard.Key.space:
            current_keys["space"] = 1

def on_release(key):
    try:
        if key.char in current_keys:
            current_keys[key.char] = 0
    except AttributeError:
        if key == keyboard.Key.space:
            current_keys["space"] = 0

def on_click(x, y, button, pressed):
    if button == mouse.Button.left:
        current_keys["left_click"] = 1 if pressed else 0

def on_move(x, y):
    global raw_mouse_dx, raw_mouse_dy 
    if not hasattr(on_move, "last_x"):
        on_move.last_x = x
        on_move.last_y = y
        return  # Don't calculate delta on first call
    raw_mouse_dx += x - on_move.last_x
    raw_mouse_dy += y - on_move.last_y
    on_move.last_x = x
    on_move.last_y = y

if __name__ == "__main__":
    # setup listeners
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)
    keyboard_listener.start()
    mouse_listener.start()

    data_log = []
    frame_count = 0
    print("Starting in 10 seconds... Ctrl+C to stop and save.")    
    time.sleep(10) # 10 sec to swap to user window
    print("Recording!")
    old_state = send_message.get_state()

    with mss() as sct:
        monitor = sct.monitors[1]  # primary monitor
        try:
            while True:
                start_time = time.time()
                
                # Get current mouse position and calculate delta
                dx = raw_mouse_dx
                dy = raw_mouse_dy   
                # reset last position
                raw_mouse_dx, raw_mouse_dy = 0, 0

                # Capture screen
                img = np.array(sct.grab(monitor))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                img = cv2.resize(img, (320, 320))

                new_state = send_message.get_state()
                reward, reward_info = calculate_reward(old_state, new_state)
                old_state = new_state

                # Save frame
                frame_filename = os.path.join(FRAME_DIR, f"frame_{frame_count:06d}.jpg")
                success = cv2.imwrite(frame_filename, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if not success:
                    # kill the thread if writing fails
                    raise IOError(f"Failed to write frame to {frame_filename}")

                # Log current keys and mouse movement
                log_entry = {
                    "frame": frame_filename,
                    "w": current_keys["w"],
                    "a": current_keys["a"],
                    "s": current_keys["s"],
                    "d": current_keys["d"],
                    "space": current_keys["space"],
                    "left_click": current_keys["left_click"],
                    "mouse_delta_x": dx,
                    "mouse_delta_y": dy,
                    "reward": reward
                }
                data_log.append(log_entry)

                frame_count += 1
                # Wait for next interval
                elapsed = time.time() - start_time
                time_to_wait = INTERVAL - elapsed
                if time_to_wait > 0:
                    time.sleep(time_to_wait)
                    
        except KeyboardInterrupt:
            print("Stopping recording...")
            df = pd.DataFrame(data_log)
            log_filename = os.path.join(SAVE_DIR, "movement_log.csv")
            df.to_csv(log_filename, index=False)
            print(f"Saved log to {log_filename}")
        except Exception as e:
            print(f"Error occurred: {e}")
            df = pd.DataFrame(data_log)
            log_filename = os.path.join(SAVE_DIR, "movement_log.csv")
            df.to_csv(log_filename, index=False)
            print(f"Saved log to {log_filename}")
            raise
        finally:
            keyboard_listener.stop()
            mouse_listener.stop()
