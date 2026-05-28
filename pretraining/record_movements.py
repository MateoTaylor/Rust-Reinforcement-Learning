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
import training_env.send_message as send_message
import pydirectinput as pdi
from training_env.environment_control_updated import EnvironmentControl
import win32gui


SAVE_DIR = "pretraining/"
FRAME_DIR = "pretraining/imitation_frames/"
INTERVAL = 0.25 # 5 fps
pdi.PAUSE = 0.002

current_keys = {"w": 0, "a": 0, "s": 0, "d": 0, "space": 0, "left_click": 0, "up": 0, "down": 0, "left": 0, "right": 0, "shift_r":0}
raw_mouse_dx = 0  # Move to global scope
raw_mouse_dy = 0  # Move to global scope
def is_rust_focused():
        """Check if Rust window is in focus"""
        try:
            foreground_window = win32gui.GetForegroundWindow()
            window_title = win32gui.GetWindowText(foreground_window)
            return "Rust" in window_title
        except:
            return False
        
# listen to keyboard events
def on_press(key):
    try:
        if key.char in current_keys:
            current_keys[key.char] = 1
    except AttributeError:
        if key == keyboard.Key.space:
            current_keys["space"] = 1
        if key == keyboard.Key.up:
            current_keys["up"] = 1
        if key == keyboard.Key.down:
            current_keys["down"] = 1
        if key == keyboard.Key.left:
            current_keys["left"] = 1
        if key == keyboard.Key.right:
            current_keys["right"] = 1
        if key == keyboard.Key.shift_r:
            current_keys["shift_r"] = 1

def on_release(key):
    try:
        if key.char in current_keys:
            current_keys[key.char] = 0
    except AttributeError:
        if key == keyboard.Key.space:
            current_keys["space"] = 0
        if key == keyboard.Key.up:
            current_keys["up"] = 0
        if key == keyboard.Key.down:
            current_keys["down"] = 0
        if key == keyboard.Key.left:
            current_keys["left"] = 0
        if key == keyboard.Key.right:
            current_keys["right"] = 0
        if key == keyboard.Key.shift_r:
            current_keys["shift_r"] = 0

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
    keyboard_listener.start()

    env = EnvironmentControl()

    data_log = []
    frame_count = 12978 
    print("Starting in 5 seconds... Ctrl+C to stop and save.")    
    time.sleep(5) # 10 sec to swap to user window
    env.reset()
    print("Recording!")
    old_state = send_message.get_state()

    with mss() as sct:
        monitor = sct.monitors[1]  # primary monitor
        try:
            while True:
                if not is_rust_focused():
                    raise Exception("Rust window out of focus)")
                start_time = time.time()
                # Capture screen
                img = np.array(sct.grab(monitor))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                img = cv2.resize(img, (320, 320))

                new_state = send_message.get_state()
                old_state = new_state

                # Save frame
                frame_filename = os.path.join(FRAME_DIR, f"frame_{frame_count:06d}.jpg")
                success = cv2.imwrite(frame_filename, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if not success:
                    # kill the thread if writing fails
                    raise IOError(f"Failed to write frame to {frame_filename}")

                # use arrowkeys to control mouse movement in the log (instead of raw mouse movement) to avoid noise and make it easier for the model to learn
                current_keys["mouse_delta_y"] = 0
                current_keys["mouse_delta_x"] = 0
                if current_keys["up"]:
                    current_keys["mouse_delta_y"] = -150
                elif current_keys["down"]:
                    current_keys["mouse_delta_y"] = 150
                
                if current_keys["left"]:
                    current_keys["mouse_delta_x"] = -150 
                elif current_keys["right"]:
                    current_keys["mouse_delta_x"] = 150

                # use pdi to move keys and always click
                pdi.moveRel(current_keys["mouse_delta_x"], current_keys["mouse_delta_y"], relative=True)
                
                # Log current keys and mouse movement
                log_entry = {
                    "frame": frame_filename,
                    "w": current_keys["w"],
                    "a": current_keys["a"],
                    "s": current_keys["s"],
                    "d": current_keys["d"],
                    "space": current_keys["space"],
                    "left_click": current_keys["shift_r"],
                    "mouse_delta_x": current_keys["mouse_delta_x"],
                    "mouse_delta_y": current_keys["mouse_delta_y"],
                }
                data_log.append(log_entry)
                
                start_time = time.time()
                frame_count += 1

                # Wait for next interval
                elapsed = time.time() - start_time
                time_to_wait = INTERVAL - elapsed
                if time_to_wait > 0:
                    time.sleep(time_to_wait)

                # reset env every 45 seconds
                if frame_count % (45*5) == 0:
                    env.reset()
                    time.sleep(5)
                    
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
