'''
Record movements in Rust using Windows Raw Input API for raw mouse deltas.
Works on Windows 10/11 - uses native Windows API, no external drivers needed.
Must be run with focus on a window (creates hidden window to capture input).
'''

import time
import os
import cv2
import pandas as pd
from mss import mss
from pynput import keyboard as pynput_keyboard
from PIL import Image
import numpy as np
import training_env.send_message as send_message
import ctypes
from ctypes import wintypes
import threading

SAVE_DIR = "pretraining/"
FRAME_DIR = "pretraining/imitation_frames/"
INTERVAL = 0.25  # 4 fps

current_keys = {"w": 0, "a": 0, "s": 0, "d": 0, "space": 0, "left_click": 0}
raw_mouse_dx = 0
raw_mouse_dy = 0

# Windows API constants
WM_INPUT = 0x00FF
RIM_TYPEMOUSE = 0
RIM_TYPEKEYBOARD = 1
RIDEV_INPUTSINK = 0x00000100
RID_INPUT = 0x10000003

# Virtual key codes
VK_W = 0x57
VK_A = 0x41
VK_S = 0x53
VK_D = 0x44
VK_SPACE = 0x20

# Windows structures that aren't in wintypes
# Use c_int64/c_uint64 for proper 64-bit handling
LRESULT = ctypes.c_int64 if ctypes.sizeof(ctypes.c_void_p) == 8 else wintypes.LONG
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

# Define types that may not be in older wintypes
HINSTANCE = wintypes.HANDLE
HICON = wintypes.HANDLE
HCURSOR = wintypes.HANDLE
HBRUSH = wintypes.HANDLE

class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", HINSTANCE),
        ("hIcon", HICON),
        ("hCursor", HCURSOR),
        ("hbrBackground", HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR)
    ]

class MSG(ctypes.Structure):
    _fields_ = [
        ("hWnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", wintypes.POINT)
    ]

# Raw Input structures
class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND)
    ]

class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM)
    ]

# Correct RAWMOUSE: usButtonFlags/usButtonData are in a UNION with ulButtons,
# not sequential fields. Getting this wrong shifts lLastX/lLastY to wrong offsets.
class _RAWMOUSE_BUTTONS(ctypes.Structure):
    _fields_ = [
        ("usButtonFlags", wintypes.USHORT),
        ("usButtonData", wintypes.USHORT),
    ]

class _RAWMOUSE_BUTTONS_UNION(ctypes.Union):
    _fields_ = [
        ("ulButtons", wintypes.ULONG),
        ("buttons", _RAWMOUSE_BUTTONS),
    ]

class RAWMOUSE(ctypes.Structure):
    _fields_ = [
        ("usFlags", wintypes.USHORT),
        ("_buttons", _RAWMOUSE_BUTTONS_UNION),  # union: 4 bytes (NOT 4+2+2)
        ("ulRawButtons", wintypes.ULONG),
        ("lLastX", wintypes.LONG),
        ("lLastY", wintypes.LONG),
        ("ulExtraInformation", wintypes.ULONG)
    ]

class RAWKEYBOARD(ctypes.Structure):
    _fields_ = [
        ("MakeCode", wintypes.USHORT),
        ("Flags", wintypes.USHORT),
        ("Reserved", wintypes.USHORT),
        ("VKey", wintypes.USHORT),
        ("Message", wintypes.UINT),
        ("ExtraInformation", wintypes.ULONG)
    ]

class RAWINPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [
            ("mouse", RAWMOUSE),
            ("keyboard", RAWKEYBOARD)
        ]
    _fields_ = [
        ("header", RAWINPUTHEADER),
        ("data", _U)
    ]

# Keyboard event listener (using pynput for simplicity)
def on_press(key):
    try:
        if key.char in current_keys:
            current_keys[key.char] = 1
    except AttributeError:
        if key == pynput_keyboard.Key.space:
            current_keys["space"] = 1

def on_release(key):
    try:
        if key.char in current_keys:
            current_keys[key.char] = 0
    except AttributeError:
        if key == pynput_keyboard.Key.space:
            current_keys["space"] = 0

# Debug counters
debug_msg_count = 0
debug_mouse_count = 0
debug_last_print_time = 0
debug_total_msgs = 0

# Window procedure to handle raw input
def wnd_proc(hwnd, msg, wparam, lparam):
    global raw_mouse_dx, raw_mouse_dy, current_keys
    global debug_msg_count, debug_mouse_count, debug_last_print_time, debug_total_msgs
    
    # Track ALL messages received
    debug_total_msgs += 1
    current_time = time.time()
    if debug_total_msgs % 100 == 1 or current_time - debug_last_print_time > 5.0:
        # print(f"DEBUG wnd_proc: Total msgs={debug_total_msgs}, msg_code={hex(msg)}, WM_INPUT msgs={debug_msg_count}")
        if debug_total_msgs % 100 == 1:
            debug_last_print_time = current_time
    
    if msg == WM_INPUT:
        debug_msg_count += 1
        
        # Get size of raw input data
        size = wintypes.UINT()
        result = ctypes.windll.user32.GetRawInputData(
            wintypes.HANDLE(lparam), RID_INPUT, None, ctypes.byref(size),
            ctypes.sizeof(RAWINPUTHEADER)
        )
        
        if result == -1:
            print(f"DEBUG: GetRawInputData size query failed")
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)
        
        # Allocate buffer and get data
        buf = (ctypes.c_byte * size.value)()
        result = ctypes.windll.user32.GetRawInputData(
            wintypes.HANDLE(lparam), RID_INPUT, buf, ctypes.byref(size),
            ctypes.sizeof(RAWINPUTHEADER)
        )
        
        if result != size.value:
            # print(f"DEBUG: GetRawInputData failed: expected {size.value}, got {result}")
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)
        
        # Parse raw input
        raw = ctypes.cast(buf, ctypes.POINTER(RAWINPUT)).contents
        
        # Handle mouse input
        if raw.header.dwType == RIM_TYPEMOUSE:
            debug_mouse_count += 1
            dx = raw.data.mouse.lLastX
            dy = raw.data.mouse.lLastY
            flags = raw.data.mouse.usFlags
            
            # Debug logging every 2 seconds
            current_time = time.time()
            if current_time - debug_last_print_time > 2.0:
                # print(f"DEBUG: MSG count={debug_msg_count}, MOUSE count={debug_mouse_count}")
                # print(f"DEBUG: usFlags={flags}, dx={dx}, dy={dy}, buttons={raw.data.mouse._buttons.buttons.usButtonFlags}")
                debug_last_print_time = current_time
            
            # Capture ALL relative mouse movement
            raw_mouse_dx += dx
            raw_mouse_dy += dy
            
            # Handle mouse buttons
            button_flags = raw.data.mouse._buttons.buttons.usButtonFlags
            if button_flags & 0x0001:  # RI_MOUSE_LEFT_BUTTON_DOWN
                current_keys["left_click"] = 1
            elif button_flags & 0x0002:  # RI_MOUSE_LEFT_BUTTON_UP
                current_keys["left_click"] = 0
        else:
            # Debug: what other input types are we getting?
            current_time = time.time()
            # if current_time - debug_last_print_time > 2.0:
            #     print(f"DEBUG: Non-mouse input type: {raw.header.dwType}")
    
    return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

# Setup user32 with proper return types
user32 = ctypes.windll.user32
user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.DefWindowProcW.restype = LRESULT

# Must keep a reference to the callback to prevent garbage collection!
_wnd_proc_callback = WNDPROC(wnd_proc)

# Create window class and window for raw input
def create_raw_input_window():
    """Create a hidden window to receive raw input messages"""
    # Define window class
    wndclass = WNDCLASSW()
    wndclass.style = 0
    wndclass.lpfnWndProc = _wnd_proc_callback
    wndclass.cbClsExtra = 0
    wndclass.cbWndExtra = 0
    wndclass.hInstance = ctypes.windll.kernel32.GetModuleHandleW(None)
    wndclass.hIcon = None
    wndclass.hCursor = None
    wndclass.hbrBackground = None
    wndclass.lpszMenuName = None
    wndclass.lpszClassName = "RawInputWindowClass"
    
    # Register window class
    if not user32.RegisterClassW(ctypes.byref(wndclass)):
        raise ctypes.WinError()
    
    # Create hidden window
    hwnd = user32.CreateWindowExW(
        0, wndclass.lpszClassName, "RawInputWindow",
        0, 0, 0, 0, 0, None, None, wndclass.hInstance, None
    )
    
    if not hwnd:
        raise ctypes.WinError()
    
    # Register for raw input from mouse
    devices = (RAWINPUTDEVICE * 1)()
    devices[0].usUsagePage = 0x01  # HID_USAGE_PAGE_GENERIC
    devices[0].usUsage = 0x02      # HID_USAGE_GENERIC_MOUSE
    devices[0].dwFlags = RIDEV_INPUTSINK
    devices[0].hwndTarget = hwnd
    
    if not user32.RegisterRawInputDevices(devices, 1, ctypes.sizeof(RAWINPUTDEVICE)):
        raise ctypes.WinError()
    
    # print(f"DEBUG: Raw Input registered for window handle {hwnd}")
    # print(f"DEBUG: usUsagePage={devices[0].usUsagePage}, usUsage={devices[0].usUsage}")
    # print(f"DEBUG: dwFlags={devices[0].dwFlags}, hwndTarget={devices[0].hwndTarget}")
    
    return hwnd

def raw_input_message_loop():
    """Create window AND run message loop on the SAME thread.
    Windows dispatches messages to the thread that created the window."""
    
    # Create window on THIS thread so messages come here
    hwnd = create_raw_input_window()
    
    msg = MSG()
    
    print("DEBUG: Message loop thread started (same thread as window)")
    msg_loop_count = 0
    
    while True:
        result = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
        if result == 0:
            print("DEBUG: GetMessageW returned 0 (WM_QUIT)")
            break
        if result == -1:
            print("DEBUG: GetMessageW returned -1 (error)")
            break
        
        msg_loop_count += 1
        if msg_loop_count % 500 == 1:
            print(f"DEBUG message_loop: Processed {msg_loop_count} msgs, last msg={hex(msg.message)}")
        
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))

if __name__ == "__main__":
    data_log = []
    frame_count = 0
    
    print("Setting up Raw Input API...")
    
    # Start raw input thread (creates window + runs message loop on same thread)
    message_thread = threading.Thread(target=raw_input_message_loop, daemon=True)
    message_thread.start()
    
    # Give thread time to create window and register raw input
    time.sleep(1.0)
    # print(f"DEBUG: Message thread alive: {message_thread.is_alive()}")
    
    # Start keyboard listener
    keyboard_listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    keyboard_listener.start()
    
    print("Starting in 5 seconds... Press Ctrl+C to stop and save.")
    time.sleep(5)  # 5 sec to swap to game window
    print("Recording!")
    
    old_state = send_message.get_state()
    
    with mss() as sct:
        monitor = sct.monitors[1]  # primary monitor
        try:
            while True:
                start_time = time.time()
                
                # Get current mouse deltas and reset
                dx = raw_mouse_dx
                dy = raw_mouse_dy
                raw_mouse_dx, raw_mouse_dy = 0, 0
                
                # Capture screen
                img = np.array(sct.grab(monitor))
                img = cv2.resize(img, (640, 360))
                
                new_state = send_message.get_state()
                old_state = new_state
                
                # Save frame
                frame_filename = os.path.join(FRAME_DIR, f"frame_{frame_count:06d}.jpg")
                success = cv2.imwrite(frame_filename, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if not success:
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
                }
                data_log.append(log_entry)
                
                # Print debug info every 25 frames
                # if frame_count % 25 == 0:
                    # print(f"Frame {frame_count}: dx={dx}, dy={dy}, keys={current_keys}")
                    # print(f"DEBUG main: raw_mouse_dx={raw_mouse_dx}, raw_mouse_dy={raw_mouse_dy}, thread_alive={message_thread.is_alive()}")

                frame_count += 1
                
                # Wait for next interval
                elapsed = time.time() - start_time
                time_to_wait = INTERVAL - elapsed
                if time_to_wait > 0:
                    time.sleep(time_to_wait)
                    
        except KeyboardInterrupt:
            print("\nStopping recording...")
        except Exception as e:
            print(f"Error occurred: {e}")
            raise
        finally:
            # Save data
            df = pd.DataFrame(data_log)
            log_filename = os.path.join(SAVE_DIR, "movement_log.csv")
            df.to_csv(log_filename, index=False)
            print(f"Saved {len(data_log)} frames to {log_filename}")
            
            keyboard_listener.stop()
