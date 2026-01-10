'''
trying to see if the mouse movement recorded / online is the same.'''

from pynput import mouse
import pydirectinput as pdi
from random import randint
import time

raw_mouse_dx = 0  # Move to global scope
raw_mouse_dy = 0  # Move to global scope

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
    # Check DPI scaling
    mouse_listener = mouse.Listener(on_move=on_move)
    mouse_listener.start()
    time.sleep(2)
    for _ in range(10):
        # Reset BEFORE the move
        raw_mouse_dx, raw_mouse_dy = 0, 0
        time.sleep(0.3)
        random_x = randint(-100, 100)
        random_y = randint(-100, 100)

        pdi.moveRel(random_x, random_y)

        # Give more time for all events to process
        time.sleep(0.3)
        

        dx = raw_mouse_dx
        dy = raw_mouse_dy
        print(f"Intended move: ({random_x}, {random_y}), Recorded move: ({dx}, {dy})")

    mouse_listener.stop()