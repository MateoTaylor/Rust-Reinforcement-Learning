
'''
goal is to open up the mouse movement data and visualize it's distribution, use that to create
discrete bins.
'''

import matplotlib.pyplot as plt
import os
import pandas as pd


if __name__ == "__main__":
    # open movement_log.csv
    log_path = os.path.join("pretraining", "movement_log.csv")
    df = pd.read_csv(log_path, header=0)

    # extract mouse dx and dy
    mouse_dx = df['mouse_delta_x'].values
    mouse_dy = df['mouse_delta_y'].values
    # Filter out values between -5 and 5
    mouse_dx_nonzero = mouse_dx[(mouse_dx < -10) | (mouse_dx > 10)]
    mouse_dy_nonzero = mouse_dy[(mouse_dy < -10) | (mouse_dy > 10)]
    
    # plot hist of both (non-zero values)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(mouse_dx_nonzero, bins=50, color='blue', alpha=0.7)
    plt.title('Mouse DX Distribution (Non-Zero)')
    plt.xlabel('Mouse DX')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    plt.hist(mouse_dy_nonzero, bins=50, color='green', alpha=0.7)
    plt.title('Mouse DY Distribution (Non-Zero)')
    plt.xlabel('Mouse DY')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join("pretraining", "mouse_movement_distribution.png"))
    plt.show()

# bins:
# > 100, 50-100, 10-50, 5-10, 5-(-5), -5--10, -10--50, -50--100, < -100