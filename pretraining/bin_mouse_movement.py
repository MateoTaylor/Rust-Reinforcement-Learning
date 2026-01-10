import os
import pandas as pd

'''
bin mouse dx and dy into 9 discrete categories:

> 100, 50-100, 10-50, 3-10, 3-(-3), -3--10, -10--50, -50--100, < -100
'''

if __name__ == "__main__":
    log_path = os.path.join("pretraining", "movement_log.csv")
    df = pd.read_csv(log_path, header=0)
    binned_dx = []
    binned_dy = []
    
    bins = [float('-inf'), -100, -50, -10, -3, 3, 10, 50, 100, float('inf')]
    df['modified_delta_x'] = pd.cut(df['mouse_delta_x'], bins=bins, labels=False, right=False)
    df['modified_delta_y'] = pd.cut(df['mouse_delta_y'], bins=bins, labels=False, right=False)


    # now cleanup the array and get it ready for numpy
    df.drop(columns=["frame", 'mouse_delta_x', 'mouse_delta_y'], inplace=True)
    # remove headers
    df.to_csv(os.path.join("pretraining", "binned_movement_log.csv"), index=False, header=False)

