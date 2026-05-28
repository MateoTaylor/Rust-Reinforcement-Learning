import os
import pandas as pd

'''
bin mouse dx and dy into 9 discrete categories:

>50, 10-50, 0, -10--50, -50<
'''

if __name__ == "__main__":
    log_path = os.path.join("pretraining", "movement_log.csv")
    df = pd.read_csv(log_path, header=0)
    # binned_dx = []
    # binned_dy = []
    
    # bins = [float('-inf'), -50, -5, 5, 50, float('inf')]
    # df['modified_delta_x'] = pd.cut(df['mouse_delta_x'], bins=bins, labels=False, right=False)
    # df['modified_delta_y'] = pd.cut(df['mouse_delta_y'], bins=bins, labels=False, right=False)

    # # now cleanup the array and get it ready for numpy
    # df.drop(columns=["frame", 'mouse_delta_x', 'mouse_delta_y'], inplace=True)

    # current cols: frame,w,a,s,d,space,left_click,mouse_delta_x,mouse_delta_y
    # remove a, s, d, space, 
    df.drop(columns=["frame", "a", "s", "d", "space"], inplace=True)
    # mouse movements currently are either -150 or 150, so we can just convert to 0, 1, 2
    df['mouse_delta_x'] = df['mouse_delta_x'].apply(lambda x: 1 if x < 0 else (2 if x > 0 else 0))
    df['mouse_delta_y'] = df['mouse_delta_y'].apply(lambda x: 1 if x < 0 else (2 if x > 0 else 0))

    # # remove headers
    df.to_csv(os.path.join("pretraining", "binned_movement_log.csv"), index=False, header=False)

    
