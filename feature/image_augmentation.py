import cv2
import numpy as np

if __name__ == "__main__":

    # current directory
    dir_path = "C:/Users/mateo/Desktop/RustServer/ppo_agent/feature/"
    img = cv2.imread(dir_path + "frame_000000.jpg")

    # heavy bilateral filter
    # img = cv2.bilateralFilter(img, 8, 200, 200)

    # downscale from 640x360 to 80x45
    img = cv2.resize(img, (256, 160))

    # save the augmented image
    cv2.imwrite(dir_path + "frame_000000_augmented1.jpg", img)
