import cv2
import numpy as np

from utils import read_image

img_path1 = "image_samples\racoon.jpg"

def q13():
    image = read_image(img_path1)
    cv2.imshow("racoon", image)
    cv2.waitKey(0)
    filt = cv2.getGaussianKernel()
    return filt

"""

%% TODO: Create a Gaussian filter

%% TODO: Apply it, specifying an edge parameter (try different parameters)

"""

if __name__ == '__main__':
    print(q13())
