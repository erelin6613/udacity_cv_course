import os
import cv2

from utils import read_image, linux_display

img_path1 = os.path.join("image_samples", "racoon.jpg")


def q13():
    image = read_image(img_path1)
    linux_display(image, "racoon")
    filt = cv2.getGaussianKernel(ksize=5, sigma=1)
    filtered = cv2.filter2D(src=image, ddepth=-1, kernel=filt)
    linux_display(filtered, "gaus_blur")


if __name__ == '__main__':
    q13()
