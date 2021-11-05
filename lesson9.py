import os
from itertools import product
import cv2
import numpy as np

from utils import read_image, downscale_image, linux_display

img_path1 = os.path.join("image_samples", "browsers.jpg")


def hough_circles_finding():
    image = read_image(img_path1)
    linux_display(image, "image")

    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1,
                                image.shape[0] / 8, param1=100, param2=30,
                                minRadius=10, maxRadius=100)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            cv2.circle(image, center, 1, 0, 2)
            radius = i[2]
            cv2.circle(image, center, radius, 0, 2)

    linux_display(image, "detected_circles")


if __name__ == '__main__':
    hough_circles_finding()
