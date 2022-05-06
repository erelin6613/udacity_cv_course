import os
import numpy as np
import cv2

from utils import read_image, linux_display

img_path = os.path.join("image_samples", "neutron_star.jpg")


def compute_integral(img):
    return np.cumsum(np.cumsum(img, 0), 1)


def main():
    img = read_image(img_path, "gray")
    int_img = compute_integral(img)
    # disp_img =
    linux_display(int_img / int_img[-1, -1], "image")

    x1, x2 = 100, 200
    y1, y2 = 100, 200

    print(sum(img[x1:x2, y1:y2].flatten()))
    print(int_img[x2, y2] - int_img[x1, y2] - int_img[x2, y1] + int_img[x1, y1])


if __name__ == '__main__':
    main()
