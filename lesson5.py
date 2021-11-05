import os
import cv2
import numpy as np
from scipy.signal import correlate

from utils import read_image, downscale_image, linux_display

img_path1 = os.path.join("image_samples", "elf.jpg")


def teplate_match_1d(signal, template):
    corr = correlate(signal, template, "full")
    idx = np.argmax(corr)
    return idx-1


def template_match_2d(signal, template):
    corr = cv2.matchTemplate(signal, template, cv2.TM_CCOEFF_NORMED)
    min_, max_, min_l, max_l = cv2.minMaxLoc(corr)
    width, height = template.shape[::-1]
    rect_coord = (max_l, (max_l[0]+width, max_l[1]+height))
    return cv2.rectangle(signal, rect_coord[0], rect_coord[1], 255, 2)


def q4():
    s = [-1, 0, 0, 1, 1, 1, 0, -1, -1, 0, 1, 0, 0, -1]
    t = [1, 1, 0]
    print(s, '\n', t, '\n', teplate_match_1d(s, t))


def q6():
    image = downscale_image(read_image(img_path1))
    patch = image[200:300, 125:275]
    linux_display(image, "image")
    linux_display(patch, "patch")
    img2 = template_match_2d(image, patch)
    linux_display(img2, "matched")


if __name__ == '__main__':
    q4()
    q6()
