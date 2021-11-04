import cv2
import numpy as np
from scipy.signal import correlate

from utils import read_image, downscale_image

img_path1 = "image_samples\\elf.jpg"


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
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.imshow("patch", patch)
    cv2.waitKey(0)
    img2 = template_match_2d(image, patch)
    cv2.imshow("matched", img2)
    cv2.waitKey(0)


if __name__ == '__main__':
    q4()
    q6()
