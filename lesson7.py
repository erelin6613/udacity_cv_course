import cv2
import numpy as np
from scipy.signal import correlate

from utils import read_image, downscale_image

img_path1 = "image_samples\\elf.jpg"

def q9():
    # image = read_image(img_path1)
    image = downscale_image(read_image(img_path1))
    cv2.imshow("image", image)
    cv2.waitKey(0)
    image = cv2.medianBlur(image, 3)
    canny = edges = cv2.Canny(image=image, threshold1=50, threshold2=150)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.imshow("edges", canny)
    cv2.waitKey(0)

if __name__ == '__main__':
    q9()
