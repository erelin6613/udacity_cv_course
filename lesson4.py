import os
import cv2

from utils import read_image, linux_display

img_path1 = os.path.join("image_samples", "racoon.jpg")


def q13():

    borders = [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE,
        cv2.BORDER_REFLECT, cv2.BORDER_ISOLATED]
    image = read_image(img_path1)
    linux_display(image, "racoon")
    filt = cv2.getGaussianKernel(9, 0)
    for bord in borders:
        img_filtered = cv2.sepFilter2D(image, 0, filt, filt, borderType=bord)
        linux_display(img_filtered, f"border: {str(bord)}")


if __name__ == '__main__':
    q13()
