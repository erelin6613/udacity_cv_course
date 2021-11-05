import os
import cv2

from utils import read_image, downscale_image, linux_display

img_path1 = os.path.join("image_samples", "elf.jpg")


def q9():
    image = downscale_image(read_image(img_path1))
    linux_display(image, "image")
    image = cv2.medianBlur(image, 3)
    canny = cv2.Canny(image=image, threshold1=50, threshold2=150)
    linux_display(image, "blurred")
    linux_display(canny, "edges")


if __name__ == '__main__':
    q9()
