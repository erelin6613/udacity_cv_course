import os
import cv2
import numpy as np

from utils import read_image, linux_display

img_path1 = os.path.join("image_samples", "hexagon.jpg")


def select_gdir(grad_magnitude, grad_direction, ang_low, ang_high):
    result = np.where(
        (grad_direction >= ang_low) & (grad_direction < ang_high), grad_magnitude, 0)
    return result


def q14():
    image = read_image(img_path1)
    image = image[:500, :700]
    linux_display(image, "image")
    ygrad = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=9)
    xgrad = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=9)
    grad_magnitude, grad_dir = np.gradient(cv2.bitwise_or(ygrad, xgrad))
    alt_magnitude = grad_magnitude / (4*np.sqrt(grad_magnitude))
    alt_direction = (grad_dir + 180) / 360
    img_grad_35_55 = select_gdir(alt_magnitude, alt_direction, 35, 55)
    linux_display(img_grad_35_55, "grads_35-55_degrees")


if __name__ == '__main__':
    q14()
