import numpy as np
import cv2

from utils import read_image, linux_display


def downsample(img):
    # print(img.shape)
    y_idx = [i for i in range(img.shape[0]) if i % 2 != 0]
    x_idx = [i for i in range(img.shape[1]) if i % 2 != 0]

    rescaled = np.delete(img, y_idx, axis=0)
    rescaled = np.delete(rescaled, x_idx, axis=1)
    return rescaled


def blur(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def blur_downsample(img):
    img = blur(img)
    return downsample(img)


def main():
    img = cv2.imread("image_samples/neutron_star.jpg")
    downsampled = downsample(img)
    blur_n_down = blur_downsample(img)
    linux_display(downsampled, 'downsampled')
    linux_display(blur_n_down, 'blur_n_down')


if __name__ == '__main__':
    main()
