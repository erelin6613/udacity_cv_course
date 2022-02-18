# from scipy.spatial import distance
import numpy as np
import cv2

from utils import read_image, linux_display

path_left = "image_samples/stereo_left.png"
path_right = "image_samples/stereo_right.png"

patch_loc, patch_size = [240, 280], [100, 100]


def find_best_match(patch, strip):
    best_score, best_loc = np.inf, -1
    for i in range(strip.shape[1] - patch.shape[1]):
        crop = strip[:, i : i+patch.shape[0]]
        # score = distance.cdist(patch, crop, 'cosine').sum()
        score = np.sum((patch - crop)**2)
        if score < best_score:
            best_score = score
            best_loc = i
    return best_score, best_loc


def match_strips(strip_left, strip_right, b):
    n_blocks = int(strip_left.shape[1] / b)
    disp = np.zeros((1, n_blocks))

    for block in range(n_blocks):
        y_1 = block * b
        patch_left = strip_left[:, y_1:y_1 + b]
        # print(patch_left.shape, strip_right.shape, strip_left.shape)
        _, y = find_best_match(patch_left, strip_right)
        disp[0, block] = (block * b - y)

    return disp


def main3():
    img_left = read_image(path_left) / 255
    img_right = read_image(path_right) / 255

    patch_left = img_left[patch_loc[0]: patch_loc[0] + patch_size[0], patch_loc[1]: patch_loc[1] + patch_size[1]]
    strip_right = img_right[patch_loc[0]: patch_loc[0] + patch_size[0], :]

    linux_display(patch_left, "left patch")
    linux_display(strip_right, "right strip")

    corr, y = find_best_match(patch_left, strip_right)

    patch_right = strip_right[:, y: y + patch_size[1]]

    linux_display(patch_right, "match")


def main9():
    img_left = read_image(path_left) / 255
    img_right = read_image(path_right) / 255

    strip_left = img_left[patch_loc[0]: patch_loc[0] + patch_size[0], :]
    strip_right = img_right[patch_loc[0]: patch_loc[0] + patch_size[0], :]

    disparity = match_strips(strip_left, strip_right, 100)
    print(disparity)


if __name__ == '__main__':
    # main3()
    main9()
