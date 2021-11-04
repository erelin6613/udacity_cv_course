import cv2
import numpy as np

from utils import read_image

img_path1 = "image_samples\\racoon.jpg"

def q13():

    borders = [cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_ISOLATED
            ]
    image = read_image(img_path1)
    cv2.imshow("racoon", image)
    cv2.waitKey(0)
    filt = cv2.getGaussianKernel(9,0)
    for bord in borders:
        img_filtered = cv2.sepFilter2D(image, 0, filt, filt, borderType=bord)
        cv2.imshow(f"racoon: border - {bord}", image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    q13()
