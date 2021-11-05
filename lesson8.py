import os
from itertools import product
import cv2
import numpy as np

from utils import read_image, downscale_image, linux_display

img_path1 = os.path.join("image_samples", "browsers.jpg")


def hough_lines_finding():
    image = downscale_image(read_image(img_path1))
    linux_display(image, "image")

    canny = cv2.Canny(image=image, threshold1=50, threshold2=150)
    linux_display(canny, "edges")

    params = {
        "rho": [0.001, 0.1, 0.15],
        "theta_angle": [1, 25, 45],
        "threshold": [100, 150, 200]
        }
    combinations = product(params["rho"], params["theta_angle"], params["threshold"])
    for p in combinations:
        # print(params["rho"][p], params["theta_angle"][p]*np.pi/180, int(params["threshold"][p]))
        rho_, theta_, thresh_ = p
        lines = cv2.HoughLines(
            canny, rho=rho_,
            theta=theta_*np.pi/180,
            threshold=int(thresh_))
        if lines is None:
            print(f"No lines found with rho={rho_}, theta={theta_}, thresh={int(thresh_)}")
            continue
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a*rho, b*rho
                x1, y1 = int(x0 + 1000*(-b)), int(y0 + 1000*(a))
                x2, y2 = int(x0 - 1000*(-b)), int(y0 - 1000*(a))

                cv2.line(image, (x1, y1), (x2, y2), 0, 2)
        linux_display(image, f"rho={rho_}, theta={theta_}, thresh={int(thresh_)}")


if __name__ == '__main__':
    hough_lines_finding()
