import cv2
import numpy as np

from utils import read_image, downscale_image

img_path1 = "image_samples\\browsers.jpg"

def hough_lines_finding():
    image = downscale_image(read_image(img_path1))
    cv2.imshow("image", image)
    cv2.waitKey(0)

    # image = np.where(image>70, 255, 0).astype(np.uint8)
    canny = cv2.Canny(image=image, threshold1=50, threshold2=150)
    cv2.imshow("lines", canny)
    cv2.waitKey(0)
    lines = cv2.HoughLines(
        canny, rho=1, theta=1*np.pi/180 ,threshold=50)
    print(len(lines))
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            # acosx+bsiny
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a*rho, b*rho
            x1, y1 = int(x0 + 10*(-b)), int(y0 + 10*(a))
            x2, y2 = int(x0 - 10*(-b)), int(y0 - 10*(a))

            cv2.line(image, (x1,y1), (x2,y2), 0, 2)
    cv2.imshow("lines", image)
    cv2.waitKey(0)

if __name__ == '__main__':
    hough_lines_finding()
