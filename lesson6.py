import cv2
import numpy as np
from scipy.signal import correlate

from utils import read_image, downscale_image

img_path1 = "image_samples\\hexagon.jpg"

def select_gdir(grad_magnitude, grad_direction, ang_low, ang_high):
    result = np.where((grad_direction >= ang_low)&(grad_direction < ang_high), grad_magnitude, 0)
    return result
    # pass

def q14():
    # image = downscale_image(read_image(img_path1))/255
    image = read_image(img_path1)
    image = image[:500, :700]
    cv2.imshow("image", image)
    cv2.waitKey(0)
    ygrad = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=9)
    xgrad = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=9)
    grad_magnitude, grad_dir = np.gradient(cv2.bitwise_or(ygrad, xgrad))
    alt_magnitude = grad_magnitude / (4*np.sqrt(grad_magnitude))
    alt_direction = (grad_dir + 180) / 360
    img_grad_35_55 = select_gdir(alt_magnitude, alt_direction, 35, 55)
    cv2.imshow("grads", img_grad_35_55)
    cv2.waitKey(0)


"""% Gradient Direction
function result = select_gdir(gmag, gdir, mag_min, angle_low, angle_high)
    % TODO Find and return pixels that fall within the desired mag, angle range
endfunction

pkg load image;

%% Load and convert image to double type, range [0, 1] for convenience
img = double(imread('octagon.png')) / 255.;
imshow(img); % assumes [0, 1] range for double images

%% Compute x, y gradients
[gx gy] = imgradientxy(img, 'sobel'); % Note: gx, gy are not normalized

%% Obtain gradient magnitude and direction
[gmag gdir] = imgradient(gx, gy);
imshow(gmag / (4 * sqrt(2))); % mag = sqrt(gx^2 + gy^2), so [0, (4 * sqrt(2))]
imshow((gdir + 180.0) / 360.0); % angle in degrees [-180, 180]

%% Find pixels with desired gradient direction
my_grad = select_gdir(gmag, gdir, 1, 30, 60); % 45 +/- 15
%imshow(my_grad);  % NOTE: enable after you've implemented select_gdir
"""
if __name__ == '__main__':
    q14()
