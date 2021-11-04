import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import read_image, resize_images

quasar = "image_samples\\quasar.png"
oak = "image_samples\\oak.jpg"


def q13():
    image = read_image(oak)
    cv2.imshow("oak", image)
    cv2.waitKey(0)
    print(image.shape)
    print(image.dtype)


def q17():
    image = read_image(quasar)
    cv2.imshow("quasar", image)
    cv2.waitKey(0)
    print(image.shape)
    print(image[101:103, 201:203])
    # to reproduce ocvtave results
    print()
    print(image[101:104, 201:204])


def q19():
    image = read_image(quasar)
    cropped = image[110:310, 10:160]
    print(cropped.shape)
    # to reproduce ocvtave results
    cropped = image[110:311, 10:161]
    print()
    print(cropped.shape)


def q22():
    image = read_image(quasar, color_space="bgr")
    cv2.imshow("quasar", image)
    cv2.waitKey(0)
    print(image.shape)
    blue_channel = image[:, :, 0]
    print(blue_channel.shape)
    plt.plot(blue_channel[23, :])
    plt.show()


def blend_images(img1, img2, alpha):
    img1, img2 = resize_images([img1, img2])
    result = img1 * alpha + (1 - alpha) * img2
    return result.astype(np.uint8)


def q26():
    image1 = read_image(quasar)
    image2 = read_image(oak)
    result = blend_images(image1, image2, 0.75)
    cv2.imshow("quasar", result)
    cv2.waitKey(0)


def q31():
    noise = np.random.normal(size=1000)
    values, bins = np.histogram(noise, bins=np.linspace(-3, 3, 100))
    sns.distplot(noise)
    print(values[:5], bins[:5])
    # plt.hist(values, 50)
    plt.show()


if __name__ == "__main__":
    # q13()
    # q17()
    # q19()
    # q22()
    # q26()
    q31()
