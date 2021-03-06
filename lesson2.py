import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

quasar = os.path.join("image_samples", "quasar.png")
oak = os.path.join("image_samples", "oak.jpg")


def read_image(img_path, color_space="gray"):
    image = cv2.imread(img_path)
    if color_space == "gray":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


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


def resize_images(images):
    largest_size = [x.shape[0] * x.shape[1] for x in images]
    largest_id = np.argmax(largest_size)
    target_dim = (images[largest_id].shape[1], images[largest_id].shape[0])

    resized = []
    for i, image in enumerate(images):
        if i == largest_id:
            resized_image = image
        else:
            resized_image = cv2.resize(image, target_dim, interpolation=cv2.INTER_AREA)
        resized.append(resized_image)

    return resized


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


# % L2 Q31
# % Generate Gaussian noise
# noise = randn([1 1000]);
# [n x] = hist(noise, linspace(-3, 3, 21));
# %disp([x; n]);
# plot(x, n);
#
# % TODO: Try generating other kinds of random numbers.
# %       How about a 2D grid of random Gaussian values?
#
# more_noise = rand([1 100]);
# [n x] = hist(more_noise, linspace(-100, 100, 30));
# %disp([x; n]);
# plot(x, n);
#
# more_noise = randi([1 10000]);
# [n x] = hist(more_noise, linspace(-100, 200, 30));
# %disp([x; n]);
# plot(x, n);


if __name__ == "__main__":
    # q13()
    # q17()
    # q19()
    # q22()
    # q26()
    q31()
