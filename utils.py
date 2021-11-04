import cv2
import numpy as np

def read_image(img_path, color_space="gray"):
    image = cv2.imread(img_path)
    if color_space == "gray":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

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

def downscale_image(image):
    target_dim = image.shape[1]//2, image.shape[0]//2
    resized_image = cv2.resize(image, target_dim, interpolation=cv2.INTER_AREA)
    return resized_image
