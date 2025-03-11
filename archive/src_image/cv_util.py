import cv2
import numpy as np

def pad_to_size(image, target_size=(1920, 1080)):
    rows, cols, _ = image.shape
    target_cols, target_rows = target_size
    scale_factor = min(target_cols / cols, target_rows / rows)
    new_cols = int(cols * scale_factor)
    new_rows = int(rows * scale_factor)
    resized_image = cv2.resize(image, (new_cols, new_rows), interpolation=cv2.INTER_LINEAR)

    top = (target_rows - new_rows) // 2
    bottom = target_rows - new_rows - top
    left = (target_cols - new_cols) // 2
    right = target_cols - new_cols - left
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image

def apply_random_transformation(image, angle, tx, ty, scale):
    rows, cols, _ = image.shape

    # rotation
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))

    # translation
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(rotated_image, M, (cols, rows))

    # scaling
    scaled_image = cv2.resize(translated_image, None, fx=scale[0], fy=scale[1], interpolation=cv2.INTER_LINEAR)

    # consistency
    final_image = cv2.resize(scaled_image, (cols, rows), interpolation=cv2.INTER_LINEAR)

    return final_image

def preprocess_image(image, target_size=(1920, 1080), angle=0, tx=0, ty=0, scale=(1, 1), reflection=False):
    # left-right reflection
    if reflection:
        image = cv2.flip(image, 1)
    # sharpen the image to improve the quality
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel)
    # normalize the image
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    padded_image = pad_to_size(image, target_size)
    transformed_image = apply_random_transformation(padded_image, angle, tx, ty, scale)
    # final_image = ensure_in_bounds(transformed_image, target_size)
    # return final_image
    return transformed_image
