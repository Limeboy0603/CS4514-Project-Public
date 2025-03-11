import cv2
import numpy as np

def pad_to_size(image, target_size=(1920, 1080)):
    rows, cols, _ = image.shape
    target_cols, target_rows = target_size

    # Calculate the scaling factor while keeping the aspect ratio
    scale_factor = min(target_cols / cols, target_rows / rows)
    new_cols = int(cols * scale_factor)
    new_rows = int(rows * scale_factor)

    # Resize the image
    resized_image = cv2.resize(image, (new_cols, new_rows), interpolation=cv2.INTER_LINEAR)

    # Calculate padding
    top = (target_rows - new_rows) // 2
    bottom = target_rows - new_rows - top
    left = (target_cols - new_cols) // 2
    right = target_cols - new_cols - left

    # Pad the image with black pixels
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def apply_random_transformation(image, angle, tx, ty, scale):
    rows, cols, _ = image.shape

    # Random rotation
    # angle = np.random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))

    # Random translation
    # tx = np.random.uniform(-50, 50)
    # ty = np.random.uniform(-50, 50)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(rotated_image, M, (cols, rows))

    # Random scaling
    # scale = np.random.uniform(0.8, 1.2)
    scaled_image = cv2.resize(translated_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # Ensure the image size is consistent
    final_image = cv2.resize(scaled_image, (cols, rows), interpolation=cv2.INTER_LINEAR)

    return final_image

def ensure_in_bounds(image, target_size=(1920, 1080)):
    rows, cols, _ = image.shape
    target_cols, target_rows = target_size

    # Create a black canvas of the target size
    canvas = np.zeros((target_rows, target_cols, 3), dtype=np.uint8)

    # Find the bounding box of non-black pixels
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])

    # Calculate the position to place the image on the canvas
    top = (target_rows - h) // 2
    left = (target_cols - w) // 2

    # Place the image on the canvas
    canvas[top:top+h, left:left+w] = image[y:y+h, x:x+w]

    return canvas