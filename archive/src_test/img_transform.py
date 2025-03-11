import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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

def apply_random_transformation(image):
    rows, cols, _ = image.shape

    # Random rotation
    angle = np.random.uniform(-10, 10)
    # angle = 0
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))

    # Random translation
    tx = np.random.uniform(-100, 100) # we dont need too much translation, 100 is subtle enough
    ty = np.random.uniform(-60, 60)
    # tx = 0
    # ty = 0
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(rotated_image, M, (cols, rows))

    # Random scaling
    scale = np.random.uniform(0.8, 1.2)
    # scale = 1
    scaled_image = cv2.resize(translated_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # Ensure the image size is consistent
    final_image = cv2.resize(scaled_image, (cols, rows), interpolation=cv2.INTER_LINEAR)

    return final_image

if __name__ == "__main__":
    test_image = cv2.imread("dataset/tvb-hksl-news/frames/2020-01-16/000453-000550/000453.jpg")
    output_dir = "src_test/img_test"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(10):
        # Pad the image to 1920x1080
        padded_image = pad_to_size(test_image)

        # Apply a random transformation to the image while keeping the keypoints in the same position
        transformed_image = apply_random_transformation(padded_image)

        # Ensure all non-pure black pixels are in bounds of 1920x1080
        # final_image = ensure_in_bounds(transformed_image)
        final_image = transformed_image

        # Save the result image
        output_path = os.path.join(output_dir, f"transformed_image_{i+1}.png")
        cv2.imwrite(output_path, final_image)

        # Optionally, display the result image
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Transformed Image")
        plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.savefig(os.path.join(output_dir, f"comparison_{i+1}.png"))
        plt.close()