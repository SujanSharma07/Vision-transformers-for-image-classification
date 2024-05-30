import os

import cv2
import numpy as np
from imgaug import augmenters as iaa
from skimage import io
from tqdm import tqdm


def augment_image(image):
    # Define augmentation pipeline
    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5),  # Horizontal flips
            iaa.Flipud(0.5),  # Vertical flips
            iaa.Affine(rotate=(-30, 30)),  # Rotation
            iaa.GaussianBlur(sigma=(0, 1.0)),  # Gaussian blur
        ],
        random_order=True,
    )

    # Apply augmentation
    augmented_image = seq(image=image)
    return augmented_image


def augment_images_in_directory(directory, target_count):
    # Get list of image files in the directory
    image_files = [f for f in os.listdir(directory) if f.endswith(".jpg")]

    # Calculate the number of images to create
    existing_count = len(image_files)
    images_to_create = target_count - existing_count

    if images_to_create <= 0:
        print("No additional images needed.")
        return

    print(f"Creating {images_to_create} additional images...")

    # Create augmented images
    for i in tqdm(range(images_to_create)):
        # Choose a random existing image
        random_image_name = np.random.choice(image_files)
        image_path = os.path.join(directory, random_image_name)

        # Read the image
        image = io.imread(image_path)

        # Augment the image
        augmented_image = augment_image(image)

        # Save the augmented image
        new_image_name = f"augmented_{i + existing_count}.jpg"
        new_image_path = os.path.join(directory, new_image_name)
        cv2.imwrite(new_image_path, augmented_image)

    print("Augmentation complete.")


# Example usage:


# CLASSES = ["cardboard", "glass", "ied", "metal", "paper", "plastic", "trash"]
CLASSES = ["ied", "nonied"]

target_count = 76  # Set the desired number of files

for i in CLASSES:
    directory_path = f"/home/sujan/Downloads/Categorical_to_binary/val/{i}"
    augment_images_in_directory(directory_path, target_count)
