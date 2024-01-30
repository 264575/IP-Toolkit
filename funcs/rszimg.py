# rszimg.py
# license: see LICENSE file
import os
import numpy as np
from PIL import Image
import cv2

def resize_image_cv2(image, width, height, maintain_aspect_ratio):
    """
    Upscaling with OpenCV
    """

    original_width, original_height = image.size

    if maintain_aspect_ratio:
        aspect_ratio = original_width / original_height
        new_aspect_ratio = width / height

        if new_aspect_ratio > aspect_ratio:
            new_width = int(height * aspect_ratio)
            new_height = height
        else:
            new_width = width
            new_height = int(width / aspect_ratio)
    else:
        new_width = width
        new_height = height

    # Convert PIL to OpenCV format
    image_np = np.array(image)
    resized_image_np = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Convert back to PIL format
    resized_image = Image.fromarray(resized_image_np)

    return resized_image

def resize_image(image, width, height, maintain_aspect_ratio, use_cv):
    """
    Resizing Images
    - uses_cv == True then upscaling
    """

    if use_cv:
        return resize_image_cv2(image, width, height, maintain_aspect_ratio)

    if maintain_aspect_ratio:
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height
        new_aspect_ratio = width / height

        if new_aspect_ratio > aspect_ratio:
            new_width = int(height * aspect_ratio)
            new_height = height
        else:
            new_width = width
            new_height = int(width / aspect_ratio)

        resized_image = image.resize((new_width, new_height))
    else:
        resized_image = image.resize((width, height))

    return resized_image


def rszimg(read_paths, write_path, height, width, maintain_aspect_ratio=True, use_cv=False):
    """
    Resize Handler
    """

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    if isinstance(read_paths, list):
        read_paths = read_paths
    elif isinstance(read_paths, str):
        read_paths = [os.path.join(read_paths, fname) for fname in os.listdir(read_paths)]
    else:
        raise ValueError('Unsupported input type.')

    for idx, image in enumerate(read_paths):

        if isinstance(image, str):
            r_dir_name, r_file_name = os.path.split(image)
            with Image.open(image) as img:
                resized_images = resize_image(img, width, height, maintain_aspect_ratio, use_cv)
        else:
            r_dir_name, r_file_name = os.path.split(image.name)
            with Image.open(image) as img:
                resized_images = resize_image(img, width, height, maintain_aspect_ratio, use_cv)

        ext = r_file_name.split('.')[-1]
        w_file_name = r_file_name.split('.')[0]

        img_path = os.path.join(write_path, f"{w_file_name}.{ext}")
        resized_images.save(img_path)

    #print(f"Resized images saved to {write_path}")
