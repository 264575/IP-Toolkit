# segimg.py
# license: see LICENSE file
import os
import numpy as np
from PIL import Image

import streamlit as st

import traceback
import cv2

from utils.fileinfo import setimages

# Load the pre-trained Haar Cascade model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_label_info(label_file_path):
    """
    Loading (Composing) Label Information
    """

    label_info = {}
    with open(label_file_path, 'r') as f:
        for line_number, line in enumerate(f, 1):
            parts = line.strip().split(', ')
            try:
                label_id = int(parts[0].replace('ID: ', ''))
                instance_id = int(parts[1].replace('Instance: ', ''))
                label_name = parts[2].replace('Label: ', '')
                color_info1 = parts[3].replace('Color: [', '')
                color_info2 = parts[4]
                color_info3 = parts[5].replace(']', '')
                color_info = [color_info1, color_info2, color_info3]
                color = tuple(int(color_info[index]) for index in range(3))

                if label_name in label_info:
                    label_info[label_name].append({'id': label_id, 'instance': instance_id, 'color': color})
                else:
                    label_info[label_name] = [{'id': label_id, 'instance': instance_id, 'color': color}]
            except ValueError as e:
                continue
    return label_info

def hex_to_rgb(hex_color):
    """
    Encoding Hexa-decimal Value to RGB Color Data
    """

    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

def extract_subdir_name(full_path, base_path):
    """
    Getting Sub-directory Name
    """

    relative_path = os.path.relpath(full_path, base_path)
    subdir_name = os.path.dirname(relative_path)
    return subdir_name

def get_bounding_box_from_mask(mask):
    """
    Returns the bounding box coordinates for the True region in the given mask.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def detect_face(img_array):
    # Convert the image to grayscale for Haar Cascades
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Detect face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return faces

def detect_head_and_crop(img: Image, output_path: str):
    # Convert PIL Image to numpy array
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Detect face in the image
    face_coordinates = detect_face(img_array)
    if len(face_coordinates) == 0:
        return None
    
    # Extract bounding box values
    x, y, w, h = face_coordinates[0]
    
    # Expand the bounding box
    longest_side = max(w, h)
    half_extension = int(0.5 * longest_side)
    
    x = max(x - half_extension, 0)
    y = max(y - half_extension, 0)
    
    # Crop the image using expanded bounding box
    cropped_image = img_array[y:y + 2 * longest_side, x:x + 2 * longest_side]
    
    # Save the cropped image
    cv2.imwrite(output_path, cropped_image)


def process_images(iter_no, batch_size, source_option, read_paths, output_directory=None):
    img_list, original_file_names, original_file_paths = setimages(iter_no, batch_size, source_option, read_paths)
    
    for img_array, file_name, file_path in zip(img_list, original_file_names, original_file_paths):
        img = Image.fromarray(img_array)
        
        output_file_name = file_name
        if output_directory:
            output_path = os.path.join(output_directory, output_file_name)
            check = detect_head_and_crop(img, output_path)
            if check is None:
                print(f"No face detected in the image: {file_name}.")

def segimg(iter_no, batch_size, source_option, read_paths, write_path, segmentation_map_paths, target_label_name, fill_color, rm, clip):
    """
    Segmenting Images
    """

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    fill_color = hex_to_rgb(fill_color)
    img_list, original_file_names, original_file_paths = setimages(iter_no, batch_size, source_option, read_paths)
    try:
        for file_no, subdir_paths in enumerate(original_file_paths):
            target_file = original_file_names[file_no]
            target_file_name = target_file.split('.')[0]

            original_img_array = img_list[file_no]
            segmap_file_path = segmentation_map_paths + '/' + target_file_name

            if not clip:

                label_info_file = os.path.join(segmap_file_path, f"{target_file_name}_labels_and_colors.txt")
                if not os.path.exists(label_info_file):
                    st.write(f"File not found: {label_info_file}")
                    continue

                label_info = load_label_info(label_info_file)
                if target_label_name not in label_info:
                    continue

                for instance_info in label_info[target_label_name]:
                    target_label_color = np.array(instance_info['color'], dtype=np.uint8)

                    target_label_id = instance_info['id']
                    instance_id = instance_info['instance']
                    target_file_name = str(target_label_id)+'_'+str(instance_id)+'.png'

                    original_segmap_file_path = segmap_file_path
                    original_segmap_file_name = target_file_name
                    seg_map_path = os.path.join(original_segmap_file_path, original_segmap_file_name)
                    if not os.path.exists(seg_map_path):
                        st.write(f"File not found: {seg_map_path}")
                        continue

                    seg_map = Image.open(seg_map_path)
                    seg_map_array = np.array(seg_map)[:, :, :3]

                    mask_info = np.all(seg_map_array == target_label_color, axis=-1)
                    if rm == 'Remove':
                        mask = mask_info
                    else:
                        mask = np.logical_not(mask_info)

                    filled_image = original_img_array.copy()
                    filled_image[mask] = fill_color
                    
                    if clip:
                        rmin, rmax, cmin, cmax = get_bounding_box_from_mask(mask)
                        image = filled_image[rmin:rmax+1, cmin:cmax+1]
                    else:
                        image = filled_image
                
                image = filled_image

                target_file_name = target_file
                output_path = os.path.join(write_path, os.path.basename(target_file_name))
                filled_image_img = Image.fromarray(image.astype('uint8'), 'RGB')
                filled_image_img.save(output_path)
    
            else:
                process_images(iter_no, batch_size, source_option, read_paths, write_path)
                

    except Exception as e:
        print(f"Error during segment removal and filling: {e}")
        traceback.print_exc()
