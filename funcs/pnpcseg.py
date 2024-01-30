# pnpcseg.py
# license: see LICENSE file
import io
import os
import time
import numpy as np
from PIL import Image

import torch
from transformers import DetrFeatureExtractor

from utils.fileinfo import setimages


feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")


def rgb_to_id(color):
    """
    Composing ID from RGB Color
    """

    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def pnpcseg(iter_no, batch_size, device, model, source_option, read_paths, write_path):
    """
    Panopic Segmentation
    """

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    img_list, original_file_names, original_file_paths = setimages(iter_no, batch_size, source_option, read_paths)

    if len(img_list)>0:
        inputs = feature_extractor(images=img_list, return_tensors="pt").to(device)

        outputs = []
        with torch.no_grad():
            outputs = model.forward(**inputs)

        processed_sizes = [torch.as_tensor([img.shape[0], img.shape[1]]) for img in img_list]
        results = feature_extractor.post_process_panoptic(outputs, processed_sizes)

        del outputs

        for idx, (result, file_name) in enumerate(zip(results, original_file_names)):

            name_without_extension = os.path.splitext(file_name)[0]
            new_dir_path = os.path.join(write_path, name_without_extension)

            if not os.path.exists(new_dir_path):
                os.makedirs(new_dir_path)

            panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
            panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)

            unique_labels = np.unique(panoptic_seg)
            color_map = {label: np.random.randint(0, 255, size=(3,)) for label in unique_labels}
            panoptic_seg_id = rgb_to_id(panoptic_seg)

            colorful_map = np.zeros((panoptic_seg.shape[0], panoptic_seg.shape[1], 3), dtype=np.uint8)
            labels_and_colors = []

            for (id, color), segment_info in zip(color_map.items(), result['segments_info']):
                category_id = segment_info['category_id']
                label = model.config.id2label[int(category_id)]
                instance_id = segment_info.get('id', None)

                mask = panoptic_seg_id == id

                for i in range(3):
                    colorful_map[mask, i] = color[i]

                labels_and_colors.append(f"ID: {category_id}, Instance: {instance_id}, Label: {label}, Color: {color.tolist()}")

                single_color_image = np.zeros((panoptic_seg.shape[0], panoptic_seg.shape[1], 3), dtype=np.uint8)
                single_color_image[mask] = color
                single_color_img = Image.fromarray(single_color_image, 'RGB')
                if instance_id is not None:
                    single_color_img.save(os.path.join(new_dir_path, f"{category_id}_{instance_id}.png"))
                else:
                    single_color_img.save(os.path.join(new_dir_path, f"{category_id}.png"))

            colorful_img = Image.fromarray(colorful_map, 'RGB')
            colorful_img.save(os.path.join(new_dir_path, "colorful_map.png"))

            with open(os.path.join(new_dir_path, f"{name_without_extension}_labels_and_colors.txt"), "w") as f:
                for item in labels_and_colors:
                    f.write(f"{item}\n")

        del inputs
        del results
    torch.cuda.empty_cache()
    time.sleep(2)
