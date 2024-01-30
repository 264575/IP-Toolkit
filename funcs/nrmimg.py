# nrmimg.py
# license: see LICENSE file
import os
import time
import torch

from utils.fileinfo import setimages


def nrmimg(iter_no, batch_size, device, source_option, read_paths, write_path):
    """
    Normalizing Images
    """

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    try:
        img_list, original_file_names, original_file_paths = setimages(iter_no, batch_size, source_option, read_paths)

        img_tensor = torch.tensor(img_list, dtype=torch.float32).to(device)
        normalized_image = (img_tensor - img_tensor.mean()) / img_tensor.std()
        normalized_image = normalized_image.clamp(-1.0, 1.0)

        for index, original_file_name in enumerate(original_file_names):
            w_file_name = original_file_name.split('.')[0] + '.pt'
            save_path = os.path.join(write_path, w_file_name)
            torch.save(normalized_image[index], save_path)

        torch.cuda.empty_cache()
        time.sleep(2)

    except Exception as e:
        print(f"An error occurred while processing: {e}")
