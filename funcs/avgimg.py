# avgimg.py
# license: see LICENSE file
import os
import time
import numpy as np
from PIL import Image
import torch

from utils.fileinfo import setimages


def compute_average_image(device, img_list, iter_no):
    """
    Making Average Image by PyTorch
    """

    if iter_no>0:
        img = Image.open('./workspace/params_avgimg_o.png')
        img_array = np.array(img)
        img_list.insert(0, img_array)

    img_tensor = torch.tensor(img_list, dtype=torch.float32).to(device)
    average_image = torch.mean(img_tensor, dim=0).to("cpu").numpy().astype(np.uint8)

    torch.cuda.empty_cache()
    time.sleep(2)

    return Image.fromarray(average_image)

def avgimg(iter_no, batch_size, device, source_option, read_paths, write_path=None):
    """
    Handling Averaging Images
    """

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    img_list, filename_list, original_file_paths = setimages(iter_no, batch_size, source_option, read_paths)
    average_image = compute_average_image(device, img_list, iter_no)
    r_dir_name, r_file_name = os.path.split(filename_list[0])
    ext = r_file_name.split('.')[-1]

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    save_path = os.path.join(write_path, original_file_paths[0].split('/')[-1]+'average_image.'+ext)
    average_image.save(save_path)

    save_path = os.path.join('./workspace/params_avgimg_o.png')
    average_image.save(save_path)

    print(f"Average image saved to: {save_path}")
