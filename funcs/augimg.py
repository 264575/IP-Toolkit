# augimg.py
# license: see LICENSE file
from PIL import Image, ImageChops
import os

from utils.fileinfo import setimages


def save_augimg(aug_images, write_path, aug_names):
    """
    Save Generated Files
    """

    for names, images in zip(aug_names, aug_images):
        for name, image in zip(names, images):
            augmented_img = image
            new_file_name = f"augmented_{name}"
            new_img_path = os.path.join(write_path, new_file_name)
            augmented_img.save(new_img_path)

def rotate(original_file_name, images, rotate_angle, rotate_range, rotate_angles_tic):
    """
    Rotating
    """

    if float(rotate_angles_tic) < 0.0:
        rotate_angles_tic = -rotate_angles_tic
    if float(rotate_range) < 0.0:
        rotate_range = -rotate_range

    cnt_range = int(2.0*rotate_range/rotate_angles_tic)

    angles = []
    for angle in range(cnt_range):
        rotate = float(rotate_angle)-float(rotate_range)+float(angle)*float(rotate_angles_tic)
        angles.append(rotate)

    rotated_names = []
    rotated_images = []
    for index, (name, img) in enumerate(zip(original_file_name, images)):
        images = []
        names = []
        for angle in angles:
            images.append(img.rotate(angle))
            if angle > 0:
                name = 'rotate_'+str(angle)+'_'+original_file_name[index]
            else:
                name = 'rotate_m'+str(abs(angle))+'_'+original_file_name[index]
                
            names.append(name)

        rotated_images.append(images)
        rotated_names.append(names)

    return rotated_names, rotated_images

def flip_lr(original_file_name, images):
    """
    Flipping Left-Right
    """

    flipped_names = []
    flipped_images = []
    for index, img in enumerate(images):
        flipped_images.append(img.transpose(Image.FLIP_LEFT_RIGHT))
        name = 'flip_lr_'+original_file_name[index]
        flipped_names.append(name)

    return [flipped_names], [flipped_images]

def flip_tb(original_file_name, images):
    """
    Flipping Top-Bottomn
    """

    flipped_names = []
    flipped_images = []
    for index, img in enumerate(images):
        flipped_images.append(img.transpose(Image.FLIP_TOP_BOTTOM))
        name = 'flip_tb_'+original_file_name[index]
        flipped_names.append(name)

    return [flipped_names], [flipped_images]

def slide_image(original_file_name, images, slide_range, tic):
    """
    Sliding (Shifting)
    - X&Y Directions
    """

    x_range = slide_range[0]
    y_range = slide_range[1]
    x_tic = tic[0]
    y_tic = tic[1]
    m_x_range = -x_range//x_tic
    m_y_range = -y_range//y_tic

    slided_names = []
    slided_images = []
    for index, img in enumerate(images):
        aug_images = []
        names = []
        
        shifted_x0_y0 = ImageChops.offset(img, 0, 0)
        name = 'slide_x'+str(0)+'_y'+str(0)+'_'+original_file_name[index]
        aug_images.append(shifted_x0_y0)
        names.append(name)
        
        if x_range >= x_tic:
            for x in range(x_tic, x_range+1, x_tic):
                shifted_px_y0 = ImageChops.offset(img, x, 0)
                name = 'slide_px'+str(x)+'_y'+str(0)+'_'+original_file_name[index]
                aug_images.append(shifted_px_y0)
                names.append(name)
                
                shifted_mx_y0 = ImageChops.offset(img, -x, 0)
                name = 'slide_mx'+str(x)+'_y'+str(0)+'_'+original_file_name[index]
                aug_images.append(shifted_mx_y0)
                names.append(name)

        if y_range >= y_tic:
            for y in range(y_tic, y_range+1, y_tic):
                shifted_x0_py = ImageChops.offset(img, 0, y)
                name = 'slide_x'+str(0)+'_py'+str(y)+'_'+original_file_name[index]
                aug_images.append(shifted_x0_py)
                names.append(name)
                
                shifted_x0_my = ImageChops.offset(img, 0, -y)
                name = 'slide_x'+str(0)+'_my'+str(y)+'_'+original_file_name[index]
                aug_images.append(shifted_x0_my)
                names.append(name)

        if x_range >= x_tic and y_range >= y_tic and x_range != 0 and y_range != 0:
            for x in range(x_tic, x_range+1, x_tic):
                for y in range(y_tic, y_range+1, y_tic):
                    shifted_px_py = ImageChops.offset(img, x, y)
                    name = 'slide_px'+str(x)+'_py'+str(y)+'_'+original_file_name[index]
                    aug_images.append(shifted_px_py)
                    names.append(name)
                    
                    shifted_mx_py = ImageChops.offset(img, -x, y)
                    name = 'slide_mx'+str(x)+'_py'+str(y)+'_'+original_file_name[index]
                    aug_images.append(shifted_mx_py)
                    names.append(name)
                    
                    shifted_px_my = ImageChops.offset(img, x, -y)
                    name = 'slide_px'+str(x)+'_my'+str(y)+'_'+original_file_name[index]
                    aug_images.append(shifted_px_my)
                    names.append(name)
                    
                    shifted_mx_my = ImageChops.offset(img, -x, -y)
                    name = 'slide_mx'+str(x)+'_my'+str(y)+'_'+original_file_name[index]
                    aug_images.append(shifted_mx_my)
                    names.append(name)

        slided_names.append(names)
        slided_images.append(aug_images)

    return slided_names, slided_images


def augimg(iter_no, batch_size, source_option, read_paths, write_path, aug_op, rotate_angle, rotate_range, rotate_angles_tic, slide_range, slide_tic):
    """
    Generate Similar Images
        - Flip horizontal
        - Flip Vertical
        - Rotate
        - Slide(Shift)
    """

    img_list, original_file_names, original_file_paths = setimages(iter_no, batch_size, source_option, read_paths)

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    if isinstance(read_paths, list):
        read_paths = read_paths
    elif isinstance(read_paths, str):
        read_paths = [os.path.join(read_paths, fname) for fname in os.listdir(read_paths)]
    else:
        raise ValueError('Unsupported input type.')

    images = []
    for index, read_path in enumerate(read_paths):
        images.append(Image.open(read_path))
        
    if aug_op == 'flip_lr':
        aug_names, aug_images = flip_lr(original_file_names, images)
    if aug_op == 'flip_tb':
        aug_names, aug_images = flip_tb(original_file_names, images)
    if aug_op == 'slide':
        aug_names, aug_images = slide_image(original_file_names, images, slide_range, slide_tic)
    if aug_op == 'rotate':
        aug_names, aug_images = rotate(original_file_names, images, rotate_angle, rotate_range, rotate_angles_tic)

    save_augimg(aug_images, write_path, aug_names)
