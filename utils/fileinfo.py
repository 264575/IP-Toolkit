## fileinfo.py
#   license: see LICENSE file
import os
import io
from typing import List, Dict
from PIL import Image
import numpy as np
import json


def check_subdirectories(directory_path):
    """
    Check Sub-Directory
    """

    all_files = os.listdir(directory_path)
    for filename in all_files:
        full_path = os.path.join(directory_path, filename)
        if os.path.isdir(full_path):
            return False
    return True

def get_directory_and_file_info(directory_path: str) -> Dict[str, List[str]]:
    """
    Get Information of Directory and File Name
    """

    dirs = []
    files = []

    try:
        with os.scandir(directory_path) as it:
            for entry in it:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        dirs.append(entry.name)
                    elif entry.is_file(follow_symlinks=False):
                        files.append(entry.name)
                except PermissionError:
                    continue

        return dirs, files

    except Exception as e:
        return {"error": f"An error occurred: {e}"}

def read_images_from_file_list(start_no, end_no, read_paths):
    """
    Read Images with Image File List
    """

    sorted_original_dir_paths = []
    sorted_original_file_names = []
    img_list = []
    for item_no, item in enumerate(read_paths):
        if item_no >= start_no and item_no <= end_no:
            img_data = item.read()
            try:
                sorted_original_dir_paths.append(os.path.dirname(item.name))
                sorted_original_file_names.append(os.path.basename(item.name))
                img = Image.open(io.BytesIO(img_data))
                img_array = np.array(img)
                img_list.append(img_array[:, :, :3])
                item.seek(0)
            except Exception as e:
                print(f"An error occurred: {e}")

    return img_list, sorted_original_dir_paths, sorted_original_file_names

def setimages(iter_no, batch_size, source_option, read_paths):
    """
    Set Image List
    """

    img_list = []
    original_file_names = {}
    original_file_paths = {}
    sorted_original_file_names = []
    sorted_original_dir_paths = []
    image_extensions = ['.jpeg', '.png', '.gif', '.bmp', '.tiff']

    start_no = iter_no*batch_size
    end_no = (iter_no+1)*batch_size-1

    def process_directories(count, start_no, end_no, directory, base_directory):
        nonlocal img_list, original_file_names, original_file_paths

        if not directory.startswith(base_directory):
            return count

        dirs, _ = get_directory_and_file_info(directory)

        if isinstance(dirs, dict) and 'error' in dirs:
            print(dirs['error'])
            return count

        for entry in dirs:
            full_path = os.path.join(directory, entry)

            if os.path.isdir(full_path):
                count = process_directories(count, start_no, end_no, full_path, base_directory)
            elif os.path.isfile(full_path) and count >= start_no and count <= end_no:
                count += 1
                file_extension = os.path.splitext(full_path)[1].lower()
                if file_extension in image_extensions:
                    try:
                        img = Image.open(full_path)
                        img_array = np.array(img)
                        subdir_name = os.path.basename(os.path.dirname(full_path))

                        if subdir_name not in original_file_names:
                            original_file_names[subdir_name] = []
                        if subdir_name not in original_file_paths:
                            original_file_paths[subdir_name] = []

                        original_file_names[subdir_name].append(os.path.basename(full_path))
                        original_file_paths[subdir_name].append(directory)

                        img_list.append(img_array[:, :, :3])
                    except Exception as e:
                        print(f"An error occurred while opening {full_path}: {e}")

        return count


    def process_directory(start_no, end_no, read_paths):
        nonlocal img_list, sorted_original_file_names, sorted_original_dir_paths

        dirs, file_list = get_directory_and_file_info(read_paths)

        for entry in file_list[start_no:end_no+1]:
                full_path = os.path.join(read_paths, entry)
                try:
                    img = Image.open(full_path)
                    img_array = np.array(img)
                    sorted_original_file_names.append(entry)
                    sorted_original_dir_paths.append(read_paths)
                    img_list.append(img_array[:, :, :3])
                except Exception as e:
                    print(f"An error occurred while opening {full_path}: {e}")


    which_func = False
    if source_option == 'Upload Files Reconstructed':
        img_list, sorted_original_dir_paths, sorted_original_file_names = read_images_from_file_list(start_no, end_no, read_paths)
    else:
        if isinstance(read_paths, str):
            if os.path.isdir(read_paths) and not check_subdirectories(read_paths):
                count = process_directories(0, start_no, end_no, read_paths, read_paths)
                which_func = False
            elif check_subdirectories(read_paths):
                process_directory(start_no, end_no, read_paths)
                which_func = True

            if not which_func:
                for subdir, names in original_file_names.items():
                    paths = original_file_paths[subdir]
                    aligned_names, aligned_paths = zip(*sorted(zip(names, paths)))
                    sorted_original_dir_paths.append(list(aligned_paths))
                    sorted_original_file_names.append(list(aligned_names))

        else:
           img_list, sorted_original_dir_paths, sorted_original_file_names = read_images_from_file_list(start_no, end_no, read_paths)

    return img_list, sorted_original_file_names, sorted_original_dir_paths

def save_settings(filename, source_option, read_paths, write_path, func_info):
    """
    Save Setting of Session
    """

    list_read_paths = []
    if isinstance(read_paths, list):
        for path in read_paths:
            list_read_paths.append(path)
    else:
        list_read_paths = read_paths

    settings = {
        'source_option': source_option,
        'read_paths': list_read_paths,
        'write_path': write_path,
        'func_info': func_info
    }
    with open(filename, 'w') as f:
        json.dump(settings, f)

def load_settings(filename):
    """
    Load Setting of Session
    """

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            settings = json.load(f)
            return settings.get('source_option', ''), settings.get('read_paths', ''), settings.get('write_path', ''), settings.get('func_info', '')
    else:
        return [], ''

def extract_file_info(read_paths, source_option):
    if source_option == 'Upload Files':
        directory = os.path.dirname(read_paths[0].name)
    else:
        directory = read_paths
    return {'directory': directory}

def reconstruct_file_info(directory):
    return directory

def read_parameters(file_path):
    """
    Read Parameters
        - Read config.txt having init val for write_path
    """

    parameters = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                key, value = line.strip().split('=')
                key = key.replace(' ', '')
                value = value.replace(' ', '')
                parameters[key] = value
    except FileNotFoundError:
        print('File not found.')
    except Exception as e:
        print(f"An error occurred: {e}")
    return parameters
