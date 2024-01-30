# cnvtfmt.py
# license: see LICENSE file
import os
import shutil
import random
from PIL import Image

from utils.fileinfo import setimages
import streamlit as st


def randomize_filenames(directory, write_path):
    """
    Renaming file name with random numbwer
    - rename to "total number of files"_"randomnumber"
    """

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    total_files = len(files)

    used_numbers = set()
    for filename in files:
        while True:
            rand_num = random.randint(1, 10**len(str(total_files)))
            if rand_num not in used_numbers:
                used_numbers.add(rand_num)
                break

        new_name = f"{total_files}_{rand_num:0{len(str(total_files))}}{os.path.splitext(filename)[1]}"
        shutil.copy(os.path.join(directory, filename), os.path.join(write_path, new_name))

def prefix_filenames(directory, write_path, prefix):
    """
    Renaming file name with prefix and numbwer
    - rename to "prefix"_"number"
    """

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    for index, filename in enumerate(files, 1):
        new_name = f"{prefix}_{filename}"
        shutil.copy(os.path.join(directory, filename), os.path.join(write_path, new_name))

def remove_prefix(directory, write_path):
    """
    Removing prefix in file name
    """

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    orinal_name_list = []
    orinal_name_count = []
    for index, filename in enumerate(files, 1):

        new_name = filename.split('_')
        #original_new_name = new_name[len(new_name)-2]+'_'+new_name[len(new_name)-1]
        original_new_name = new_name[len(new_name)-1]
        if original_new_name in orinal_name_list:
            idx = orinal_name_list.index(original_new_name)
            orinal_name_count[idx] += 1
        else:
            orinal_name_list.append(original_new_name)
            orinal_name_count.append(1)

        idx = orinal_name_list.index(original_new_name)
        new_name = str(orinal_name_count[idx])+'_'+original_new_name

        shutil.copy(os.path.join(directory, filename), os.path.join(write_path, new_name))
        #st.write(f'filename:{original_new_name}')

def remove_postfix(directory, write_path):

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    orinal_name_list = []
    for index, filename in enumerate(files):
        if '_' in filename:
            new_name = filename.split('_')
            original_new_name = new_name[len(new_name)-2]+'.'+new_name[len(new_name)-1].split('.')[1]

            shutil.copy(os.path.join(directory, filename), os.path.join(write_path, original_new_name))
            #st.write(f'filename:{original_new_name}')
        else:
            #st.write(f'skip:{filename}')
            shutil.copy(os.path.join(directory, filename), os.path.join(write_path, filename))

def spawn_file(directory, write_path, prefix):
    """
    Copy file
    """

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    iter_num = int(prefix)
    for index, filename in enumerate(files, 1):

        for no in range(1, iter_num+1, 1):
            new_name = str(no)+'_'+filename

            shutil.copy(os.path.join(directory, filename), os.path.join(write_path, new_name))

def convert_format(iter_no, batch_size, source_option, read_paths, write_path, target_format):
    """
    Convert image format
    """

    img_list, original_file_names, original_file_paths = setimages(iter_no, batch_size, source_option, read_paths)

    for img_array, original_file_name in zip(img_list, original_file_names):
        new_filename = original_file_name.split('.')[0] + '.' + target_format
        new_file_path = os.path.join(write_path, new_filename)

        if os.path.exists(new_file_path):
            continue

        try:
            img = Image.fromarray(img_array)
            img.save(new_file_path, format=target_format.upper())
        except Exception as e:
            print(f"An error occurred while processing {new_filename}: {e}")

def cnvtfmt(iter_no, batch_size, source_option, read_paths, write_path, op_sel, target_format, prefix):
    """
    Converting Image Format
    """

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    if op_sel == 'Convert':
        convert_format(iter_no, batch_size, source_option, read_paths, write_path, target_format)

    elif op_sel == 'Random' and isinstance(read_paths, str):
        randomize_filenames(read_paths, write_path)

    elif op_sel == 'Prefix' and isinstance(read_paths, str):
        prefix_filenames(read_paths, write_path, prefix)

    elif op_sel == 'Remove' and isinstance(read_paths, str):
        remove_prefix(read_paths, write_path)
        #remove_postfix(read_paths, write_path)

    elif op_sel == 'Spawn' and isinstance(read_paths, str):
        spawn_file(read_paths, write_path, prefix)
