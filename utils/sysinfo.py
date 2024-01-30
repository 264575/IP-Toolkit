## sysinfo.py
#   license: see LICENSE file
import os
import streamlit as st
import time
import torch


def calc_total_size(file_path):
    """
    Calculate Total Data Size
    """

    total_data_size = 0
    if os.path.isdir(file_path):
        file_paths = [os.path.join(file_path, filename) for filename in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, filename))]
        for file_path in file_paths:
            total_data_size += os.path.getsize(file_path)
    elif os.path.isfile(file_path):
        total_data_size += os.path.getsize(file_path)

    return total_data_size

def get_total_data_size(restored, read_paths):
    """
    Calculate Total Data Size under Directory
    """

    total_data_size = 0
    st_data_loader = False
    for file_path in read_paths:
        if restored and isinstance(file_path, list):
            for path in file_path:
                total_data_size += os.path.getsize(path)
        elif isinstance(file_path, str):
            total_data_size = calc_total_size(file_path)
        else:
            total_data_size += file_path.size
            file_path.seek(0)
            st_data_loader = True
    return st_data_loader, total_data_size

def calculate_epoch_size(available_memory, data_size):
    """
    Calculate Epoch Size
    """

    if data_size > 0:
        epoch_size = data_size // available_memory + 1
    else:
        epoch_size = 1

    return epoch_size

def count_files(read_paths):
    files = os.listdir(read_paths)
    file_count = sum(os.path.isfile(os.path.join(read_paths, f)) for f in files)
    num_files = file_count

    return num_files

# Calculate Exec Environment
def calc_env(restored, device, total_memory, memory_size, read_paths, coef, weight=1):
    """
    Calculate Epoch and Batch Sizes
    """

    if isinstance(read_paths, list):
        if len(read_paths)>0 and isinstance(read_paths[0], str):
            num_files = count_files(read_paths[0])
        else:
            num_files = len(read_paths)
    else:
        num_files = count_files(read_paths)

    reserved_memory = torch.cuda.memory_reserved(device)
    allocated_memory = torch.cuda.memory_allocated(device)
    cached_memory= torch.cuda.memory_reserved(device)
    used_memory = reserved_memory+allocated_memory+cached_memory
    offset = 0
    if weight>1:
        #Care of Transformer
        offset = 12*1024*1024*1024
    else:
        offset = 0

    if total_memory>used_memory:
        available_memory = total_memory-used_memory-offset
    else:
        available_memory = 0.1

    available_memory = min(available_memory, memory_size)

    if isinstance(read_paths, str):
        st_data_loder, total_data_size = get_total_data_size(restored, [read_paths])
    else:
        st_data_loder, total_data_size = get_total_data_size(restored, read_paths)

    if num_files > 0:
        average_data_size = (total_data_size+num_files) // num_files
    else:
        average_data_size = 0

    if st_data_loder:
        # st.data_loader allocate 200MB for every file
        total_data_size = total_data_size*(200 + 40*num_files)
    elif weight > 1:
        total_data_size = total_data_size + ((average_data_size*average_data_size*num_files)//int(coef))

    epoch_size = calculate_epoch_size(available_memory, total_data_size)
    

    if total_data_size > 0 and epoch_size > 1:
        batch_size = (num_files + epoch_size) // epoch_size
    elif total_data_size > 0 and epoch_size > 0:
        batch_size = num_files
    else:
        batch_size = 0

    return epoch_size, batch_size

# Batch Processing
def batch_process(epoch_size, batch_size, input_files, task_function):
    """
    Batch Processor
    """

    progress_bar = st.progress(0)
    batch_text = st.empty()
    num_files = len(input_files)

    for i in range(epoch_size):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_files)
        batch_files = input_files[start_idx:end_idx]

        epoch_num = i + 1
        batch_text.text(f"Epoch: {epoch_num}/{epoch_size}")

        progress_bar.progress((epoch_num / epoch_size))

        task_function(i, batch_files)

        torch.cuda.empty_cache()
        time.sleep(8)

    st.write('finish the work.')
