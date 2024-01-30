## iptoolkit.py
#   license: see LICENSE file
import os
import time
import json
import streamlit as st
import torch
from transformers import DetrForSegmentation

## Image Preprocessor's Functions
from utils.fileinfo import read_parameters, save_settings, load_settings
from utils.sysinfo import batch_process, calc_env
from funcs.cnvtfmt import cnvtfmt
from funcs.pnpcseg import pnpcseg
from funcs.shrpimg import shrpimg
#from funcs.hmnprs import hmnprs
from funcs.segimg import segimg
from funcs.rszimg import rszimg
from funcs.augimg import augimg
from funcs.augtxt import augtxt
from funcs.nrmimg import nrmimg
from funcs.avgimg import avgimg


## DETR Model
#   Detect GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#   Create Model Object
pnpc_model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-50-panoptic')
#hmnprs_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)


## Streamlit layout
st.title('Image Preprocessing Toolkit')

#   Side-Bar Configuration
st.sidebar.header('Settings')

if torch.cuda.is_available():
    st.sidebar.write('GPU is detected')
else:
    st.sidebar.write('CPU is detected')

#   Get Memory Info for device
total_memory = torch.cuda.get_device_properties(device).total_memory
allocated = torch.cuda.memory_allocated(device)
reserved_memory = torch.cuda.memory_reserved(device)
cached = torch.cuda.memory_reserved(device)
if total_memory>(reserved_memory+allocated+cached):
    available_memory = total_memory-(reserved_memory+allocated+cached)
else:
    # assign adjusting value
    available_memory = 0.1

#   Display Memory Status
st.sidebar.markdown(f"""
- Total Memory: {int(total_memory/(1024*1024))} MiB
- PyTorch Memory: {int(reserved_memory/(1024*1024))} MiB
- Allocated Memory: {int(allocated/(1024*1024))} MiB
- Cached Memory: {int(cached/(1024*1024))} MiB
- Available Memory: {int(available_memory/(1024*1024))} MiB
""")


if available_memory == 0:
    available_memory = 1.0


## Setting Memory Size for Use
memory_size = int(st.sidebar.slider('Use of Memory Size (in GiB):', min_value=0.5, max_value=float(available_memory/(1024*1024*1024)), value=1.0, step=0.25)*1024*1024*1024)

coef = st.sidebar.slider('Adjust Coefficiency:', min_value=1, max_value=64, value=1, step=1)


#   Update Memory Status
mem_update = st.sidebar.button("Update")
if mem_update:
    torch.cuda.empty_cache()
    time.sleep(1)
    mem_update = False


## Main Part for Functions
#   Batch-Control Map Table
function_map = {
    "Convert Files": cnvtfmt,
    "Picture Parse": pnpcseg,
    "Segment": segimg,
    #"Human Parse": hmnprs,
    "Sharpen Images": shrpimg,
    "Resize Images": rszimg,
    "Augment Images": augimg,
    "Augment Texts": augtxt,
    "Generate Normalized Images": nrmimg,
    "Generate Average Image": avgimg,
}

st.sidebar.header("Select a task")


## Tab on Left-side
tab_selection = st.sidebar.radio("task:", [
                        "Picture Parse",
                        #"Human Parse",
                        "Segment",
                        "Sharpen Images",
                        "Resize Images",
                        "Augment Images",
                        "Augment Texts",
                        "Generate Normalized Images",
                        "Generate Average Image",
                        "Convert Files",
                        "End-to-End Run"
                    ])

## Read Configuration File
#   config.txt file define initial read&write paths
parameters = read_parameters("./config.txt")


## Init
read_paths = None
start_run = False

if 'settings_cnvtfmt' not in st.session_state:
    st.session_state['settings_cnvtfmt'] = {
        'source_option': None,
        'cnvtfmt_read_paths': parameters['cnvtfmt_in'],
        'cnvtfmt_write_path': parameters['cnvtfmt_out'],
        'func_info': {'cnvtfmt_op_sel': 'Convert', 'cnvtfmt_target_format': "png", 'cnvtfmt_prefix': ''}
    }
if 'settings_pnpcseg' not in st.session_state:
    st.session_state['settings_pnpcseg'] = {
        'source_option': None,
        'pnpcseg_read_paths': parameters['pnpcseg_in'],
        'pnpcseg_write_path': parameters['pnpcseg_out']
    }
if 'settings_segimg' not in st.session_state:
    st.session_state['settings_segimg'] = {
        'source_option': None,
        'segimg_read_paths': parameters['segimg_in'],
        'segimg_write_path': parameters['segimg_out'],
        'func_info': {'segimg_target_label': 'person', 'segimg_seg_map_path': './', 'segimg_fill_color': 'FFFFFFFF', 'segimg_rm': 'Crop', 'segimg_clip': False}
    }
if 'settings_shrpimg' not in st.session_state:
    st.session_state['settings_shrpimg'] = {
        'source_option': None,
        'shrpimgread_paths': parameters['shrpimg_in'],
        'shrpimg_write_path': parameters['shrpimg_out'],
        'func_info': {'coefs': [4.0, -0.8, 0.0], 'shrpimg_threshold': 300}
    }
#if 'settings_hmnprs' not in st.session_state:
#    st.session_state['settings_hmnprs'] = {
#        'source_option': None,
#        'segimg_read_paths': "./",
#        'segimg_write_path': None,
#        'func_info': {'face_clip': False}
#    }
if 'settings_rszimg' not in st.session_state:
    st.session_state['settings_rszimg'] = {
        'source_option': None,
        'rszimg_read_paths': parameters['rszimg_in'],
        'rszimg_write_path': parameters['rszimg_out'],
        'func_info': {'rszimg_height': 0, 'rszimg_width': 0, 'rszimg_maintain_aspect_ratio': True, 'use_cv': False}
    }
if 'settings_augimg' not in st.session_state:
    rotate = {'augimg_rotate_angle': 0.0, 'augimg_rotate_range': 0.0, 'augimg_rotate_angle_tic': 0.0}
    slide = {'augimg_slide_range': [0,0], 'augimg_slide_tic': [0,0]}
    func_info = {'aug_op': 'slide', 'rotate': rotate, 'slide': slide}

    st.session_state['settings_augimg'] = {
        'source_option': None,
        'augimg_read_paths': parameters['augimg_in'],
        'augimg_write_path': parameters['augimg_out'],
        'func_info': func_info
    }
if 'settings_augtxt' not in st.session_state:
    st.session_state['settings_augtxt'] = {
        'source_option': None,
        'augtxt_read_paths': parameters['augtxt_in'],
        'augtxt_write_path': parameters['augtxt_out'],
        'func_info': {'op_sel': 'Remove Tags', 'augtxt_max_length': 100, 'undesired_tokens_path': './'}
    }
if 'settings_nrmimg' not in st.session_state:
    st.session_state['settings_nrmimg'] = {
        'source_option': None,
        'nrmimg_read_paths': parameters['nrmimg_in'],
        'nrmimg_write_path': parameters['nrmimg_out']
    }
if 'settings_avgimg' not in st.session_state:
    st.session_state['settings_avgimg'] = {
        'source_option': None,
        'avgimg_read_paths': parameters['cnvtfmt_in'],
        'avgimg_write_path': parameters['avgimg_out'],
        'func_info': {'device': 'cuda', 'iter_no': 0}
    }

def selload(func, settion):
    source_option_ = st.radio('Select source type:', ['Upload Files', 'Specify Directory'], key='sel_{func}')
    if settion == None:
        source_option = source_option_
    else:
        source_option = source_option_#settion

    return source_option


## End-to-End Batch Processing
selected_functions = []
if 'selected_functions' not in st.session_state:
    st.session_state.selected_functions = []


## Main Layout
if tab_selection == 'Convert Files':
    # Settings
    source_option = st.radio('Select source type:', ['Upload Files', 'Specify Directory'], key='sel_cnvtfmt')

    if source_option == 'Upload Files':
        cnvtfmt_read_paths = st.file_uploader('Files for Convert:', type=None, accept_multiple_files=True)
        restored = False
    else:
        cnvtfmt_read_paths = st.text_input('Directory for Convert:', st.session_state.settings_cnvtfmt.get('cnvtfmt_read_paths'))
        restored = False

    cnvtfmt_write_path = st.text_input('Convert Write Path:', st.session_state.settings_cnvtfmt.get('cnvtfmt_write_path', parameters['cnvtfmt_out']))

    cnvtfmt_op_sel = st.radio('Select task type', ['Convert', 'Random', 'Prefix', 'Remove', 'Spawn'], key='sel_cvrtfmt_op')

    if cnvtfmt_op_sel == 'Prefix':
        cnvtfmt_prefix = st.text_input('Enter Prefix Text')
    elif cnvtfmt_op_sel == 'Spawn':
        cnvtfmt_prefix = st.text_input('Enter Spawning Number')
    else:
        cnvtfmt_prefix = ''
    cnvtfmt_prefix = cnvtfmt_prefix.replace(' ', '')

    options = ['png', 'jpeg', 'bmp', 'tiff']
    if st.session_state.settings_cnvtfmt:
        default_index = options.index(st.session_state.settings_cnvtfmt.get('cnvtfmt_target_format', 'png'))
    else:
        default_index = options.index('png')
    if cnvtfmt_op_sel == 'Convert':
        cnvtfmt_target_format = st.selectbox('Target Format:', options, index=default_index)
    else:
        cnvtfmt_target_format = 'png'

    # Working Part
    if source_option != 'Upload Files':
        save_button = st.button('Save Settings')
        load_button = st.button('Load Settings')
        filename = st.text_input('Settings Filename:', 'settings_cnvtfmt.json')

        if save_button:
            func_info = {'cnvtfmt_op_sel': cnvtfmt_op_sel, 'cnvtfmt_target_format': cnvtfmt_target_format, 'cnvtfmt_prefix': cnvtfmt_prefix}
            save_settings(filename, 'source_option', cnvtfmt_read_paths, cnvtfmt_write_path, func_info)
            st.success(f'Settings saved to {filename}')
            st.session_state.settings_cnvtfmt['source_option'] = source_option
            st.session_state.settings_cnvtfmt['cnvtfmt_read_paths'] = cnvtfmt_read_paths
            st.session_state.settings_cnvtfmt['cnvtfmt_write_path'] = cnvtfmt_write_path
            st.session_state.settings_cnvtfmt['func_info'] = func_info

        if load_button:
            source_option, cnvtfmt_read_paths, cnvtfmt_write_path, func_info = load_settings(filename)
            st.session_state.settings_cnvtfmt['source_option'] = source_option
            st.session_state.settings_cnvtfmt['cnvtfmt_read_paths'] = cnvtfmt_read_paths
            st.session_state.settings_cnvtfmt['cnvtfmt_write_path'] = cnvtfmt_write_path
            cnvtfmt_op_sel = func_info.get('cnvtfmt_op_sel')
            cnvtfmt_target_format = func_info.get('cnvtfmt_target_format')
            cnvtfmt_prefix = func_info.get('cnvtfmt_prefix')
            st.session_state.settings_cnvtfmt['cnvtfmt_op_sel'] = cnvtfmt_op_sel
            st.session_state.settings_cnvtfmt['cnvtfmt_target_format'] = cnvtfmt_target_format
            st.session_state.settings_cnvtfmt['cnvtfmt_prefix'] = cnvtfmt_prefix
            restored = True

            source_option = 'Upload Files Reconstructed'


    start_run = st.button('Start Convert')

    epoch_size, batch_size = calc_env(restored, device, available_memory, memory_size, cnvtfmt_read_paths, coef, 1)

    if start_run and memory_size > 0:
        cnvtfmt(0, batch_size, source_option, cnvtfmt_read_paths, cnvtfmt_write_path, cnvtfmt_op_sel, cnvtfmt_target_format, cnvtfmt_prefix)

elif tab_selection == 'Sharpen Images':
    # Settings
    source_option = 'Specify Directory'
    #source_option = st.radio('Select source type:', ['Upload Files', 'Specify Directory'], key='sel_cnvtfmt')

    if source_option == 'Upload Files':
        shrpimg_read_paths = st.file_uploader('Files for Sharpen:', type=None, accept_multiple_files=True)
        restored = False
    else:
        shrpimg_read_paths = st.text_input('Directory for Sharpen:', st.session_state.settings_shrpimg.get('shrpimg_read_paths', parameters['shrpimg_in']))
        restored = False

    shrpimg_write_path = st.text_input('Convert Write Path:', st.session_state.settings_shrpimg.get('shrpimg_write_path', parameters['shrpimg_out']))

    shrpimg_coef_center = st.slider('Set center coefficient:', 1.0, 10.0, 4.0, step=0.2)
    shrpimg_coef_neighbor = st.slider('Set neighbor coefficient:', -1.5, -0.5, -0.8, step=0.1)
    shrpimg_coef_diagonarl = st.slider('Set neighbor coefficient:', -0.25, 0.25, 0.0, step=0.05)
    shrpimg_coefs = [shrpimg_coef_center, shrpimg_coef_neighbor, shrpimg_coef_diagonarl]

    shrpimg_threshold = st.slider('Set Threshold:', 50, 1000, 500, step=50)

    # Working Part
    if source_option != 'Upload Files':
        save_button = st.button('Save Settings')
        load_button = st.button('Load Settings')
        filename = st.text_input('Settings Filename:', 'settings_shrpimg.json')

        if save_button:
            func_info = {'shrpimg_coefs': shrpimg_coefs, 'shrpimg_threshold': shrpimg_threshold}
            save_settings(filename, 'source_option', shrpimg_read_paths, shrpimg_write_path, func_info)
            st.success(f'Settings saved to {filename}')
            st.session_state.settings_shrpimg['source_option'] = source_option
            st.session_state.settings_shrpimg['shrpimg_read_paths'] = shrpimg_read_paths
            st.session_state.settings_shrpimg['shrpimg_write_path'] = shrpimg_write_path
            st.session_state.settings_shrpimg['func_info'] = func_info

        if load_button:
            source_option, shrpimg_read_paths, shrpimg_write_path, func_info = load_settings(filename)
            st.session_state.settings_shrpimg['source_option'] = source_option
            st.session_state.settings_shrpimg['shrpimg_read_paths'] = shrpimg_read_paths
            st.session_state.settings_shrpimg['shrpimg_write_path'] = shrpimg_write_path
            shrpimg_coefs = func_info.get('coefs')
            st.session_state.settings_shrpimg['coefs'] = shrpimg_coefs
            shrpimg_threshold = func_info.get('shrpimg_threshold')
            st.session_state.settings_shrpimg['shrpimg_threshold'] = shrpimg_threshold
            restored = True

            source_option = 'Upload Files Reconstructed'


    start_run = st.button('Start Convert')

    epoch_size, batch_size = calc_env(restored, device, available_memory, memory_size, shrpimg_read_paths, coef, 1)

    if start_run and memory_size > 0:
        shrpimg(0, batch_size, source_option, shrpimg_read_paths, shrpimg_write_path, shrpimg_coefs, shrpimg_threshold)


elif tab_selection == 'Picture Parse':
    # Settings
    source_option = st.radio('Select source type:', ['Upload Files', 'Specify Directory'], key='sel_pnpcdseg')

    if source_option == 'Upload Files':
        pnpcseg_read_paths = st.file_uploader('Files for Panopic Seg:', type=None, accept_multiple_files=True)
        restored = False
    else:
        pnpcseg_read_paths = st.text_input('Directory for Panopic Seg:', st.session_state.settings_pnpcseg.get('pnpcseg_read_paths'))
        restored = False

    pnpcseg_write_path = st.text_input('Panopic Seg Write Path:', st.session_state.settings_pnpcseg.get('pnpcseg_write_path', parameters['pnpcseg_out']))
    func_info = ''


    # Working Part
    if source_option != 'Upload Files':
        save_button = st.button('Save Settings')
        load_button = st.button('Load Settings')
        filename = st.text_input('Settings Filename:', 'settings_pnpcseg.json')

        if save_button:
            func_info = {}
            save_settings(filename, 'source_option', pnpcseg_read_paths, pnpcseg_write_path, func_info)
            st.success(f'Settings saved to {filename}')
            st.session_state.settings_pnpcseg['source_option'] = source_option
            st.session_state.settings_pnpcseg['pnpcseg_read_paths'] = pnpcseg_read_paths
            st.session_state.settings_pnpcseg['pnpcseg_write_path'] = pnpcseg_write_path

        if load_button:
            source_option, pnpcseg_read_paths, pnpcseg_write_path, func_info = load_settings(filename)
            st.session_state.settings_pnpcseg['source_option'] = source_option
            st.session_state.settings_pnpcseg['pnpcseg_read_paths'] = pnpcseg_read_paths
            st.session_state.settings_pnpcseg['pnpcseg_write_path'] = pnpcseg_write_path
            restored = True

            source_option = 'Upload Files Reconstructed'

    start_run = st.button('Start Panopic  Segment')

    epoch_size, batch_size = calc_env(restored, device, available_memory, memory_size, pnpcseg_read_paths, coef, 201)

    def panopic_segment(iter_no, file):
        pnpcseg(iter_no, batch_size, device, pnpc_model, source_option, pnpcseg_read_paths, pnpcseg_write_path)

    if start_run and memory_size > 0:
        #   Transfer model to device
        pnpc_model = pnpc_model.to(device)
        batch_process(epoch_size, batch_size, pnpcseg_read_paths, panopic_segment)

    start_run = False


#elif tab_selection == 'Human Parse':
#    # Settings
#    source_option  = st.radio('Select source type:', ['Upload Files', 'Specify Directory'], key='sel_hmnprsimg')
#
#    if source_option == 'Upload Files':
#        hmnprs_read_paths = st.file_uploader('Human Parse Read:', type=None, accept_multiple_files=True)
#        restored = False
#    else:
#        hmnprs_read_paths = st.text_input('Directory for Human Parse:', st.session_state.settings_hmnprs.get('hmnprs_read_paths'))
#        restored = False
#
#    hmnprs_write_path = st.text_input('Resized Write Path:', st.session_state.settings_hmnprs.get('hmnprs_write_path', parameters['rszimg_out']))
#
#    face_clip = st.checkbox('Face Clipping')
#    func_info = ''
#
#    # Working Part
#    save_button = st.button('Save Settings')
#    load_button = st.button('Load Settings')
#    filename = st.text_input('Settings Filename:', 'settings_hmnprs.json')
#
#    if save_button:
#        func_info = {'face_clip': face_clip}
#        save_settings(filename, 'source_option', hmnprs_read_paths, hmnprs_write_path, func_info)
#        st.success(f'Settings saved to {filename}')
#        st.session_state.settings_hmnprs['source_option'] = 'Specify Directory'
#        st.session_state.settings_hmnprs['hmnprs_read_paths'] = hmnprs_read_paths
#        st.session_state.settings_hmnprs['hmnprs_write_path'] = hmnprs_write_path
#        st.session_state.settings_hmnprs['func_info'] = func_info
#
#    if load_button:
#        source_option, hmnprs_read_paths, hmnprs_write_path, func_info = load_settings(filename)
#        st.session_state.settings_hmnprs['source_option'] = source_option
#        st.session_state.settings_hmnprs['hmnprs_read_paths'] = hmnprs_read_paths
#        st.session_state.settings_hmnprs['hmnprs_write_path'] = hmnprs_write_path
#        face_clip = func_info.get('face_clip', False)
#        st.session_state.settings_hmnprs['face_clip'] = face_clip
#        restored = True
#
#        source_option = 'Upload Files Reconstructed'
#
#    start_run = st.button('Start Segment')
#
#    epoch_size, batch_size = calc_env(restored, device, available_memory, memory_size, hmnprs_read_paths, coef, 201)
#
#    def human_parse(iter_no, file):
#        hmnprs(iter_no, batch_size, source_option, hmnprs_model, hmnprs_read_paths, hmnprs_write_path, face_clip)
#
#    if start_run and memory_size > 0:
#        #   Transfer model to device
#        hmnprs_model = hmnprs_model.to(device)
#        batch_process(epoch_size, batch_size, hmnprs_read_paths, human_parse)
#
#    start_run = False


elif tab_selection == 'Segment':
    # Settings
    source_option = 'Specify Directory'
    segimg_read_paths = st.text_input('Specify Source Directory:', st.session_state.settings_segimg.get('segimg_read_paths'))
    restored = False

    segimg_write_path = st.text_input('Segment Write Path:', st.session_state.settings_segimg.get('segimg_write_path', parameters['segimg_out']))
    func_info = ''

    segimg_target_label = st.text_input('Label Name:', 'person')
    segimg_seg_map_path = st.text_input('Segment Map Path:', './')

    segimg_fill_color = st.color_picker('Select fill color:', '#FFFFFF')

    segimg_rm = st.radio('Select segment type:', ['Remove', 'Crop'])

    if segimg_rm == 'Crop':
        segimg_clip = st.checkbox('clipping')
    else:
        segimg_clip = False

    color_square = f'<div style="width: 30px; height: 30px; background-color: {segimg_fill_color};"></div>'
    #st.markdown(color_square, unsafe_allow_html=True)


    # Working Part
    save_button = st.button('Save Settings')
    load_button = st.button('Load Settings')
    filename = st.text_input('Settings Filename:', 'settings_segimg.json')

    if save_button:
        func_info = {'segimg_target_label': segimg_target_label, 'segimg_seg_map_path': segimg_seg_map_path, 'segimg_fill_color': segimg_fill_color, 'segimg_rm': segimg_rm, 'segimg_clip': segimg_clip}
        save_settings(filename, 'source_option', segimg_read_paths, segimg_write_path, func_info)
        st.success(f'Settings saved to {filename}')
        st.session_state.settings_segimg['source_option'] = 'Specify Directory'
        st.session_state.settings_segimg['segimg_read_paths'] = segimg_read_paths
        st.session_state.settings_segimg['segimg_write_path'] = segimg_write_path
        st.session_state.settings_segimg['func_info'] = func_info

    if load_button:
        source_option, segimg_read_paths, segimg_write_path, func_info = load_settings(filename)
        st.session_state.settings_segimg['source_option'] = source_option
        st.session_state.settings_segimg['segimg_read_paths'] = segimg_read_paths
        st.session_state.settings_segimg['segimg_write_path'] = segimg_write_path
        segimg_target_label = func_info.get('segimg_target_label', 'person')
        segimg_seg_map_path = func_info.get('segimg_seg_map_path', './')
        segimg_fill_color = func_info.get('segimg_fill_color')
        segimg_rm = func_info.get('segimg_rm', 'Crop')
        segimg_clip = func_info.get('segimg_clip', False)
        st.session_state.settings_segimg['segimg_target_label'] = segimg_target_label
        st.session_state.settings_segimg['segimg_seg_map_path'] = segimg_seg_map_path
        st.session_state.settings_segimg['segimg_fill_color'] = segimg_fill_color
        st.session_state.settings_segimg['segimg_rm'] = segimg_rm
        restored = True

        source_option = 'Upload Files Reconstructed'

    start_run = st.button('Start Segment')

    epoch_size, batch_size = calc_env(restored, device, available_memory, memory_size, segimg_read_paths, coef, 1)

    def segment(iter_no, file):
        segimg(iter_no, batch_size, source_option, segimg_read_paths, segimg_write_path, segimg_seg_map_path, segimg_target_label, segimg_fill_color, segimg_rm, segimg_clip)

    if start_run and memory_size > 0:
        batch_process(epoch_size, batch_size, segimg_read_paths, segment)

    start_run = False


elif tab_selection == 'Resize Images':
    # Settings
    source_option  = st.radio('Select source type:', ['Upload Files', 'Specify Directory'], key='sel_rszimg')

    if source_option == 'Upload Files':
        rszimg_read_paths = st.file_uploader('Resize Read:', type=None, accept_multiple_files=True)
        restored = False
    else:
        rszimg_read_paths = st.text_input('Directory for Resize:', st.session_state.settings_rszimg.get('rszimg_read_paths'))
        restored = False

    rszimg_write_path = st.text_input('Resized Write Path:', st.session_state.settings_rszimg.get('rszimg_write_path', parameters['rszimg_out']))

    try:
        rszimg_height = int(st.text_input('Height to resize', '512'))
        rszimg_width = int(st.text_input('Width to resize:', '512'))
    except ValueError:
        st.warning('Please enter a valid number for height and width.')
        rszimg_height, rszimg_width = None, None

    rszimg_use_cv = st.checkbox('Up Scaling')

    options = ['with aspect ratio', 'without aspect ratio']
    rszimg_maintain_aspect_ratio = st.radio('Choose an option:', options) =='with aspect ratio'

    func_info = ''


    # Working Part
    if source_option != 'Upload Files':
        save_button = st.button('Save Settings')
        load_button = st.button('Load Settings')
        filename = st.text_input('Settings Filename:', 'settings_rszimg.json')

        if save_button:
            func_info = {'rszimg_height': rszimg_height, 'rszimg_width': rszimg_width, 'rszimg_maintain_aspect_ratio': rszimg_maintain_aspect_ratio, 'rszimg_use_cv': rszimg_use_cv}
            save_settings(filename, 'source_option', rszimg_read_paths, rszimg_write_path, func_info)
            st.success(f'Settings saved to {filename}')
            st.session_state.settings_rszimg['source_option'] = source_option
            st.session_state.settings_rszimg['rszimg_read_paths'] = rszimg_read_paths
            st.session_state.settings_rszimg['rszimg_write_path'] = rszimg_write_path
            st.session_state.settings_rszimg['func_info'] = func_info

        if load_button:
            source_option, rszimg_read_paths, rszimg_write_path, func_info = load_settings(filename)
            st.session_state.settings_rszimg['source_option'] = source_option
            st.session_state.settings_rszimg['rszimg_read_paths'] = rszimg_read_paths
            st.session_state.settings_rszimg['rszimg_write_path'] = rszimg_write_path
            rszimg_height = func_info.get('rszimg_height')
            rszimg_width = func_info.get('rszimg_width')
            rszimg_maintain_aspect_ratio = func_info.get('rszimg_maintain_aspect_ratio')
            rszimg_use_cv = func_info.get('rszimg_use_cv')
            st.session_state.settings_rszimg['func_info'] = func_info
            restored = True

            source_option = 'Upload Files Reconstructed'

    start_run = st.button('Start Resize')

    epoch_size, batch_size = calc_env(restored, device, available_memory, memory_size, rszimg_read_paths, coef, 1)

    if start_run and memory_size > 0:
        rszimg(rszimg_read_paths, rszimg_write_path, rszimg_height, rszimg_width, rszimg_maintain_aspect_ratio, rszimg_use_cv)

    start_run = False


elif tab_selection == 'Augment Images':
    # Settings
    source_option  = st.radio('Select source type:', ['Upload Files', 'Specify Directory'], key='sel_augimg')

    if source_option == 'Upload Files':
        augimg_read_paths = st.file_uploader('Files for Augment Images:', type=None, accept_multiple_files=True)
        restored = False
    else:
        augimg_read_paths = st.text_input('Directory for Augment Images:', st.session_state.settings_augimg.get('augimg_read_paths'))
        restored = False

    augimg_write_path = st.text_input('Aug-Image Write Path:', st.session_state.settings_augimg.get('augimg_write_path', parameters['augimg_out']))
    func_info = ''

    options = ['slide', 'rotate', 'flip_lr', 'flip_tb']
    if 'settings_augimg' not in st.session_state or 'aug_op' not in st.session_state.settings_augimg.get('func_info'):
        initial_index = options.index('slide')
    else:
        aug_op_value = st.session_state.settings_augimg.get('func_info')['aug_op']

        if aug_op_value in options:
            initial_index = options.index(aug_op_value)
        else:
            initial_index = options.index('slide')
    aug_op = st.radio('choice work', options, index=initial_index)

    if aug_op == 'rotate':
        try:
            augimg_rotate_angle = int(st.text_input('Base Angle:', '0'))
        except ValueError:
            st.warning('Please enter a valid number for base angle.')
            augimg_rotate_angle = None

        augimg_rotate_range = st.slider('Rotate Range:', min_value=0.0, max_value=90.0, step=1.0)
        augimg_rotate_angle_tic = st.number_input('Angle Tic:', min_value=0.1, value=1.0, step=0.1)

        st.write(f'NOTE: total {int(2.0 * augimg_rotate_range / augimg_rotate_angle_tic)} images per one source image will be generated by rotate.')

        augimg_slide_range = [0,0]
        augimg_slide_tic = [0,0]

    elif aug_op == 'slide':
        try:
            augimg_slide_x_range = st.slider('Slide X-Range:', min_value=0, max_value=15, step=1)
            augimg_slide_x_tic = st.slider('X-Tic:', min_value=1, max_value=5, step=1)
            augimg_slide_y_range = st.slider('Slide Y-Range:', min_value=0, max_value=15, step=1)
            augimg_slide_y_tic = st.slider('Y-Tic:', min_value=1, max_value=5, step=1)
            augimg_slide_range = [augimg_slide_x_range, augimg_slide_y_range]
            augimg_slide_tic = [augimg_slide_x_tic, augimg_slide_y_tic]

            x_range = (augimg_slide_x_range*2 +1)// augimg_slide_x_tic
            y_range = (augimg_slide_y_range*2 +1)// augimg_slide_y_tic
            num_aug_files = x_range*y_range

            st.write(f'NOTE: total {num_aug_files} images per one source image will be generated by slide.')

        except ValueError:
            st.warning('Please enter a valid number for sliding.')
            augimg_slide_range = None
            augimg_slide_tic = None

        augimg_rotate_range = 0.0
        augimg_rotate_angle_tic = 0.0
        augimg_rotate_angle = None

    else:
        augimg_slide_range = [0,0]
        augimg_slide_tic = [0,0]
        augimg_rotate_range = 0.0
        augimg_rotate_angle_tic = 0.0
        augimg_rotate_angle = None


    # Working Part
    if source_option != 'Upload Files':
        save_button = st.button('Save Settings')
        load_button = st.button('Load Settings')
        filename = st.text_input('Settings Filename:', 'settings_augimg.json')

        if save_button:
            rotate = {'augimg_rotate_angle': augimg_rotate_angle, 'augimg_rotate_range': augimg_rotate_range, 'augimg_rotate_angle_tic': augimg_rotate_angle_tic}
            slide = {'augimg_slide_range': augimg_slide_range, 'augimg_slide_tic':augimg_slide_tic}
            func_info = {'aug_op': aug_op, 'rotate': rotate, 'slide': slide}
            save_settings(filename, 'source_option', augimg_read_paths, augimg_write_path, func_info)
            st.success(f'Settings saved to {filename}')
            st.session_state.settings_augimg['source_option'] = source_option
            st.session_state.settings_augimg['augimg_read_paths'] = augimg_read_paths
            st.session_state.settings_augimg['augimg_write_path'] = augimg_write_path
            st.session_state.settings_augimg['func_info'] = func_info

        if load_button:
            source_option, augimg_read_paths, augimg_write_path, func_info = load_settings(filename)
            st.session_state.settings_augimg['source_option'] = source_option
            st.session_state.settings_augimg['augimg_read_paths'] = augimg_read_paths
            st.session_state.settings_augimg['augimg_write_path'] = augimg_write_path

            aug_op = func_info.get('aug_op')

            rotate = func_info.get('rotate')
            augimg_rotate_angle = rotate.get('augimg_rotate_angle')
            augimg_rotate_range = rotate.get('augimg_rotate_range')
            augimg_rotate_angle_tic = rotate.get('augimg_rotate_angle_tic')

            slide = func_info.get('slide')
            augimg_slide_range = slide.get('augimg_slide_range')
            augimg_slide_tic = slide.get('augimg_slide_tic')

            st.session_state.settings_augimg['func_info'] = func_info
            restored = True

            source_option = 'Upload Files Reconstructed'

    start_run = st.button('Start Aug Images')

    epoch_size, batch_size = calc_env(restored, device, available_memory, memory_size, augimg_read_paths, coef, 1)

    if start_run and memory_size > 0:
        augimg(0, batch_size, source_option, augimg_read_paths, augimg_write_path, aug_op, augimg_rotate_angle, augimg_rotate_range, augimg_rotate_angle_tic, augimg_slide_range, augimg_slide_tic)

    start_run = False


elif tab_selection == 'Augment Texts':
    op_sel = st.radio('Select work type', ['Stat Texts', 'Remove Tags', 'Find File', 'Edit Texts', 'Augment Texts'])

    # Settings
    if op_sel == 'Augment Texts':
        source_option  = st.radio('Select source type:', ['Upload Files', 'Specify Directory'], key='sel_augtxt')

        if source_option == 'Upload Files':
            augtxt_read_paths = st.file_uploader('Files for Augment Texts:', type=None, accept_multiple_files=True)
            restored = False
        else:
            augtxt_read_paths = st.text_input('Directory for Augment Texts:', st.session_state.settings_augtxt.get('augtxt_read_paths'))
            restored = False
    elif op_sel == 'Remove Tags':
        source_option = 'Specify Directory'
        augtxt_read_paths = st.text_input('Directory for Augment Texts:', st.session_state.settings_augtxt.get('augtxt_read_paths'))
        restored = False
    elif op_sel == 'Stat Texts':
        ext = st.text_input('file extension:', 'caption')
        source_option = {'ext': ext}
    elif op_sel == 'Find File':
        tokens_words_file_path = st.text_input('Specify Directory for Tokens/Words')
        ext = st.text_input('file extension:', 'caption')
        source_option = {'ext': ext, 'tokens_words_file_path': tokens_words_file_path}
    elif op_sel == 'Edit Texts':
        tokens_words_file_path  = st.text_input('Specify Directory for Undesired Tokens/Words')
        source_option = {'tokens_words_file_path': tokens_words_file_path}


    augtxt_write_path = st.text_input('Aug-Text Write Path:', st.session_state.settings_augtxt.get('augtxt_write_path', parameters['augtxt_out']))

    if op_sel == 'Augment Texts':
        augtxt_max_length = st.slider('Max Length:', min_value=0, max_value=150, step=5)
        undesired_tokens_path = './'
    else:
        augtxt_max_length = 0
        undesired_tokens_path = st.text_input('Directory and file name for Undesired tag file')

    # Working Part
    if source_option != 'Upload Files':
        save_button = st.button('Save Settings')
        load_button = st.button('Load Settings')
        filename = st.text_input('Settings Filename:', 'settings_augtxt.json')

        if save_button:
            func_info = {'op_sel': op_sel, 'augtxt_max_length': augtxt_max_length, 'undesired_tokens_path': undesired_tokens_path}
            save_settings(filename, 'source_option', augtxt_read_paths, augtxt_write_path, func_info)
            st.success(f'Settings saved to {filename}')
            st.session_state.settings_augtxt['source_option'] = source_option
            st.session_state.settings_augtxt['augtxt_read_paths'] = augtxt_read_paths
            st.session_state.settings_augtxt['augtxt_write_path'] = augtxt_write_path
            st.session_state.settings_augtxt['func_info'] = func_info

        if load_button:
            source_option, augtxt_read_paths, augimg_write_path, func_info = load_settings(filename)
            st.session_state.settings_augtxt['source_option'] = source_option
            st.session_state.settings_augtxt['augtxt_read_paths'] = augtxt_read_paths
            st.session_state.settings_augtxt['augtxt_write_path'] = augtxt_write_path
            op_sel = func_info.get('op_sel')
            augtxt_max_length = func_info.get('augtxt_max_length', 100)
            undesired_tokens_path = func_info.get('undesired_tokens_path')
            st.session_state.settings_augtxt['op_sel'] = op_sel
            st.session_state.settings_augtxt['augtxt_max_length'] = augtxt_max_length
            st.session_state.settings_augtxt['undesired_tokens_path'] = undesired_tokens_path
            restored = True

            source_option = 'Upload Files Reconstructed'


    start_run = st.button('Start Aug Texts')

    epoch_size, batch_size = calc_env(restored, device, available_memory, memory_size, augtxt_read_paths, coef, 1)

    def aug_text(iter_no, file):
        augtxt(batch_size, source_option, augtxt_read_paths, augtxt_write_path, op_sel, augtxt_max_length, undesired_tokens_path)

    if start_run and memory_size > 0:
        batch_process(epoch_size, batch_size, augtxt_read_paths, aug_text)


    start_run = False


elif tab_selection == 'Generate Normalized Images':
    # Settings
    source_option  = st.radio('Select source type:', ['Upload Files', 'Specify Directory'], key='sel_nrmimg')

    if source_option == 'Upload Files':
        nrmimg_read_paths = st.file_uploader('Files for Normalize:', type=None, accept_multiple_files=True)
        restored = False
    else:
        nrmimg_read_paths = st.text_input('Directory for Normalize:', st.session_state.settings_nrmimg.get('nrmimg_read_paths'))
        restored = False

    nrmimg_write_path = st.text_input('Normalized Write Path:', st.session_state.settings_nrmimg.get('nrmimg_write_path', parameters['nrmimg_out']))
    func_info = ''


    # Working Part
    if source_option != 'Upload Files':
        save_button = st.button('Save Settings')
        load_button = st.button('Load Settings')
        filename = st.text_input('Settings Filename:', 'settings_nrmimg.json')

        if save_button:
            save_settings(filename, 'source_option', nrmimg_read_paths, nrmimg_write_path, func_info)
            st.success(f'Settings saved to {filename}')
            st.session_state.settings_nrmimg['source_option'] = source_option
            st.session_state.settings_nrmimg['nrmimg_read_paths'] = nrmimg_read_paths
            st.session_state.settings_nrmimg['nrmimg_write_path'] = nrmimg_write_path

        if load_button:
            source_option, nrmimg_read_paths, augimg_write_path, func_info = load_settings(filename)
            st.session_state.settings_nrmimg['source_option'] = source_option
            st.session_state.settings_nrmimg['nrmimg_read_paths'] = nrmimg_read_paths
            st.session_state.settings_nrmimg['nrmimg_write_path'] = nrmimg_write_path
            restored = True

            source_option = 'Upload Files Reconstructed'


    start_run = st.button('Start Normalize')

    epoch_size, batch_size = calc_env(restored, device, available_memory, memory_size, nrmimg_read_paths, coef, 1)

    def normalize_images(iter_no, file):
        nrmimg(iter_no, batch_size, device, source_option, nrmimg_read_paths, nrmimg_write_path)

    if start_run and memory_size > 0:
        batch_process(epoch_size, batch_size, nrmimg_read_paths, normalize_images)

    start_run = False


elif tab_selection == 'Generate Average Image':
    # Settings
    source_option = st.radio('Select source type:', ['Upload Files', 'Specify Directory'], key='sel_avgimg')

    if source_option == 'Upload Files':
        avgimg_read_paths = st.file_uploader('Files for Average:', type=None, accept_multiple_files=True)
        restored = False
    else:
        avgimg_read_paths = st.text_input('Directory for Average:', st.session_state.settings_avgimg.get('avgimg_read_paths'))
        restored = False

    avgimg_write_path = st.text_input('Averaged Write Path:', st.session_state.settings_avgimg.get('avgimg_write_path', parameters['avgimg_out']))
    func_info = ''


    # Working Part
    if source_option != 'Upload Files':
        save_button = st.button('Save Settings')
        load_button = st.button('Load Settings')
        filename = st.text_input('Settings Filename:', 'settings_avgimg.json')

        if save_button:
            save_settings(filename, 'source_option', avgimg_read_paths, avgimg_write_path, func_info)
            st.success(f'Settings saved to {filename}')
            st.session_state.settings_avgimg['source_option'] = source_option
            st.session_state.settings_avgimg['avgimg_read_paths'] = avgimg_read_paths
            st.session_state.settings_avgimg['avgimg_write_path'] = avgimg_write_path

        if load_button:
            source_option, avgimg_read_paths, augimg_write_path, func_info = load_settings(filename)
            st.session_state.settings_avgimg['source_option'] = source_option
            st.session_state.settings_avgimg['avgimg_read_paths'] = avgimg_read_paths
            st.session_state.settings_avgimg['avgimg_write_path'] = avgimg_write_path
            restored = True

            source_option = 'Upload Files Reconstructed'

    start_run = st.button('Start Average')

    epoch_size, batch_size = calc_env(restored, device, available_memory, memory_size, avgimg_read_paths, coef, 1)

    def average_image(iter_no, file):
        avgimg(iter_no, batch_size, device, source_option, avgimg_read_paths, avgimg_write_path)

    if start_run and memory_size > 0:
        iter_no = 0
        st.session_state.nrmimg_selected['iter_no'] = iter_no
        st.session_state.nrmimg_selected['device'] = device
        batch_process(epoch_size, batch_size, avgimg_read_paths, average_image)

    start_run = False


elif tab_selection == 'End-to-End Run' and not start_run:
    st.session_state.start = {'start': True}

    tab = st.selectbox('Choose a function', list(function_map.keys()))


    if tab == 'Convert Files':
        if 'settings_cnvtfmt' in st.session_state and st.session_state.settings_cnvtfmt['cnvtfmt_read_paths'] is not None:
            source_option = st.session_state.settings_cnvtfmt['source_option']
            cnvtfmt_read_paths = st.session_state.settings_cnvtfmt['cnvtfmt_read_paths']
            cnvtfmt_write_path = st.session_state.settings_cnvtfmt['cnvtfmt_write_path']
            func_info = st.session_state.settings_cnvtfmt['func_info']
            cnvtfmt_op_sel = func_info.get['cnvtfmt_op_sel']
            cnvtfmt_target_format = func_info.get('cnvtfmt_target_format')
            cnvtfmt_prefix = func_info.get['cnvtfmt_prefix']
        else:
            source_option = 'Specify Directory'
            cnvtfmt_read_paths = './'
            cnvtfmt_write_path = './'
            func_info = {'cnvtfmt_op_sel': 'Convert', 'cnvtfmt_target_format': 'png', 'cnvtfmt_prefix': ''}

        if st.button('Add Convert Files'):
            if cnvtfmt_read_paths == None:
                st.write('Specify Files before Add')
            else:
                selected_functions.append({'function': 'Convert Files', 'params': {'iter_no': 0, 'batch_size': 1, 'source_option': source_option, 'read_paths': cnvtfmt_read_paths, 'write_path': cnvtfmt_write_path, 'cnvtfmt_op_sel': cnvtfmt_op_sel, 'target_format': cnvtfmt_target_format, 'cnvtfmt_prefix': cnvtfmt_prefix}})
                st.session_state.selected_functions.append(selected_functions)


    elif tab == 'Picture Parse':
        if 'settings_pnpcseg' in st.session_state and st.session_state.settings_pnpcseg['pnpcseg_read_paths'] is not None:
            source_option = st.session_state.settings_pnpcseg['source_option']
            pnpcseg_read_paths = st.session_state.settings_pnpcseg['pnpcseg_read_paths']
            pnpcseg_write_path = st.session_state.settings_pnpcseg['pnpcseg_write_path']
        else:
            source_option = 'Specify Directory'
            pnpcseg_read_paths = './'
            pnpcseg_write_path = './'
            func_info = {}

        if st.button('Add Picture Parse'):
            if pnpcseg_read_paths == None:
                st.write('Specify Files before Add')
            else:
                selected_functions.append({'function': 'Picture Parse', 'params': {'iter_no': 0, 'batch_size': 1, 'deviice': device, 'pnpc_model': pnpc_model, 'source_option': source_option, 'read_paths': pnpcseg_read_paths, 'write_path': pnpcseg_write_path}})
                st.session_state.selected_functions.append(selected_functions)


    elif tab == 'Segment':
        if 'settings_segimg' in st.session_state and st.session_state.settings_segimg['segimg_read_paths'] is not None:
            source_option = st.session_state.settings_segimg['source_option']
            segimg_read_paths = st.session_state.settings_segimg['segimg_read_paths']
            segimg_write_path = st.session_state.settings_segimg['segimg_write_path']
            func_info = st.session_state.settings_cnvtfmt['func_info']
            segimg_target_label = func_info.get('segimg_target_label')
            segimg_seg_map_path = func_info.get('segimg_seg_map_path')
            segimg_fill_color = func_info.get('segimg_fill_color')
            segimg_rm = func_info.get('segimg_rm')
            segimg_clip = func_info.get('segimg_clip')
        else:
            source_option = 'Specify Directory'
            segimg_read_paths = './'
            segimg_write_path = './'
            segimg_target_label = 'person'
            segimg_seg_map_path = './'
            segimg_fill_color =  'FFFFFFFF'
            segimg_rm = 'Crop'
            segimg_clip = False
            func_info = {'segimg_target_label': segimg_target_label, 'segimg_seg_map_path': segimg_seg_map_path, 'segimg_fill_color': segimg_fill_color, 'segimg_rm': segimg_rm, 'segimg_clip': segimg_clip}

        if st.button('Add Segment'):
            if segimg_read_paths == None:
                st.write('Specify Files before Add')
            else:
                selected_functions.append({'function': 'Segment', 'params': {'iter_no': 0, 'batch_size': 1, 'source_option': source_option,  'read_paths': segimg_read_paths, 'write_path': segimg_write_path, 'segmentation_map_paths': segimg_seg_map_path, 'target_label_name': segimg_target_label, 'fill_color': segimg_fill_color, 'rm': segimg_rm }})
                st.session_state.selected_functions.append(selected_functions)

    elif tab == 'Sharpen Images':
        if 'settings_shrpimg' in st.session_state and st.session_state.settings_shrpimg['segimg_read_paths'] is not None:
            source_option = st.session_state.settings_shrpimg['source_option']
            segimg_read_paths = st.session_state.settings_shrpimg['segimg_read_paths']
            segimg_write_path = st.session_state.settings_shrpimg['segimg_write_path']
            func_info = st.session_state.settings_cnvtfmt['func_info']
            shrpimg_coefs = func_info.get('shrpimg_coefs')
            shrpimg_threshold = func_info.get('shrpimg_threshold')
        else:
            source_option = 'Specify Directory'
            segimg_read_paths = './'
            segimg_write_path = './'
            shrpimg_coefs = [4.0, -0.8, 0.0]
            shrpimg_threshold = 500
            func_info = {'shrpimg_coefs': shrpimg_coefs, 'shrpimg_threshold': shrpimg_threshold}

        if st.button('Add Segment'):
            if segimg_read_paths == None:
                st.write('Specify Files before Add')
            else:
                selected_functions.append({'function': 'Sharpen Images', 'params': {'iter_no': 0, 'batch_size': 1, 'source_option': source_option,  'read_paths': segimg_read_paths, 'write_path': segimg_write_path, 'shrpimg_coefs': shrpimg_coefs, 'shrpimg_threshold': shrpimg_threshold}})
                st.session_state.selected_functions.append(selected_functions)

#    elif tab == 'Human Parse':
#        if 'settings_hmnprs' in st.session_state and st.session_state.settings_hmnprs['hmnprs_read_paths'] is not None:
#            source_option = st.session_state.settings_hmnprs['source_option']
#            hmnprs_read_paths = st.session_state.settings_hmnprs['hmnprs_read_paths']
#            hmnprs_write_path = st.session_state.settings_hmnprs['hmnprs_write_path']
#            func_info = st.session_state.settings_cnvtfmt['func_info']
#            face_clip = func_info.get('face_clip')
#        else:
#            source_option = 'Specify Directory'
#            hmnprs_read_paths = './'
#            hmnprs_write_path = './'
#            face_clip = False
#            func_info = {'face_clip': False}
#
#        if st.button('Add Human Parse'):
#            if hmnprs_read_paths == None:
#                st.write('Specify Files before Add')
#            else:
#                selected_functions.append({'function': 'Segment', 'params': {'iter_no': 0, 'batch_size': 1, 'source_option': source_option,  'read_paths': hmnprs_read_paths, 'write_path': hmnprs_write_path, 'face_clip': face_clip}})
#                st.session_state.selected_functions.append(selected_functions)


    elif tab == 'Resize Images':
        if 'settings_rszimg' in st.session_state and st.session_state.settings_rszimg['rszimg_read_paths'] is not None:
            source_option = st.session_state.settings_rszimg['source_option']
            rszimg_read_paths = st.session_state.settings_rszimg['rszimg_read_paths']
            rszimg_write_path = st.session_state.settings_rszimg['rszimg_write_path']
            func_info = st.session_state.settings_cnvtfmt['func_info']
            rszimg_height = func_info.get('rszimg_height')
            rszimg_width = func_info.get('rszimg_width')
            rszimg_maintain_aspect_ratio = func_info.get('rszimg_maintain_aspect_ratio')
            rszimg_use_cv = func_info.get('rszimg_use_cv')
        else:
            source_option = 'Specify Directory'
            rszimg_read_paths = './'
            rszimg_write_path = './'
            rszimg_height = 0,
            rszimg_width = 0,
            rszimg_maintain_aspect_ratio = True
            rszimg_use_cv = False
            func_info = {'rszimg_height': rszimg_height, 'rszimg_width': rszimg_width, 'rszimg_maintain_aspect_ratio': rszimg_maintain_aspect_ratio, 'rszimg_use_cv': rszimg_use_cv}

        if st.button('Add Resize Image'):
            if rszimg_read_paths == None:
                st.write('Specify Files before Add')
            else:
                selected_functions.append({'function': 'Resize Images', 'params': {'iter_no': 0, 'batch_size': 1, 'source_option': source_option, 'read_paths': rszimg_read_paths, 'write_path': rszimg_write_path, 'height': rszimg_height, 'width': rszimg_width, 'maintain_aspect_ratio': rszimg_maintain_aspect_ratio, 'use_cv': rszimg_use_cv}})
                st.session_state.selected_functions.append(selected_functions)


    elif tab == 'Augment Images':
        if 'settings_augimg' in st.session_state and st.session_state.settings_augimg['augimg_read_paths'] is not None:
            source_option = st.session_state.settings_augimg['source_option']
            augimg_read_paths = st.session_state.settings_augimg['augimg_read_paths']
            augimg_write_path = st.session_state.settings_augimg['augimg_write_path']
            func_info = st.session_state.settings_cnvtfmt['func_info']
            aug_op = func_info.get('aug_op')
            rotate = func_info.get('rotate')
            slide = func_info.get('slide')

            if not rotate is None:
                augimg_rotate_angle = rotate['augimg_rotate_angle']
                augimg_rotate_range = rotate['augimg_rotate_range']
                augimg_rotate_angle_tic = rotate['augimg_rotate_angle_tic']
            else:
                augimg_rotate_angle = 0
                augimg_rotate_range = 0
                augimg_rotate_angle_tic = 0

            if not slide is None:
                augimg_slide_range = slide['augimg_slide_range']
                augimg_slide_tic = slide['augimg_slide_tic']
            else:
                augimg_slide_range = [0,0]
                augimg_slide_tic = [0,0]
        else:
            source_option = 'Specify Directory'
            augimg_read_paths = './'
            augimg_write_path = './'
            aug_op = 'slide'
            augimg_rotate_angle = 0
            augimg_rotate_range = 0
            augimg_rotate_angle_tic = 0
            augimg_slide_range = [0,0]
            augimg_slide_tic = [0,0]
            func_info = {'aug_op': aug_op, 'rotate': {'augimg_rotate_angle': augimg_rotate_angle, 'augimg_rotate_range': augimg_rotate_range, 'augimg_slide_range': augimg_slide_range}}

        if st.button('Add Augment Image'):
            if augimg_read_paths == None:
                st.write('Specify Files before Add')
            else:
                selected_functions.append({'function': 'Augment Images', 'params': {'iter_no': 0, 'batch_size': 1, 'source_option': source_option, 'read_paths': augimg_read_paths, 'write_path': augimg_write_path, 'aug_op': aug_op, 'rotate_angle': augimg_rotate_angle, 'rotate_range': augimg_rotate_range, 'rotate_angle_tic': augimg_rotate_angle_tic, 'slide_range': augimg_slide_range, 'slide_tic': augimg_slide_tic}})
                st.session_state.selected_functions.append(selected_functions)


    elif tab == 'Augment Texts':
        if 'settings_augtxt' in st.session_state and st.session_state.settings_augtxt['augtxt_read_paths'] is not None:
            source_option = st.session_state.settings_augtxt['source_option']
            augtxt_read_paths = st.session_state.settings_augtxt['augtxt_read_paths']
            augtxt_write_path = st.session_state.settings_augtxt['augtxt_write_path']
            func_info = st.session_state.settings_cnvtfmt['func_info']
            op_sel = func_info.get('op_sel')
            augtxt_max_length = func_info.get('augtxt_max_length')

        else:
            source_option = 'Specify Directory'
            augtxt_read_paths = './'
            augtxt_write_path = './'
            op_sel = 'Remove Tags'
            augtxt_max_length = 100
            undesired_tokens_path = './'
            func_info = {'op_sel': op_sel, 'augtxt_max_length': augtxt_max_length, 'undesired_tokens_path': undesired_tokens_path}

        if st.button('Add Augment Texts'):
            if augtxt_read_paths == None:
                st.write('Specify Files before Add')
            else:
                selected_functions.append({'function': 'Augment Texts', 'params': {'source_option': source_option, 'augtxt_read_paths': augtxt_read_paths, 'augtxt_write_path': augtxt_write_path, 'op_sel': op_sel, 'augtxt_max_length': augtxt_max_length, 'undesired_tokens_path': undesired_tokens_path}})
                st.session_state.selected_functions.append(selected_functions)


    elif tab == 'Generate Normalized Images':
        if 'settings_nrmimg' in st.session_state and st.session_state.settings_nrmimg['nrmimg_read_paths'] is not None:
            source_option = st.session_state.settings_nrmimg['source_option']
            nrmimg_read_paths = st.session_state.settings_nrmimg['nrmimg_read_paths']
            nrmimg_write_path = st.session_state.settings_nrmimg['nrmimg_write_path']
        else:
            source_option = 'Specify Directory'
            nrmimg_read_paths = './'
            nrmimg_write_path = './'
            func_info = {}

        if st.button('Add Generate Normalized Images'):
            if nrmimg_read_paths == None:
                st.write('Specify Files before Add')
            else:
                selected_functions.append({'function': 'Generate Normalized Images', 'params':{'iter_no': 0, 'batch_size': 1, 'device': device, 'source_option': source_option, 'read_paths': nrmimg_read_paths, 'write_path': nrmimg_write_path}})
                st.session_state.selected_functions.append(selected_functions)


    elif tab == 'Generate Average Image':
        if 'settings_avgimg' in st.session_state and st.session_state.settings_avgimg['avgimg_read_paths'] is not None:
            source_option = st.session_state.settings_avgimg['source_option']
            avgimg_read_paths = st.session_state.settings_avgimg['avgimg_read_paths']
            avgimg_write_path = st.session_state.settings_avgimg['avgimg_write_path']
            func_info = st.session_state.settings_cnvtfmt['func_info']
            device = func_info.get('device')
            iter_no = func_info.get('iter_no')
        else:
            source_option = 'Specify Directory'
            avgimg_read_paths = './'
            avgimg_write_path = './'
            device = 'cpu'
            iter_no = 0
            func_info = {'device': device, 'iter_no': iter_no}

        if st.button('Add Generate Average Image'):
            if avgimg_read_paths == None:
                st.write('Specify Files before Add')
            else:
                selected_functions.append({'function': 'Generate Average Image', 'params':{'iter_no': 0, 'batch_size': 1, 'device': device, 'source_option': source_option, 'read_paths': avgimg_read_paths, 'write_path': avgimg_write_path}})
                st.session_state.selected_functions.append(selected_functions)


    #Save Sessions
    if st.button('Save Configuration'):
        with open('config_batch.json', 'w') as f:
            json.dump(st.session_state.selected_functions, f)

    #Load Sessions
    if st.button('Load Configuration'):
        try:
            with open('config_batch.json', 'r') as f:
                st.session_state.selected_functions = json.load(f)

        except FileNotFoundError:
            st.write('Configuration file not found.')


    #Processing Body
    st.write('Instructions:')

    if len(st.session_state.selected_functions) > 0:
        for i, function in enumerate(st.session_state.selected_functions):
            for j, func in enumerate(function):
                col1, col2 = st.columns([4, 1])
                col1.write(f'{func["function"]}')

                if col2.button(f'Delete {i}'):
                    del st.session_state.selected_functions[i]
                    st.experimental_rerun()
                break


        if st.button('Run'):
            for functions in st.session_state.selected_functions:
                for func in functions:
                    function_name = func['function']
                    function_params = func['params']

                    st.write(f'Running {function_name}')

                    if 'params' in function_params:
                        function_params = function_params['params']

                    if 'function' in function_params:
                        del function_params['function']

                    function_map[function_name](**function_params)


# End of run
torch.cuda.empty_cache()
total_memory = torch.cuda.get_device_properties(device).total_memory
allocated = torch.cuda.memory_allocated(device)
reserved_memory = torch.cuda.memory_reserved(device)
cached = torch.cuda.memory_reserved(device)

# Restore Configuration
if parameters is None:
    parameters = read_parameters('./config.txt')
