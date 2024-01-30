# hmnprs.py
# license: see LICENSE file
import os
import time
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import cv2

from utils.fileinfo import setimages


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rgb_to_id(color):
    """
    Composing ID from RGB Color
    """

    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def hmnprs(iter_no, batch_size, source_option, model, read_paths, write_path, face_clip):
    """
    Human Parsing
    """
    model.eval()
    
    if not os.path.exists(write_path):
        os.makedirs(write_path)
        
    img_list, original_file_names, original_file_paths = setimages(iter_no, batch_size, source_option, read_paths)
    
    if face_clip:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        batch_face_clips = []
        for index, pil_img in enumerate(img_list):
            img = np.array(pil_img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            face_clips = []
            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                face_clips.append(Image.fromarray(face))
                
            target_file_name = original_file_names[index]
            for i, face_img in enumerate(face_clips):
                output_path = os.path.join(write_path, f"{os.path.basename(target_file_name)}_face_{i}.png")
                face_img.save(output_path)
    else:
        preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_batch = [preprocess(Image.fromarray(img)) for img in img_list]
        input_batch = torch.stack(input_batch)

        if torch.cuda.is_available():
            input_batch = input_batch.to(device)
            model.to(device)

        with torch.no_grad():
            outputs = model(input_batch)['out']
    
        for idx, (result, file_name) in enumerate(zip(outputs, original_file_names)):

            parsed_image = result.cpu().numpy()
            unique_labels = np.unique(parsed_image)
            color_map = {label: np.random.randint(0, 255, size=(3,)) for label in unique_labels}
            parsed_image_id = rgb_to_id(parsed_image)

            colorful_map = np.zeros((parsed_image.shape[0], parsed_image.shape[1], 3), dtype=np.uint8)
            labels_and_colors = []

            for id, color in color_map.items():
                label = model.config.id2label[int(id)]

                mask = parsed_image_id == id

                for i in range(3):
                    colorful_map[mask, i] = color[i]

                labels_and_colors.append(f"ID: {id}, Label: {label}, Color: {color.tolist()}")

                single_color_image = np.zeros((parsed_image.shape[0], parsed_image.shape[1], 3), dtype=np.uint8)
                single_color_image[mask] = color
                single_color_img = Image.fromarray(single_color_image, 'RGB')
                single_color_img.save(os.path.join(write_path, f"{id}.png"))

            colorful_img = Image.fromarray(colorful_map, 'RGB')
            colorful_img.save(os.path.join(write_path, "colorful_map.png"))

            with open(os.path.join(write_path, f"{file_name.split('.')[0]}_labels_and_colors.txt"), "w") as f:
                for item in labels_and_colors:
                    f.write(f"{item}\n")
        del input_batch
        del outputs
        
    del img_list
    torch.cuda.empty_cache()
    time.sleep(2)
