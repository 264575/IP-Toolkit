import os
import cv2
import numpy as np

def shrpimg(iter_no, batch_size, source_option, read_paths, write_path, coefs, threshold):
    """
    Sharpen blurred image
    """

    extensions = ['png', 'jpg', 'jpeg', 'bmp', 'tiff']

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    coef_center = coefs[0]
    coef_neigbor = coefs[1]
    coef_diagonal = coefs[2]
    kernel = np.array([[coef_diagonal,coef_neigbor,coef_diagonal], [coef_neigbor,coef_center,coef_neigbor], [coef_diagonal,coef_neigbor,coef_diagonal]], np.float32)

    for filename in os.listdir(read_paths):
        if any(ext in filename for ext in extensions):
            filepath = os.path.join(read_paths, filename)

            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            p = cv2.Laplacian(img, cv2.CV_64F).var()

            if p < threshold:
                img = cv2.filter2D(img,-1,kernel)

            output_filepath = os.path.join(write_path, filename)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_filepath, img)
