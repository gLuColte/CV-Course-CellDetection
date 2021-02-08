import numpy as np
import cv2, math, os, sys
from scipy import ndimage as ndi
import matplotlib.pyplot as plt



def edge_mask(x):
    mask = np.ones(x.shape) * 255
    mask[x.ndim * (slice(1, -1),)] = 0
    return mask
def find_min_intensity(input_image):
    frequency_array = np.zeros(256)
    # Now find size of image
    (height, width) = input_image.shape
    for y in range(0, height):
        for x in range(0, width):
            frequency_array[input_image[y][x]] = frequency_array[input_image[y][x]] + 1
    for i in range(1, 255):  # We ignore 0 - Black
        if frequency_array[i] != 0:
            return i
# Define File Path
dataset_dir = "PhC-C2DL-PSC"
dataset_path = "Segmented/" + dataset_dir + "(Raw)"
# Find Direcotry in Dataset
seq_dir_list = []
for _ in [x[0] for x in os.walk(dataset_path)][1:]:
    if "Masks" not in _:
        seq_dir_list.append(_.split("\\")[1])

for seq in seq_dir_list:
    file_path = dataset_path + "/" + seq + "/"
    file_list = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    print(f"Begining processing {seq}...")

    for file in file_list:
        print(f"Processing {file}")
        # READ IMAGE-----------------------------------------------------------------------------------
        img_num = file.split(".")[0]
        # Original - define output image
        path_img = file_path + file
        img_ori_gry = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)



        # Raw Image Number Capturing
        raw_num = int(file.split(".")[0])-1
        if raw_num < 10:
            raw_img_num = "t00" + str(raw_num) + ".tif"
        elif raw_num >= 10 and raw_num < 100:
            raw_img_num = "t0" + str(raw_num) + ".tif"
        else:
            raw_img_num = "t" + str(raw_num) + ".tif"
        # Raw Image reading
        img_ori_path = file_path.replace("Segmented", "RawData").replace("(Raw)", "") + raw_img_num
        img_ori = cv2.imread(img_ori_path)
        '''
        kernel_1 = np.ones((3, 3), np.uint8)
        kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        close_img = cv2.morphologyEx(img_ori_gry, cv2.MORPH_CLOSE, kernel_1, iterations=1) # Erode only once

        # First Remove Tiny noises. They are more noticeable in the first 200 images
        if int(img_num) < 200:
            open_img = cv2.morphologyEx(close_img, cv2.MORPH_OPEN, kernel_2,iterations=2)  # This is consistent to remove tiny noise
        else:
            open_img = cv2.morphologyEx(close_img, cv2.MORPH_OPEN, kernel_2, iterations=1)  # This is consistent to remove tiny noise
        
        seg_processed = cv2.morphologyEx(open_img, cv2.MORPH_ERODE, kernel_1, iterations=1)
        processed_img = cv2.morphologyEx(seg_processed, cv2.MORPH_CLOSE, kernel_1, iterations=1)
        
        # First Remove Tiny noises
        processed_img_cleared = cv2.morphologyEx(open_img, cv2.MORPH_OPEN, kernel_1,iterations=1)  # This is consistent to remove tiny noise
        '''
        width = int(img_ori.shape[1])
        height = int(img_ori.shape[0])
        dim = (width, height)
        out = cv2.resize(img_ori_gry, dim, cv2.INTER_AREA)

        # Segmented Image Saving (Gray-Scale)
        seg_dir = "Segmented/" + dataset_dir + "/" + seq
        path_seg = seg_dir + '/' + file
        cv2.imwrite(path_seg, out)