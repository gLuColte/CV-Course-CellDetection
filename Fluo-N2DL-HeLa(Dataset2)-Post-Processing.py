import numpy as np
import cv2, math, os, sys
from scipy import ndimage as ndi
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage import color
from skimage.feature import peak_local_max

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
dataset_dir = "Fluo-N2DL-HeLa"
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
        # Distance File Path
        dis_img_path = path_img.replace("(Raw)", "(DisT)")
        img_dis_gry = cv2.imread(dis_img_path, cv2.IMREAD_GRAYSCALE)

        marker = ndi.label(peak_local_max(img_dis_gry, indices=False, labels=img_dis_gry))[0]
        watersheded = watershed(-img_dis_gry, marker, mask=img_dis_gry)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        raw_out = cv2.morphologyEx(watersheded.astype(np.uint8), cv2.MORPH_ERODE, kernel, iterations=1)
        width = int(img_ori.shape[1])
        height = int(img_ori.shape[0])
        dimension = (width, height)
        out = cv2.resize(raw_out.astype(np.uint8), dimension)

        # Segmented Image Saving (Gray-Scale)
        seg_dir = "Segmented/" + dataset_dir + "/" + seq
        path_seg = seg_dir + '/' + file
        cv2.imwrite(path_seg, out)