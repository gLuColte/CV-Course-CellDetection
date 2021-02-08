# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 01:39:15 2020

@author: kanli
"""

import cv2
import numpy as np
import os
from os.path import isfile, join
import sys

#Define File Path
dataset_dir = input("Input Dataset Directory Name (Ensure Segmented Dir contains output images, started with 1 and *.png): ")
dataset_path = "RawData/" + dataset_dir 
#Find Direcotry in Dataset 
seq_dir_list = []
for _ in [x[0] for x in os.walk(dataset_path)][1:]:
    if "Masks" not in _:
        seq_dir_list.append(_.split("\\")[1])

for seq in seq_dir_list:
    path_img_dir = dataset_path + '/' + seq + '/'
    file_list = [f for f in os.listdir(path_img_dir) if isfile(join(path_img_dir, f))]
    img_array = []
    for file in file_list:
        file_path = path_img_dir + file
        ori_img = cv2.imread(file_path)
        #Find output
        output_path = file_path.replace('RawData', 'Results-BoundaryBox')
        output_img = cv2.imread(output_path)
        show_image = np.concatenate((ori_img, output_img), axis=1)
        # Path_img
        path_img_path = file_path.replace('RawData', 'Results-PathImage')
        path_img = cv2.imread(path_img_path)
        (height, width, layers) = ori_img.shape
        resized_path_img = cv2.resize(path_img, (width, height), interpolation=cv2.INTER_AREA)
            #Frame
        show_image = np.concatenate((show_image, resized_path_img), axis=1)

        height, width, layers = show_image.shape
        size = (width, height)
        img_array.append(show_image)
    out_video_name = path_img_dir.replace('RawData', 'Results-Video') + '/' + 'Out.avi' 
    out = cv2.VideoWriter(out_video_name,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
print('done')
    




