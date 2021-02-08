import argparse
import cv2 as cv
import numpy as np
import os
import sys
import torch

from src.unet import UNetLite
from src.preprocess import preprocess_dic, preprocess_fluo, preprocess_phc


def predict(model_path="models/", input_path="data/predict", 
            out_path="output/predict", include_orig=True, prep_func=None,
            scale=2.0, cpu=False):
    '''Runs the given model on files in input_path and prints its output to output_path'''
    with torch.no_grad():
        # Select the device to run on
        if cpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load the model from a file
        net = UNetLite(scale=0.6, double=True)
        if os.path.isdir(model_path):
            load_path = model_path + max(os.listdir(model_path))
        else:
            load_path = model_path
            
        net.load_state_dict(torch.load(load_path))
        net.to(device)
        net.eval()
        
        curr = 0
        for f in sorted(os.listdir(input_path)):
            orig_img = cv.imread(f"{input_path}/{f}", cv.IMREAD_GRAYSCALE)

            orig_img = cv.resize(orig_img, None, fx=scale, fy=scale) 

            h, w = orig_img.shape
            h = h - (h % 16)
            w = w - (w % 16)
            orig_img = cv.resize(orig_img, (w, h))

            if prep_func is not None:
                img = prep_func(orig_img)
            else:
                img = orig_img

            img = img.reshape((1, 1, img.shape[0], img.shape[1]))
            img = img.astype(np.float32)
            img = (img - img.mean()) / img.std()
            img = torch.Tensor(img)
            
            output = net(img.to(device)).cpu()
            
            out_seg = output.numpy()
            out_seg = (out_seg > 0)[0, 0, :, :].astype(np.uint8)

            out_dst = output.numpy()[0, 1, :, :]
            out_dst[out_dst > 255] = 255
            out_dst[out_dst < 0] = 0
            out_dst = out_dst.astype(np.uint8)
    
            cmp = np.concatenate((orig_img, out_seg * 255, out_dst), axis=1)
            curr += 1
            # img = img.numpy()[0, 0, :, :].astype(np.uint8)

            if (include_orig): 
                ret = cmp
            else:
                ret = out * 255
            
            cv.imwrite(f"{out_path:3}/{curr}.png", ret)


def predict_img(orig_img, mode="dic"):
    with torch.no_grad():
        # Select the device to run on
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_path = f"models_{mode}/"
        prep_funcs = {"dic": preprocess_dic,
                      "phc": preprocess_phc,
                      "fluo" : preprocess_fluo}
        prep_func = prep_funcs[mode]
        scales = {"dic": 1.0,
                  "phc": 1.5,
                  "fluo" : 1.0}
        scale = scales[mode]

        # Load the model from a file
        net = UNetLite(scale=0.6, double=True)
        if os.path.isdir(model_path):
            load_path = model_path + max(os.listdir(model_path))
        else:
            load_path = model_path
            
        net.load_state_dict(torch.load(load_path))
        net.to(device)
        net.eval()

        orig_h, orig_w = orig_img.shape
        
        orig_img = cv.resize(orig_img, None, fx=scale, fy=scale) 

        h, w = orig_img.shape
        h = h - (h % 16)
        w = w - (w % 16)
        orig_img = cv.resize(orig_img, (w, h))

        if prep_func is not None:
            img = prep_func(orig_img)
        else:
            img = orig_img

        img = img.reshape((1, 1, img.shape[0], img.shape[1]))
        img = img.astype(np.float32)
        img = (img - img.mean()) / img.std()
        img = torch.Tensor(img)
        
        output = net(img.to(device)).cpu()
        
        out_seg = output.numpy()
        out_seg = (out_seg > 0)[0, 0, :, :].astype(np.uint8)

        out_dst = output.numpy()[0, 1, :, :]
        out_dst[out_dst > 255] = 255
        out_dst[out_dst < 0] = 0
        out_dst = out_dst.astype(np.uint8)

        orig_img = cv.resize(orig_img, (orig_w, orig_h))
        out_seg = cv.resize(out_seg, (orig_w, orig_h), interpolation=cv.INTER_NEAREST)
        out_dst = cv.resize(out_dst, (orig_w, orig_h))

        return orig_img, out_seg * 255, out_dst
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path', default=None,
                        help='Model file path')
    parser.add_argument('--input_path', default="data/predict",
                        help='Input images path')
    parser.add_argument('--output_path', default="output/predict",
                        help='Output images path')
    parser.add_argument('--output_only', action='store_true',
                        help='Flag to remove original images from the output \
                              file')
                         
    args = parser.parse_args()
    
    predict(args.model_path, 
            args.input_path, 
            args.output_path, 
            (not args.output_only), 
            prep_func=preprocess_phc)
