import cv2 as cv
import numpy as np
import os

from torch.utils.data import Dataset

def build_dataset(base="COMP9517 20T2 Group Project Image Sequences/DIC-C2DH-HeLa",
                  kernel_sz=5,
                  scale=1):
    # Initialize the kernel used for morphological operations
    ker = cv.getStructuringElement(cv.MORPH_CROSS,(kernel_sz, kernel_sz))
    # ker = np.ones((kernel_sz, kernel_sz), np.uint8)
    samples = list()
    
    # Build a list of samples
    for d in os.listdir(base):
        if "Mask" in d:
            d_n = base + '/' + d
            for f in os.listdir(d_n):
                f_n = d_n + '/' + f
                samples.append((f_n, f_n.replace(" Masks", "").replace("mask", "")))

    # Iterate through each sample and its mask
    for i, (mask_path, img_path) in enumerate(samples):
        # Load the image and its mask
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        mask = cv.imread(mask_path, cv.IMREAD_ANYDEPTH)

        seg = np.zeros(mask.shape, np.uint8)
        bor = np.zeros(mask.shape, np.uint8)
        dst_exp = np.zeros(mask.shape, np.uint8)

        # Iterate through unique values in the mask image
        for j in np.unique(mask):
            if j == 0:
                continue

            # Create an image with just that object, erode it to emphasize the 
            # border and add it to the current segmentation mask
            tmp = (mask == j).astype(np.uint8)

            dst = cv.distanceTransform(tmp, cv.DIST_L2, 3)
            dst = ((np.exp((np.log(256) / dst.max())) ** dst) - 1).astype(np.uint8)
            dst_exp += dst

            tmp = cv.erode(tmp, ker)
            seg += tmp

            # Dilate the same object mask, and add it to the border image
            tmp = cv.dilate((mask == j).astype(np.uint8), ker, iterations=3)
            bor += tmp
            
        # If any pixel has a value greater than 1, then at least 2 dilated 
        # objects occupy that space. Filter to get the border, then subtract the
        # object segmentation mask to get only the borders
        bor = (bor > 1).astype(np.uint8)
        bor *= (1 - seg)

        img = cv.resize(img, None, fx=scale, fy=scale) 
        # img = img - img.min()
        # img = img * (255 // img.max())
        seg = cv.resize(seg, None, fx=scale, fy=scale, 
                        interpolation=cv.INTER_NEAREST)
        bor = cv.resize(bor, None, fx=scale, fy=scale, 
                        interpolation=cv.INTER_NEAREST)
        dst_exp = cv.resize(dst_exp, None, fx=scale, fy=scale)

        # Write images to their respective folders
        cv.imwrite(f"data/img/{i}.png", img)
        cv.imwrite(f"data/seg/{i}.png", seg)
        cv.imwrite(f"data/bnd/{i}.png", bor)
        cv.imwrite(f"data/dst/{i}.png", dst_exp)
        
        
class CellDataset(Dataset):
    def __init__(self, preprocess=None, transform=None, normalize=True):
        # Store the path to each sample in a list
        self.samples = list()
        self.transform = transform
        self.preprocess = preprocess
        self.normalize = normalize
        
        for f in os.listdir("data/img"):
            self.samples.append(f)
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = cv.imread(f"data/img/{self.samples[idx]}", cv.IMREAD_GRAYSCALE)
        seg = cv.imread(f"data/seg/{self.samples[idx]}", cv.IMREAD_GRAYSCALE)
        bnd = cv.imread(f"data/bnd/{self.samples[idx]}", cv.IMREAD_GRAYSCALE)
        dst = cv.imread(f"data/dst/{self.samples[idx]}", cv.IMREAD_GRAYSCALE)
        msk = np.ones(img.shape, dtype=np.uint8)

        h, w = img.shape
        h = h - h % 16
        w = w - w % 16
        img = cv.resize(img, (w, h))
        seg = cv.resize(seg, (w, h))
        bnd = cv.resize(bnd, (w, h))
        dst = cv.resize(dst, (w, h))
        msk = cv.resize(msk, (w, h))
         
        if self.preprocess is not None:
            img = self.preprocess(img)

        # Store the mean and standard deviation to normalize the inputs later
        mean = img.mean()
        std = img.std()

        # Perform transformation before normalization, since borders will be
        # zero padded
        if self.transform is not None:
            img, seg, bnd, msk, dst = self.transform(img, seg, bnd, msk, dst)
            
        img = img.reshape((1, img.shape[0], img.shape[1]))
        seg = seg.reshape((1, seg.shape[0], seg.shape[1]))
        bnd = bnd.reshape((1, bnd.shape[0], bnd.shape[1]))
        msk = msk.reshape((1, msk.shape[0], msk.shape[1]))
        dst = dst.reshape((1, dst.shape[0], dst.shape[1]))

        img = img.astype(np.float32)
        seg = seg.astype(np.float32)
        bnd = bnd.astype(np.float32)
        msk = msk.astype(np.float32)
        dst = dst.astype(np.float32)

        ret =  {"img": img, "seg": seg, "bnd": bnd, "msk": msk, "dst": dst}

        if self.normalize:
            nrm = (img - mean) / std
            ret["nrm"] = nrm

        return ret


def random_warp(img, seg, bnd, msk, dst):
    h, w = img.shape

    # Modify the skew of the image
    ul = [0, 0]
    ur = [0, w]
    ll = [h, 0]
    lr = [h, w]
    pts1 = np.array([ul, ur, ll, lr], dtype=np.float32)

    ul += (np.random.randn(2)) * np.array([h / 16, w / 16])
    ur += (np.random.randn(2)) * np.array([h / 16, w / 16])
    ll += (np.random.randn(2)) * np.array([h / 16, w / 16])
    lr += (np.random.randn(2)) * np.array([h / 16, w / 16])
    pts2 = np.array([ul, ur, ll, lr], dtype=np.float32)

    M1 = cv.getPerspectiveTransform(pts1, pts2)

    # Randomly rotate the image
    c_y = np.random.randn() * h / 16 + h / 2
    c_x = np.random.randn() * w / 16 + w / 2
    a = np.random.rand() * 90

    M2 = np.identity(3)
    M2[0:2, 0:3] = cv.getRotationMatrix2D((c_y, c_x), a, 1)

    # Apply the transforms
    img = cv.warpPerspective(img, M2.dot(M1), (h, w))
    seg = cv.warpPerspective(seg, M2.dot(M1), (h, w))
    bnd = cv.warpPerspective(bnd, M2.dot(M1), (h, w))
    msk = cv.warpPerspective(msk, M2.dot(M1), (h, w))
    dst = cv.warpPerspective(dst, M2.dot(M1), (h, w))

    return (img, seg, bnd, msk, dst)


def random_flip(img, seg, bnd, msk, dst):
    n = np.random.randint(-1, 3)
    
    if (n != 2):
        img = cv.flip(img, n)
        seg = cv.flip(seg, n)
        bnd = cv.flip(bnd, n)
        msk = cv.flip(msk, n)
        dst = cv.flip(dst, n)

    return (img, seg, bnd, msk, dst)


def random_crop(img, seg, bnd, msk, dst, sz=288):
    h, w = img.shape

    ul_y = np.random.randint(0, h - sz)
    ul_x = np.random.randint(0, w - sz)
    
    img = img[ul_y:ul_y+sz, ul_x:ul_x+sz]
    seg = seg[ul_y:ul_y+sz, ul_x:ul_x+sz]
    bnd = bnd[ul_y:ul_y+sz, ul_x:ul_x+sz]
    msk = msk[ul_y:ul_y+sz, ul_x:ul_x+sz]
    dst = dst[ul_y:ul_y+sz, ul_x:ul_x+sz]

    return (img, seg, bnd, msk, dst)


def random_transform(img, seg, bnd, msk, dst, sz=288, warp=True):
    h, w = img.shape

    if warp:
        img, seg, bnd, msk, dst = random_warp(img, seg, bnd, msk, dst)

    img, seg, bnd, msk, dst = random_flip(img, seg, bnd, msk, dst)
    img, seg, bnd, msk, dst = random_crop(img, seg, bnd, msk, dst, sz=sz)

    return (img, seg, bnd, msk, dst)
