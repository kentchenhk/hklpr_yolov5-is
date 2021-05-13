"""
Crop the LP out of the whole image based on the given annotation.
"""
import os
from pathlib import Path
import glob
import cv2
import numpy as np
from tqdm import tqdm
import math
from utils.datasets import exif_size, box_candidates


def main(path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']

    # check all file paths
    try:
        f = []  # image files
        for p in path if isinstance(path, list) else [path]:
            p = str(Path(p))  # os-agnostic
            parent = str(Path(p).parent) + os.sep
            if os.path.isfile(p):  # file
                with open(p, 'r') as t:
                    t = t.read().splitlines()
                    f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
            elif os.path.isdir(p):  # folder
                f += glob.iglob(p + os.sep + '*.*')
            else:
                raise Exception('%s does not exist' % p)
        img_files = sorted(
            [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in img_formats])
    except Exception as e:
        raise Exception('Error loading data from %s: %s\n' % (path, e))

    n = len(img_files)
    assert n > 0, 'No images found in %s.' % (path)

    # Define labels
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    label_files = [x.replace(sa, sb, 1).replace(os.path.splitext(x)[-1], '.txt') for x in img_files]

    # Read and crop all files
    pbar = tqdm(zip(img_files, label_files), desc='Scanning images', total=len(img_files))
    for (img, label) in pbar:
        try:
            l = []
            image = cv2.imread(img)
            shape = image.shape  # image size
            assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
            if os.path.isfile(label):
                with open(label, 'r') as f:
                    l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
            if len(l) == 0:
                l = np.zeros((0, 5), dtype=np.float32)
        except Exception as e:
            print('WARNING: %s: %s' % (img, e))
        l = l[0][1:]
        x_left = round((l[0]-l[2]/2.0) * shape[1])
        x_right = round((l[0]+l[2]/2.0) * shape[1])
        y_up = round((l[1]-l[3]/2.0) * shape[0])
        y_down = round((l[1]+l[3]/2.0) * shape[0])
        img_crop = image[y_up:y_down, x_left:x_right, :]
        # print(os.path.join(out_path, os.path.split(img)[1]))
        cv2.imwrite(os.path.join(out_path, os.path.split(img)[1]), img_crop)


if __name__ == '__main__':
    # paths = ["train", "valid", "test"]
    # for i in paths:
    #     path = "/home/jrchan/data/LPR_datasets/hklp_mtr_detect/" + i + "/images"
    #     out_path = "/home/jrchan/data/LPR_datasets/hklp_mtr_recog_phesudo/" + i +"/images"
    #     main(path, out_path)
    #
    from utils.general import kmean_anchors

    kmean_anchors(path='data/lp_recognition.yaml', n=9, img_size=(416, 416), thr=0.20, gen=1000)




