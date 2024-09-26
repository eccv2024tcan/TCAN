#### CUDA_VISIBLE_DEVICES=0 python inference_folder.py --from_dir /home/nas4_user/jeonghokim/repos/AA/inputs/applications --image_folder_name source_image
import argparse
from glob import glob
import os
from os.path import join as opj
from tqdm import tqdm
import json
import numpy as np
import pickle

import cv2
import matplotlib.pyplot as plt

from annotator.dwpose import DWposeDetector

parser = argparse.ArgumentParser()
parser.add_argument("--from_dir", type=str)
parser.add_argument("--image_folder_name")
parser.add_argument("--p_idx", type=int, default=0)
parser.add_argument("--n_proc", type=int, default=1)
args = parser.parse_args()

file_lst = sorted(glob(opj(args.from_dir, args.image_folder_name, "*")))
n_block = len(file_lst) // args.n_proc + 1
file_lst = file_lst[args.p_idx * n_block : (args.p_idx+1) * n_block]
print(len(file_lst))
print(len(file_lst))
print(len(file_lst))

model = DWposeDetector()
for p in tqdm(file_lst, total=len(file_lst)):
    img = cv2.imread(p)
    out_img, pose = model(img)

    to_p = p.replace(args.image_folder_name, "dwpose")
    to_p = os.path.splitext(to_p)[0] + ".png"
    to_p_cond = p.replace(args.image_folder_name, "dwpose_dict")
    to_p_cond = os.path.splitext(to_p_cond)[0] + ".pkl"
    os.makedirs(os.path.dirname(to_p), exist_ok=True)
    os.makedirs(os.path.dirname(to_p_cond), exist_ok=True)

    plt.imsave(to_p, out_img)
    with open(to_p_cond, "wb") as f:
        pickle.dump(pose, f)
        