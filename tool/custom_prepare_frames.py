# Sort by Disco file names and save it as pkl.

import os
from os.path import join as opj
from glob import glob
import cv2
from os.path import join as opj
import argparse
from tqdm import tqdm
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--video_dir", type=str, required=True)
parser.add_argument("--img_h", type=int, default=512)
parser.add_argument("--img_w", type=int, default=512)
args = parser.parse_args()
if args.video_dir[-1] == "/":
    args.video_dir = args.video_dir[:-1]

with open("./tool/tiktok_valid_path_lst.pkl", "rb") as f:
    disco_tiktok_path_lst = pickle.load(f)
video_ps = sorted(glob(opj(args.video_dir, "*")))
video_ps = [x for x in video_ps if "grid" not in x]
n_save = 0
n_skip = 0
for p in tqdm(video_ps, total=len(video_ps), desc="frame prepare", ncols=70):
    if "grid" in p: continue
    fn = os.path.splitext(os.path.basename(p))[0]

    if fn[0] == "0":
        video_name = fn[:5]
    elif fn[0] == "2":
        video_name = fn[:11]
    else:
        raise NotImplementedError
    prefix = "TiktokDance_" + video_name
    cap = cv2.VideoCapture(p)
    to_dir = opj("/".join(args.video_dir.split("/")[:-1]), "merged_frames")
    os.makedirs(to_dir, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        to_p = opj(to_dir, disco_tiktok_path_lst.pop(0))
        img_h, img_w = frame.shape[:2]
        if img_h != args.img_h or img_w != args.img_w:
            frame = cv2.resize(frame, (args.img_w, args.img_h))
        if not os.path.exists(to_p):
            cv2.imwrite(to_p, frame)
            n_save += 1
        else:
            n_skip += 1
print(f"# of skip : {n_skip}, # of save : {n_save}")
n_total = n_save + n_skip
assert n_total == 4123, f"total : {n_total} != 4123, there are some missing frames to predict!!!"