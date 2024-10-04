import math

import numpy as np
import cv2
def stack(x):
    return np.uint8(np.stack([x]*3, axis=-1))
def calc_tau(x):
    output = 3 * x + 1
    return output
def minmax(x):
    return (x - x.min()) / (x.max() - x.min())
def get_taumap(pose_lst):
    save_imgs = []
    img_h, img_w = pose_lst[0].shape[:2]
    pivot = math.sqrt((img_h/2)**2 + (img_w/2)**2)
    canvas = np.ones((img_h, img_w)).astype(bool)
    for pose in pose_lst:
        bi_pose = np.all(pose == [0,0,0], axis=-1)
        canvas = np.logical_and(canvas, bi_pose)
    canvas = canvas.astype(np.uint8)
    dist_map = cv2.distanceTransform(canvas, cv2.DIST_L2, 0) / pivot
    tau_map = calc_tau(dist_map)
    return tau_map