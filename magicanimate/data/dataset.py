import os
from os.path import join as opj
from glob import glob
import random

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from magicanimate.utils.util import zero_rank_print
from magicanimate.data.data_util import get_taumap

class TikTokImageDataset2(Dataset):
    def __init__(
            self,
            data_root_dir,
            sample_size=512,
            is_train=True,
            motion_type="densepose_rgb",
            use_HAP=False,
            HAP_root_dir="./DATA/HAP/train",
            tau0=0.5,
            use_augchibi=False,
            augchibi_root_dir = "./DATA/sd_augchibi",
            tau0_augchibi=0.5,
            ref_aug=None,
            motion_aug=None,
            margin=30,
            margin_augchibi=4,
    ):
        self.data_root_dir = opj(data_root_dir, "train" if is_train else "valid")
        self.sample_size = sample_size
        self.motion_type = motion_type
        self.use_HAP = use_HAP
        self.tau0 = tau0
        self.use_augchibi = use_augchibi
        self.tau0_augchibi = tau0_augchibi
        self.ref_aug = ref_aug
        self.motion_aug = motion_aug
        self.margin = margin
        self.margin_augchibi = margin_augchibi
    
        self.img_paths = []
        self.motion_paths = []
        folder_lst = os.listdir(self.data_root_dir)
        for folder in folder_lst:
            img_ps = sorted(glob(opj(self.data_root_dir, folder, "images/*")))
            motion_ps = sorted(glob(opj(self.data_root_dir, folder, f"{motion_type}/*")))
            assert len(img_ps) == len(motion_ps), f"image vs motion : {len(img_ps)} vs {len(motion_ps)}"
            self.img_paths.append(img_ps)
            self.motion_paths.append(motion_ps)
                   
        zero_rank_print(f"TikTok {len(self.img_paths)} images")
        
        self.transform = T.Compose([
            T.RandomResizedCrop(
                size=(sample_size, sample_size),
                scale=(0.9,1.0),
                ratio=(0.9,1.0),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            T.ToTensor(),
            T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
        ])
        self.motion_transform = T.Compose([
            T.RandomResizedCrop(
                size=(sample_size, sample_size),
                scale=(0.9, 1.0),
                ratio=(0.9, 1.0),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            T.ToTensor(),
        ])
        
        if use_HAP:
            zero_rank_print(f"HAP with tau0={self.tau0}")
            self.HAP_root_dir = HAP_root_dir
            self.HAP_img_dir = opj(HAP_root_dir, "images")
            self.HAP_motion_dir = opj(HAP_root_dir, self.motion_type)
            self.HAP_img_paths = []
            self.HAP_motion_paths = []
            self.HAP_path_init()
        if use_augchibi:
            zero_rank_print(f"augchibi with tau0={self.tau0_augchibi}")
            assert "pose" in motion_type, f"augchibi has only pose motion"
            self.augchibi_root_dir = augchibi_root_dir
            self.augchibi_img_paths = []
            self.augchibi_motion_paths = []
            self.augchibi_path_init()

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)
        
    def HAP_path_init(self):
        HAP_img_paths = []
        HAP_motion_paths = []
        with open(opj(self.HAP_root_dir, f"{self.motion_type}.txt"), 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_name, motion_name = line.strip().split()
                im, _ = os.path.splitext(img_name)
                mo, _ = os.path.splitext(motion_name)
                assert im == mo, f"image and motion name mush be equal! {im} vs {mo}"
                HAP_img_paths.append(opj(self.HAP_img_dir, img_name))
                HAP_motion_paths.append(opj(self.HAP_motion_dir, motion_name))
        combined = list(zip(HAP_img_paths, HAP_motion_paths))
        random.shuffle(combined)
        HAP_img_paths, HAP_motion_paths = zip(*combined)
        self.HAP_img_paths = list(HAP_img_paths)
        self.HAP_motion_paths = list(HAP_motion_paths)
        zero_rank_print(f"HAP path initialize with {len(self.HAP_img_paths)} images")

    def augchibi_path_init(self):
        augchibi_img_paths = []
        augchibi_motion_paths = []
        for folder in os.listdir(self.augchibi_root_dir):
            for subfolder in os.listdir(opj(self.augchibi_root_dir, folder)):
                img_paths = sorted(glob(opj(self.augchibi_root_dir, folder, subfolder, "grid/*")))
                motion_paths = sorted(glob(opj(self.augchibi_root_dir, folder, subfolder, "pose/*")))

                augchibi_img_paths.append(list(img_paths))
                augchibi_motion_paths.append(list(motion_paths))
        combined = list(zip(augchibi_img_paths, augchibi_motion_paths))
        random.shuffle(combined)
        augchibi_img_paths, augchibi_motion_paths = zip(*combined)
        self.augchibi_img_paths = list(augchibi_img_paths)
        self.augchibi_motion_paths = list(augchibi_motion_paths)
        zero_rank_print(f"augchibi path initialize with {len(self.augchibi_img_paths)} folders")

    def extract_path(self, img_ps, motion_ps):
        video_length = len(img_ps)
        margin = min(self.margin, video_length)
        ref_img_idx = random.randint(0, video_length - 1) 

        if video_length < 2*margin:
            trg_img_idx = (ref_img_idx + random.randint(1, video_length - 1))%video_length     
        else:
            trg_img_idx = (ref_img_idx + random.randint(margin, video_length - margin - 1))%video_length
        ref_p = img_ps[ref_img_idx]

        ref_motion_p = motion_ps[ref_img_idx]
        img_p = img_ps[trg_img_idx]
        motion_p = motion_ps[trg_img_idx]

        return ref_p, ref_motion_p, img_p, motion_p
        
    
    def extract_augchibi_path(self, img_ps, motion_ps):
        video_length = len(img_ps)
        margin = min(self.margin_augchibi, video_length)
        ref_img_idx = random.randint(0, video_length - 1)
        if ref_img_idx + margin < video_length:
            trg_img_idx = random.randint(ref_img_idx + margin, video_length-1)
        elif ref_img_idx - margin > 0:
            trg_img_idx = random.randint(0, ref_img_idx - margin)
        else:
            trg_img_idx = random.randint(0, video_length - 1)
        ref_p = img_ps[ref_img_idx]
        img_p = img_ps[trg_img_idx]
        motion_p = motion_ps[trg_img_idx]
        return ref_p, img_p, motion_p
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        '''
        img : [1,3,h,w], -1~1
        ref : [1,3,h,w], -1~1
        motion (img) : [1,3,h,w], 0~1
        '''
        # try:
        if self.use_HAP and self.use_augchibi: 
            raise NotImplementedError
        elif self.use_HAP:  # HAP
            tau = random.random()
            if tau <= self.tau0:
                img_p = self.HAP_img_paths.pop()
                ref_p = img_p
                motion_p = self.HAP_motion_paths.pop()
                ref_motion_p = motion_p
                if len(self.HAP_img_paths) == 0:
                    self.HAP_path_init()
            else:
                ref_p, ref_motion_p, img_p, motion_p = self.extract_path(self.img_paths[idx], self.motion_paths[idx])
        elif self.use_augchibi:  # augchibi
            tau = random.random()
            if tau <= self.tau0_augchibi:
                ref_p, img_p, motion_p = self.extract_augchibi_path(self.augchibi_img_paths.pop(), self.augchibi_motion_paths.pop())
                
                if len(self.augchibi_img_paths) == 0:
                    self.augchibi_path_init()
            else:            
                ref_p, ref_motion_p, img_p, motion_p = self.extract_path(self.img_paths[idx], self.motion_paths[idx])
        else:  
            ref_p, ref_motion_p, img_p, motion_p = self.extract_path(self.img_paths[idx], self.motion_paths[idx])

        state = torch.get_rng_state()
        ref =  Image.open(ref_p).convert("RGB")
        if self.ref_aug is None:
            ref = self.augmentation(ref, self.transform, state=state)[None]
        else:
            for aug_name in self.ref_aug:
                if aug_name == "resize":
                    ref = TF.resize(ref, self.sample_size, antialias=True)
                elif aug_name == "randomresize":
                    _size = random.randint(int(self.sample_size * 0.6), int(self.sample_size * 1.1))
                    ref = TF.resize(ref, _size, antialias=True)
                elif aug_name == "centercrop":
                    ref = TF.center_crop(ref, self.sample_size)
                elif aug_name == "randomcrop":
                    i,j,h,w = T.RandomCrop.get_params(ref, (self.sample_size, self.sample_size))
                    ref = TF.crop(ref, i,j,h,w)
                elif aug_name == "blur":
                    if random.random() < 0.5:
                        ref = TF.gaussian_blur(ref, kernel_size=7)
                else:
                    raise NotImplementedError(f"ref augmentation {aug_name} is not implemented")
            ref = TF.to_tensor(ref)
            ref = TF.normalize(ref, mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5])[None]

        img = Image.open(img_p).convert("RGB")
        motion = Image.open(motion_p).convert("RGB")
        ref_motion = Image.open(ref_motion_p).convert("RGB")
        if self.motion_aug is None:
            img = self.augmentation(img, self.transform, state=state)[None]
            motion = self.augmentation(motion, self.motion_transform, state=state)[None]
            ref_motion = self.augmentation(ref_motion, self.motion_transform, state=state)[None]
        else:
            for aug_name in self.motion_aug:
                if aug_name == "resize":
                    img = TF.resize(img, self.sample_size, antialias=True)
                    motion = TF.resize(motion, self.sample_size, antialias=True)
                elif aug_name == "randomresize":
                    img = TF.resize(img, _size, antialias=True)
                    motion = TF.resize(motion, _size, antialias=True)
                elif aug_name == "centercrop":
                    img = TF.center_crop(img, self.sample_size)
                    motion = TF.center_crop(motion, self.sample_size)
                elif aug_name == "randomcrop":
                    i,j,h,w = T.RandomCrop.get_params(img, (self.sample_size, self.sample_size))
                    img = TF.crop(img, i,j,h,w)
                    motion = TF.crop(motion, i,j,h,w)
                else:
                    raise NotImplementedError(f"ref augmentation {aug_name} is not implemented")    
            img = TF.to_tensor(img)
            img = TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5])[None]
            motion = TF.to_tensor(motion)[None]
            ref_motion = TF.to_tensor(ref_motion)[None]
        return {
            "img": img,
            "ref_img": ref,
            "ref_motion": ref_motion,
            "motion": motion,
            "text": ""
        }
    
class TikTokVideoDataset(Dataset):
    def __init__(
        self, 
        data_root_dir, 
        sample_size=512,
        length=16,
        is_train=True,
        motion_type="densepose_rgb",
    ):  
        
        self.data_root_dir = opj(data_root_dir, "train" if is_train else "valid")
        self.length = length
        folder_lst = os.listdir(self.data_root_dir)

        self.ref_paths = []
        self.img_paths = []
        self.motion_paths = []
        for folder in folder_lst:
            img_ps = sorted(glob(opj(self.data_root_dir, folder, "images/*")))
            motion_ps = sorted(glob(opj(self.data_root_dir, folder, f"{motion_type}/*")))
            assert len(img_ps) == len(motion_ps), f"image vs motion : {len(img_ps)} vs {len(motion_ps)}"
            if self.length <= len(img_ps):
                self.ref_paths.append(img_ps[0:1] * len(img_ps))
                self.img_paths.append(img_ps)
                self.motion_paths.append(motion_ps)
        zero_rank_print(f"{len(folder_lst)} videos")
        self.transform = T.Compose([
            T.RandomResizedCrop(
                (sample_size, sample_size),
                scale=(0.9, 1.0),
                ratio=(0.9, 1.0),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True
            ),
            T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
        ])
    
        self.motion_transform = T.Compose([
            T.RandomResizedCrop(
                (sample_size, sample_size),
                scale=(0.9, 1.0),
                ratio=(0.9, 1.0),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True
            ),
        ])

    def __len__(self):
        return len(self.img_paths)
    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)
    def __getitem__(self, idx):
        '''
        img : [f,3,h,w], -1~1
        ref : [1,3,h,w], -1~1
        motion (img) : [f,3,h,w], 0~1
        '''

        ref_path = self.ref_paths[idx][0]
        ref_motion_path = self.motion_paths[idx][0]    
        
        img_ps = self.img_paths[idx]
        motion_ps = self.motion_paths[idx]           


        video_length = len(img_ps)
        start_idx = random.randint(0, video_length-self.length)

        img_ps = img_ps[start_idx:start_idx+self.length]
        motion_ps = motion_ps[start_idx:start_idx+self.length]

        ref = torch.from_numpy(np.array(Image.open(ref_path).convert("RGB"))).permute(2,0,1) / 255.0
        ref_motion = torch.from_numpy(np.array(Image.open(ref_motion_path).convert("RGB"))).permute(2,0,1) / 255.0
        img = torch.from_numpy(np.stack([np.array(Image.open(p).convert("RGB")) for p in img_ps])).permute(0,3,1,2) / 255.0 
        motion = torch.from_numpy(np.stack([np.array(Image.open(p).convert("RGB")) for p in motion_ps])).permute(0,3,1,2) / 255.0

        state = torch.get_rng_state()
        ref = self.augmentation(ref, self.transform, state=state)[None]
        img = self.augmentation(img, self.transform, state=state)
        motion = self.augmentation(motion, self.motion_transform, state=state)
        ref_motion = self.augmentation(ref_motion, self.motion_transform, state=state)[None]

        return {
            "img": img,
            "ref_img": ref,
            "ref_motion": ref_motion,
            "motion": motion,
            "text": ""
        }