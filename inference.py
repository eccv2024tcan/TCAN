import argparse
import os
from os.path import join as opj
import pickle
from glob import glob
from collections import OrderedDict
import shutil

from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch
import torchvision.transforms as T
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file

from magicanimate.models.unet_controlnet import UNet3DConditionModel
from magicanimate.models.controlnet import ControlNetModel
from magicanimate.models.controlnet import ControlNet3DModel
from magicanimate.models.appearance_encoder import AppearanceEncoderModel
from magicanimate.pipelines.pipeline_animation import AnimationPipeline
from magicanimate.utils.util import zero_rank_print, model_load, save_videos_grid
from magicanimate.utils.videoreader import VideoReader

from dwpose.annotator.dwpose import DWposeDetector
from utils import get_retargeted

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_img", type=str, default="./inputs/webtoon_characters/source_images/director_kim_0001_01.png")
    parser.add_argument("--src_posedict", type=str, default="./inputs/webtoon_characters/openpose_dict/director_kim_0001_01.pkl")
    parser.add_argument("--driving_video", type=str, default="./inputs/webtoon_characters/driving_videos/orig_videos/running.mp4")
    parser.add_argument("--save_p", type=str, default="./result.mp4")

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--appearancenet_load_path", type=str, required=True)
    parser.add_argument("--controlnet_load_path", type=str, required=True)
    parser.add_argument("--unet_load_path", type=str, default=None)
    parser.add_argument("--load_unet_lora_weight", action="store_true")
    parser.add_argument("--use_temporal_controlnet", action="store_true")
    parser.add_argument("--use_temporal_taumap", action="store_true")
    args = parser.parse_args()
    return args


dwpose = DWposeDetector()

def get_openpose(img):
    pose, posedict = dwpose(img)
    return pose, posedict

def get_pipeline(args):
    config = OmegaConf.load(args.config)
    device = torch.device(f"cuda")

    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(config.noise_scheduler_kwargs))

    vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path)
    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_path, subfolder="text_encoder")
    unet = UNet3DConditionModel.from_pretrained_2d(config.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs))

    if args.load_unet_lora_weight:
        from peft import LoraConfig
        unet.requires_grad_(False)
        unet_lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            init_lora_weights="gaussian",
            target_modules=["attn1.to_k", "attn1.to_q", "attn1.to_v", "attn2.to_k", "attn2.to_q", "attn2.to_v"],  
        )  
        unet.add_adapter(unet_lora_config)
        zero_rank_print(f"initialize unet lora layer")

        m,u = unet.load_state_dict(model_load(args.unet_load_path), strict=False)
        zero_rank_print(f"unet is loaded from {args.unet_load_path}, {len(m)}, {len(u)}")

    appearance_encoder = AppearanceEncoderModel.from_config(OmegaConf.to_container(OmegaConf.load(config.appearance_encoder_config_path)))
    appearance_encoder.reset_final_block()
    if args.appearancenet_load_path.endswith(".safetensors"):
        load_state_dict = load_file(args.appearancenet_load_path)
    else:
        tmp_state_dict = torch.load(args.appearancenet_load_path, map_location="cpu")
        load_state_dict = OrderedDict()
        for k, v in tmp_state_dict.items():
            load_state_dict[k.replace("module.", "")] = v
    missing, unexpected = appearance_encoder.load_state_dict(load_state_dict)
    zero_rank_print(f"AppearanceNet is loaded from {args.appearancenet_load_path}, missing / unexpected / model / load_state_dict : {len(missing)} / {len(unexpected)} / {len(appearance_encoder.state_dict())} / {len(load_state_dict)}")
    
    if args.use_temporal_controlnet:
        zero_rank_print(f"use_temporal_controlnet True")
        
        ccc = OmegaConf.to_container(OmegaConf.load(config.controlnet_config_path))
        ccc.pop('down_block_types')
        controlnet = ControlNet3DModel(
            unet_use_cross_frame_attention=config.unet_additional_kwargs.unet_use_cross_frame_attention, 
            unet_use_temporal_attention=config.unet_additional_kwargs.unet_use_temporal_attention,
            use_motion_module = True,
            motion_module_type='Vanilla',            
            motion_module_kwargs = unet.motion_module_kwargs,
            **ccc)  
        
        if config.get("motion_module", None) is not None:
            m, u = controlnet.load_state_dict(torch.load(config.motion_module, map_location="cpu"), strict=False)
            print(f'missing: {len(m)}, unknown: {len(u)}')

        m, u = controlnet.load_state_dict(model_load(args.controlnet_load_path), strict=False)

        zero_rank_print(f"ControlNet is loaded from {args.controlnet_load_path}, {len(m)}, {len(u)}")
    else:
        controlnet = ControlNetModel.from_config(OmegaConf.to_container(OmegaConf.load(config.controlnet_config_path)))
        if args.controlnet_load_path.endswith(".safetensors"):
            load_state_dict = load_file(args.controlnet_load_path)
        else:
            tmp_state_dict = torch.load(args.controlnet_load_path, map_location="cpu")
            load_state_dict = OrderedDict()
            for k, v in tmp_state_dict.items():
                load_state_dict[k.replace("module.", "")] = v
        missing, unexpected = controlnet.load_state_dict(load_state_dict)
        zero_rank_print(f"ControlNet is loaded from {args.controlnet_load_path}, missing / unexpected / model / load_state_dict : {len(missing)} / {len(unexpected)} / {len(controlnet.state_dict())} / {len(load_state_dict)}")

    if args.unet_load_path is not None:
        _load_state_dict = torch.load(args.unet_load_path, map_location="cpu")
        if "state_dict" in _load_state_dict.keys(): _load_state_dict = _load_state_dict["state_dict"]

        load_state_dict = OrderedDict()
        for k, v in _load_state_dict.items():
            if "motion_modules" in k:
                load_state_dict[k.replace("module.", "")] = v

        missing, unexpected = unet.load_state_dict(load_state_dict, strict=False)
        zero_rank_print(f"motion module is loaded from {args.unet_load_path}, missing / unexpected / model / load_state_dict : {len(missing)} / {len(unexpected)} / {len(unet.state_dict())} / {len(load_state_dict)}")

    unet.enable_xformers_memory_efficient_attention()
    appearance_encoder.enable_xformers_memory_efficient_attention()
    controlnet.enable_xformers_memory_efficient_attention()
    
    vae.to(torch.float16)
    unet.to(torch.float16)
    text_encoder.to(torch.float16)
    appearance_encoder.to(torch.float16)
    controlnet.to(torch.float16)

    vae.to(device)
    unet.to(device)
    text_encoder.to(device)
    appearance_encoder.to(device)
    controlnet.to(device)

    vae.eval()
    unet.eval()
    text_encoder.eval()
    appearance_encoder.eval()
    controlnet.eval()

    validation_pipeline = AnimationPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=noise_scheduler,        
    )
    
    validation_pipeline.to(device)
    return validation_pipeline, appearance_encoder

def data_prepare(config, src_img_p, src_posedict_p, dri_video_p):
    src_img = cv2.imread(src_img_p)
    cap = cv2.VideoCapture(dri_video_p)
    to_posedict_dir = os.path.splitext(dri_video_p)[0]
    os.makedirs(to_posedict_dir, exist_ok=True)

    dri_img_lst = []
    idx = 0
    while True:
        ret, f = cap.read()
        if not ret: break
        dri_img_lst.append(f)        
        pose, posedict = get_openpose(f)
        posedict_p = opj(to_posedict_dir, f"{idx:06d}.pkl")
        with open(posedict_p, "wb") as f:
            pickle.dump(posedict, f)
        idx += 1

    dri_posedict_ps = sorted(glob(opj(to_posedict_dir, "*")))
    
    dri_pose_lst = []
    for dri_img, dri_posedict_p in tqdm(
        zip(dri_img_lst, dri_posedict_ps), 
        total=len(dri_img_lst), 
        ncols=75, 
        desc=f"retarget {len(dri_img_lst)} poses"
    ):
        crop_src_img, crop_src_pose, crop_dri_pose = get_retargeted(
            src_img=src_img,
            src_pkl_p=src_posedict_p,
            dri_img=dri_img,
            dri_pkl_p=dri_posedict_p,
            bbox=None,
            first_dri_pkl_p=dri_posedict_ps[0],
            pose_crop=True,
        )

        dri_pose_lst.append(crop_dri_pose)

    src_img_ext = os.path.splitext(src_img_p)[1]
    crop_src_img_p = src_img_p.replace(src_img_ext, f"_crop{src_img_ext}")
    cv2.imwrite(crop_src_img_p, crop_src_img)    

    dri_video_ext = os.path.splitext(dri_video_p)[1]
    retargetd_dri_video_p = dri_video_p.replace(dri_video_ext, f"_retargeted{dri_video_ext}")
    img_h, img_w = dri_pose_lst[0].shape[:2]
    writer = cv2.VideoWriter(retargetd_dri_video_p, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS), (img_w, img_h))
    for f in dri_pose_lst:
        writer.write(f)
    writer.release()

    
    size = config.size
    crop_src_img = np.array(Image.open(crop_src_img_p).convert("RGB").resize((size, size)))
    retarget_dri_video = VideoReader(retargetd_dri_video_p).read()
    if retarget_dri_video[0].shape[0] != size:
        retarget_dri_video = [np.array(Image.fromarray(c).resize((size, size))) for c in retarget_dri_video]
        retarget_dri_video = np.array(retarget_dri_video)
        
    #### delete >>>>
    os.remove(crop_src_img_p)
    os.remove(retargetd_dri_video_p)
    shutil.rmtree(to_posedict_dir)
    #### delete <<<<
    return crop_src_img, retarget_dri_video


def generate(args, src_img_p, src_json_p, driving_video_p):
    config = OmegaConf.load(args.config)
    source_image, control = data_prepare(config, src_img_p, src_json_p, driving_video_p)
    pipeline, appearance_encoder = get_pipeline(args)

    H, W, C = source_image.shape
    original_length = control.shape[0]
    if control.shape[0] % config.L > 0:
        control = np.pad(control, ((0, config.L-control.shape[0] % config.L), (0, 0), (0, 0), (0, 0)), mode='edge')
    generator = torch.Generator(device=torch.device("cuda:0"))
    generator.manual_seed(torch.initial_seed())

    with torch.autocast("cuda"):
        sample = pipeline(
            prompt                  = "",
            negative_prompt         = "",
            num_inference_steps     = config.steps,
            guidance_scale          = config.guidance_scale,
            width                   = W,
            height                  = H,
            video_length            = len(control),
            controlnet_condition    = control,
            init_latents            = None,
            generator               = generator,
            num_actual_inference_steps = config.get("num_actual_inference_steps", config.steps),
            appearance_encoder      = appearance_encoder,
            source_image            = source_image,
            use_temporal_taumap     = args.use_temporal_taumap,            
            use_temporal_controlnet = args.use_temporal_controlnet
        ).videos

    sample = sample[:, :, :original_length]
    return sample
    
    
if __name__ == "__main__":
    args = parse_args()
    sample = generate(
        args=args,
        src_img_p=args.src_img,
        src_json_p=args.src_posedict,
        driving_video_p=args.driving_video,
    )
    save_videos_grid(sample, args.save_p)

