import os
from os.path import join as opj
import argparse
from omegaconf import OmegaConf
import datetime
from contextlib import nullcontext
from importlib import import_module
from datetime import timedelta
from einops import rearrange

import torch
import torch.distributed as dist
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from magicanimate.utils.util import zero_rank_print, AverageMeter, save_videos_grid, model_load, save_args, save_config, none_filter
from magicanimate.models.unet_controlnet import UNet3DConditionModel
from magicanimate.models.controlnet import ControlNetModel
from magicanimate.models.controlnet import ControlNet3DModel
from magicanimate.models.appearance_encoder import AppearanceEncoderModel
from magicanimate.models.mutual_self_attention import ReferenceAttentionControl
from magicanimate.pipelines.pipeline_animation import AnimationPipeline
from magicanimate.utils.videoreader import VideoReader
from utils import convert_ldm_unet_checkpoint

from magicanimate.models.channel_fixer import ChannelFixer
from magicanimate.models.custom_attention_processor import CustomAttnProcessor, CustomXFormersAttnProcessor
from datetime import timedelta

def get_controlnet_output(controlnet, noisy_l_img, timesteps, text_embeddings, video_length, motion, freeze_controlnet, use_temporal_controlnet=False):
    if not freeze_controlnet:
        down_block_res_samples, mid_block_res_sample = controlnet(
            rearrange(noisy_l_img, "b c f h w -> (b f) c h w") if not use_temporal_controlnet else noisy_l_img,
            timesteps,
            encoder_hidden_states=text_embeddings.repeat_interleave(video_length, 0),
            controlnet_cond=motion if use_temporal_controlnet else rearrange(motion, "b f c h w -> (b f) c h w"),                            
            conditioning_scale=1.0,
            return_dict=False
        )
    else:
        with torch.no_grad():
            noisy_l_img = torch.ones_like(noisy_l_img)
            timesteps = torch.ones_like(timesteps)
            text_embeddings= torch.ones_like(text_embeddings)
            motion = torch.ones_like(motion)
            down_block_res_samples, mid_block_res_sample = controlnet(
                rearrange(noisy_l_img, "b c f h w -> (b f) c h w") if not use_temporal_controlnet else noisy_l_img,
                timesteps,
                encoder_hidden_states=text_embeddings.repeat_interleave(video_length, 0),
                controlnet_cond=motion if use_temporal_controlnet else rearrange(motion, "b f c h w -> (b f) c h w"),                            
                conditioning_scale=1.0,
                return_dict=False
            )

    _down_block_res_samples = []
    for down_sample in down_block_res_samples:
        if not use_temporal_controlnet:
            down_sample = rearrange(down_sample, "(b f) c h w -> b c f h w", f=video_length)
        
        _down_block_res_samples.append(down_sample)
    down_block_res_samples = _down_block_res_samples

    return down_block_res_samples, mid_block_res_sample

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/reimple_first_stage.yaml")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--motion_type", type=str, default="dwpose", choices=["densepose_rgb", "dwpose"])
    parser.add_argument("--is_second_stage", action="store_true")
    parser.add_argument("--freeze_controlnet", action="store_true")
    parser.add_argument("--freeze_ae", action="store_true")
    parser.add_argument("--use_HAP", action="store_true")
    parser.add_argument("--use_augchibi", action="store_true")
    parser.add_argument("--ref_aug", nargs="+", type=str, choices=["resize", "randomresize", "centercrop", "randomcrop", "blur"])
    parser.add_argument("--motion_aug", nargs="+", type=str, choices=["resize", "randomresize", "centercrop", "randomcrop", "blur"])
    parser.add_argument("--image_dataset_name", type=str, default="TikTokImageDataset2")
    parser.add_argument("--video_dataset_name", type=str, default="TikTokVideoDataset")
    parser.add_argument("--pretrained_vae_path", type=str, default=None)
    parser.add_argument("--pretrained_appearance_encoder_path", type=str, default=None)
    parser.add_argument("--pretrained_controlnet_path", type=str, default=None)
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--pretrained_unet_path", type=str, default=None)
    parser.add_argument("--init_unet_lora", action="store_true")
    parser.add_argument("--load_unet_lora_weight", action="store_true", help="lora weight를 load할때는 1. init lora 2. load / lora weight가 아니면 1. load 2. init lora")
    parser.add_argument("--use_learnable_taumap", action="store_true")
    parser.add_argument("--unet_lora_midup", action="store_true")
    parser.add_argument("--unet_rank", type=int, default=16)
    parser.add_argument("--ae_rank", type=int, default=16)
    parser.add_argument("--use_temporal_controlnet", action="store_true")
    parser.add_argument("--use_noisy_ref_img", action="store_true")
    parser.add_argument("--save_name", type=str, default="dummy") 

    parser.add_argument("--n_epochs", type=int, default=1251)
    parser.add_argument("--save_model_iter_freq", type=int, default=8750)
    parser.add_argument("--validation_step", type=int, default=8750)
    parser.add_argument("--n_iters", type=int, default=100000)
    parser.add_argument("--size", type=int, default=None)

    parser.add_argument("--data_root_dir", type=str, default="./DATA/TikTok")
    parser.add_argument("--HAP_root_dir", type=str, default="./DATA/HAP/train")
    parser.add_argument("--augchibi_root_dir", type=str, default="./DATA/sd_augchibi")
    parser.add_argument("--save_root_dir", type=str, default="./logs")
    parser.add_argument("--validation_step_lst", type=int, nargs="+", default=[10000000])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    args.save_dir = opj(args.save_root_dir, f"{datetime.datetime.now().strftime('%Y%m%d')}_" + args.save_name)
    args.sample_save_dir = opj(args.save_dir, "samples")
    args.model_save_dir = opj(args.save_dir, "models")
    args.tb_save_dir = opj(args.save_dir, "tb")
    args.args_save_path = opj(args.save_dir, "args.json")
    args.config_save_path = opj(args.save_dir, "config.yaml")
    os.makedirs(args.sample_save_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.tb_save_dir, exist_ok=True)

    if args.is_second_stage:
        args.validation_step_lst = [3]
        args.save_model_iter_freq = 2500
        args.validation_step = 2500
        args.n_epochs = 90
    
    args.n_gpus = torch.cuda.device_count()
    if args.n_gpus > 1: 
        args.DDP = True
    else:
        args.DDP = False
    if args.DDP:
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=7200000))
        args.num_processes = dist.get_world_size()
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.num_processes = 1
        args.local_rank = 0
    torch.cuda.set_device(args.local_rank)
    return args
    
def main(args):
    is_main_process = args.local_rank == 0
    config = OmegaConf.load(args.config)
    if args.size is not None: 
        config.size = args.size

    device = torch.device(f"cuda:{args.local_rank}")
    if args.is_second_stage:
        config.unet_additional_kwargs.motion_module_kwargs.use_learnable_taumap = args.use_learnable_taumap
    
    if is_main_process:
        tb_writer = SummaryWriter(args.tb_save_dir)

    if args.pretrained_vae_path is not None:
        zero_rank_print(f"replace vae path {config.pretrained_vae_path} to {args.pretrained_vae_path}")
        config.pretrained_vae_path = args.pretrained_vae_path
    if args.pretrained_model_path is not None:
        zero_rank_print(f"replace sd path {config.pretrained_model_path} to {args.pretrained_model_path}")
        config.pretrained_model_path = args.pretrained_model_path
    if args.pretrained_controlnet_path is not None:
        zero_rank_print(f"replace cnet path {config.pretrained_controlnet_path} to {args.pretrained_controlnet_path}")
        config.pretrained_controlnet_path = args.pretrained_controlnet_path
        zero_rank_print(f"replace {config.pretrained_controlnet_path} to {args.pretrained_controlnet_path}")
    if args.pretrained_appearance_encoder_path is not None:
        zero_rank_print(f"replace appearance encoder path {config.pretrained_appearance_encoder_path} to {args.pretrained_appearance_encoder_path}")
        config.pretrained_appearance_encoder_path = args.pretrained_appearance_encoder_path

    if not args.is_second_stage:
        train_dataset = getattr(import_module("magicanimate.data.dataset"), args.image_dataset_name)(
            args.data_root_dir,
            sample_size=config.size,
            motion_type=args.motion_type,
            use_HAP=args.use_HAP,
            HAP_root_dir=args.HAP_root_dir,
            tau0=config.get("tau0", 0.5),
            use_augchibi=args.use_augchibi,
            augchibi_root_dir = args.augchibi_root_dir,
            tau0_augchibi=config.get("tau0_chibi", 0.5),
            ref_aug=args.ref_aug,
            motion_aug=args.motion_aug,
        )
    else:
        train_dataset = getattr(import_module("magicanimate.data.dataset"), args.video_dataset_name)(
            args.data_root_dir,
            sample_size=config.size,
            length=config.L,
            motion_type=args.motion_type,
        )
    if args.DDP:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.num_processes,
            rank=args.local_rank,
            shuffle=True,
            seed=args.seed
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=none_filter,
    )

    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(config.noise_scheduler_kwargs))

    vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path)
    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_path, subfolder="text_encoder")
    unet = UNet3DConditionModel.from_pretrained_2d(config.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs))

    if (args.pretrained_unet_path is not None) and (not args.load_unet_lora_weight):
        from utils import convert_ldm_unet_checkpoint
        renew_cp = convert_ldm_unet_checkpoint(load_file(args.pretrained_unet_path), unet.config)
        unet.load_state_dict(renew_cp)
        zero_rank_print(f"unet is loaded from {args.pretrained_unet_path}")

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
        
        if config.get("motion_module", None) is not None and args.is_second_stage:
            m, u = controlnet.load_state_dict(torch.load(config.motion_module, map_location="cpu"), strict=False)
            print(f'temporal controlnet missing: {len(m)}, unexpected: {len(u)}')

        m, u = controlnet.load_state_dict(model_load(config.pretrained_controlnet_path), strict=False)
        zero_rank_print(f"ControlNet is loaded from {config.pretrained_controlnet_path}, missing: {len(m)}, unexpected: {len(u)}")
    else:
        controlnet = ControlNetModel.from_config(OmegaConf.to_container(OmegaConf.load(config.controlnet_config_path)))
        m, u = controlnet.load_state_dict(model_load(config.pretrained_controlnet_path), strict=False)
        assert len(m) == 42 or len(m) == 0, f"missing {len(m)} : there are additional missing layers except of  initial layer (named controlnet_cond_embedding.) and zero conv"
        zero_rank_print(f"ControlNet is loaded from {config.pretrained_controlnet_path}, missing: {len(m)}, unexpected: {len(u)}")
    
    appearance_encoder = AppearanceEncoderModel.from_config(OmegaConf.to_container(OmegaConf.load(config.appearance_encoder_config_path)))
    

    reference_control_writer = ReferenceAttentionControl(
        appearance_encoder, 
        do_classifier_free_guidance=True, 
        mode="write", 
        fusion_blocks=config.fusion_blocks, 
        batch_size=args.batch_size,
        is_train=True, 
        is_second_stage=args.is_second_stage,
    )
    reference_control_reader = ReferenceAttentionControl(
        unet, 
        do_classifier_free_guidance=True, 
        mode="read", 
        fusion_blocks=config.fusion_blocks, 
        batch_size=args.batch_size, 
        is_train=True,
        is_second_stage=args.is_second_stage
    )

    if config.get("motion_module", None) is not None and args.is_second_stage:
        m, u = unet.load_state_dict(torch.load(config.motion_module, map_location="cpu"), strict=False)
        assert len(u) == 0, f"If loading pretrained motion module, temporal layer structure must be same to the AnimateDiff!"
        zero_rank_print(f"Motion module is loaded from {config.motion_module}, missing: {len(m)}, unexpected: {len(u)}")
    
    unet_lora_module_names = [
        "attn1.to_q", 
        "attn1.to_k", 
        "attn1.to_v", 
        "attn2.to_q", 
        "attn2.to_k", 
        "attn2.to_v"
    ]

    if args.load_unet_lora_weight: 
        from peft import LoraConfig
        unet.requires_grad_(False)
        unet_lora_config = LoraConfig(
            r=args.unet_rank,
            lora_alpha=args.unet_rank,
            init_lora_weights="gaussian",
            target_modules=unet_lora_module_names,  
        )  
        unet.add_adapter(unet_lora_config)
        zero_rank_print(f"initialize unet lora layer")

        m,u = unet.load_state_dict(model_load(args.pretrained_unet_path), strict=not args.is_second_stage)
        zero_rank_print(f"unet is loaded from {args.pretrained_unet_path}, missing: {len(m)}, unexpected: {len(u)}")
    else: 
        if args.init_unet_lora:
            from peft import LoraConfig
            unet.requires_grad_(False)
            unet_lora_config = LoraConfig(
                r=args.unet_rank,
                lora_alpha=args.unet_rank,
                init_lora_weights="gaussian",
                target_modules=unet_lora_module_names,
            )  
            unet.add_adapter(unet_lora_config)
            zero_rank_print(f"initialize unet lora layer")

    if "civitai" in config.pretrained_appearance_encoder_path.lower():
        renew_cp = convert_ldm_unet_checkpoint(load_file(args.pretrained_unet_path), unet.config)
        m, u = appearance_encoder.load_state_dict(renew_cp, strict=False)
    else:
        m, u = appearance_encoder.load_state_dict(model_load(config.pretrained_appearance_encoder_path), strict=False)
    zero_rank_print(f"Appearance Encoder is loaded from {config.pretrained_appearance_encoder_path}, missing: {len(m)}, unexpected: {len(u)}")
    appearance_encoder.reset_final_block() 
    
    unet.enable_xformers_memory_efficient_attention()
    controlnet.enable_xformers_memory_efficient_attention()
    appearance_encoder.enable_xformers_memory_efficient_attention()

    vae.to(device)
    unet.to(device)
    text_encoder.to(device)
    controlnet.to(device)
    appearance_encoder.to(device)

    zero_rank_print("\n#### optimizer >>>>")
    trainable_params = []
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    if not args.is_second_stage: 
            
        if args.init_unet_lora:
            trainable_params += list(filter(lambda p: p.requires_grad, unet.parameters()))
            zero_rank_print(f"unet lora {len(list(filter(lambda p: p.requires_grad, unet.parameters())))} layers are added")
        else:
            zero_rank_print(f"no unet lora")
       
        if not args.freeze_ae:
            trainable_params += list(filter(lambda p: p.requires_grad, appearance_encoder.parameters()))
            zero_rank_print(f"appearance encoder {len(list(filter(lambda p: p.requires_grad, appearance_encoder.parameters())))} layers are added")
        else:
            zero_rank_print(f"freeze appearance encoder")
    

        if not args.freeze_controlnet:
            if args.use_temporal_controlnet:
                for key, param in controlnet.named_parameters():
                    if 'motion_modules.' in key:
                        param.requires_grad = False
                trainable_params += list(filter(lambda p: p.requires_grad, controlnet.parameters()))
            else:
                trainable_params += list(controlnet.parameters())        
            zero_rank_print(f"controlnet {len(list(controlnet.parameters()))} layers are added")
        else:
            zero_rank_print(f"freeze controlnet")

    else:        
        appearance_encoder.requires_grad_(False)
        controlnet.requires_grad_(False)
        unet.requires_grad_(False)
        for key, param in unet.named_parameters():
            if "motion_modules." in key:
                param.requires_grad = True
        for key, param in controlnet.named_parameters():
            if "motion_modules." in key:
                param.requires_grad = True

        trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
        trainable_params += list(filter(lambda p: p.requires_grad, controlnet.parameters()))
        zero_rank_print(f"temporal controlnet {len(list(filter(lambda p: p.requires_grad, controlnet.parameters())))} temporal layers are added")
        zero_rank_print(f"unet {len(list(filter(lambda p: p.requires_grad, unet.parameters())))} temporal layers are added")
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon
    ) 
    zero_rank_print("#### optimizer <<<<\n")

    lr_scheduler = get_scheduler( 
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.n_epochs * len(train_loader)
    )

    unet.enable_xformers_memory_efficient_attention()
    controlnet.enable_xformers_memory_efficient_attention()
    appearance_encoder.enable_xformers_memory_efficient_attention()

    for n, m in unet.named_modules():
        if ('up_blocks' in n) and (n.endswith('attn1')):
            m.processor = CustomXFormersAttnProcessor()

    vae.to(device)
    unet.to(device)
    text_encoder.to(device)
    controlnet.to(device)
    appearance_encoder.to(device)

    if args.DDP:
        if not args.is_second_stage:
            appearance_encoder = DDP(appearance_encoder, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
            if not args.freeze_controlnet:
                controlnet = DDP(controlnet, device_ids=[args.local_rank], output_device=args.local_rank)
            if args.init_unet_lora:
                unet = DDP(unet, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=args.init_unet_lora)
        else:
            unet = DDP(unet, device_ids=[args.local_rank], output_device=args.local_rank)

    validation_pipeline = AnimationPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet.module if hasattr(unet, "module") else unet,
        controlnet=controlnet.module if hasattr(controlnet, "module") else controlnet,
        scheduler=noise_scheduler
    )
    validation_pipeline.to(device)
    
        
    if is_main_process:
        zero_rank_print(f"**** Training ****")
        zero_rank_print(f"  Trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")
        zero_rank_print(f"  Num training data : {len(train_dataset)}")
        zero_rank_print(f"  Epochs : {args.n_epochs}")
        zero_rank_print(f"  Batch size : {args.batch_size}")
        zero_rank_print(f"  Num processes : {args.num_processes}")
        zero_rank_print(f"  Total batch_size : {args.batch_size * args.num_processes}")

    scaler = torch.cuda.amp.GradScaler()
    global_step = 1
    loss_avgmeter = AverageMeter()
    save_args(args, args.args_save_path)
    save_config(config, args.config_save_path)

    for epoch in range(1, args.n_epochs+1):
        if global_step >= args.n_iters: 
            break
        if args.DDP:
            train_loader.sampler.set_epoch(epoch)

        with tqdm(train_loader, unit="iter", ncols=100) as tl:
            for batch_idx, batch in enumerate(tl):
                if global_step >= args.n_iters: 
                    break
                tl.set_description(f"Epoch {epoch}")
                img = batch['img'].to(device)  # [b f 3 512 512], -1~1
                ref_img = batch["ref_img"].to(device)  # [b 1 3 512 512], -1~1
                motion = batch["motion"].to(device)  # [b f 3 512 512], 0~1                
                video_length = img.shape[1]

                if global_step == 1 and not(args.is_second_stage):
                    b,f,c,h,w = img.shape
                    b = min(b, 4)

                    tmp_img = img[:b].detach().cpu().permute(2,1,0,3,4).reshape(c,f,b*h,w).unsqueeze(0)
                    tmp_ref_img = ref_img[:b].detach().cpu().repeat_interleave(f, dim=1).permute(2,1,0,3,4).reshape(c,f,b*h,w).unsqueeze(0)
                    tmp_motion = motion[:b].detach().cpu().permute(2,1,0,3,4).reshape(c,f,b*h,w).unsqueeze(0)

                    tmp_img = (tmp_img + 1) / 2
                    tmp_ref_img = (tmp_ref_img + 1) / 2
                    tmp_imgs = torch.cat([tmp_ref_img, tmp_img, tmp_motion], dim=-1).squeeze().permute(1,2,0)
                    save_path = opj(args.sample_save_dir, "training_sanity_check.png")
                    Image.fromarray((tmp_imgs.numpy()*255).astype(np.uint8)).save(save_path)
                
                with torch.no_grad():
                    text_inputs = tokenizer(
                        batch["text"],
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids.to(device)
                    text_embeddings = text_encoder(
                        text_inputs,
                        attention_mask=None
                    )[0]
                
                with torch.no_grad():
                    assert ref_img.shape[1] == 1
                    ref_img = rearrange(ref_img, "b f c h w -> (b f) c h w")
                    l_ref_img = vae.encode(ref_img).latent_dist.mean * 0.18215
                
                with torch.no_grad():
                    img = rearrange(img, "b f c h w -> (b f) c h w")
                    l_img = vae.encode(img).latent_dist
                    l_img = l_img.sample()
                    l_img = rearrange(l_img, "(b f) c h w -> b c f h w", f=video_length)
                    l_img = l_img * 0.18215
                
                bs = l_img.shape[0]

                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs, ), device=l_img.device).long()
                noise = torch.randn_like(l_img)
                noisy_l_img = noise_scheduler.add_noise(l_img, noise, timesteps)
                if args.use_noisy_ref_img:
                    ref_noise = torch.randn_like(l_ref_img)
                    l_ref_img = noise_scheduler.add_noise(l_ref_img, ref_noise, timesteps)
                target = noise
                
                #### input shape
                # noisy_l_img : [b c f h w]
                # l_ref_img : [b c h w]
                # motion : [(b f) 3 H W]
                # timesteps : [b]
                # text_embeddings : [b 77 768]
                with torch.autocast("cuda") if args.freeze_controlnet else nullcontext():
                    appearance_encoder(
                        l_ref_img,
                        timesteps,
                        encoder_hidden_states=text_embeddings,
                        return_dict=False
                    )
                    
                    down_block_res_samples, mid_block_res_sample = get_controlnet_output(controlnet, noisy_l_img, timesteps, text_embeddings, video_length, motion, args.freeze_controlnet, args.use_temporal_controlnet)

                    reference_control_reader.update(reference_control_writer)

                    model_pred = unet(
                        noisy_l_img,
                        timesteps,
                        encoder_hidden_states=text_embeddings,
                        down_block_additional_residuals=down_block_res_samples,   
                        mid_block_additional_residual= mid_block_res_sample if args.use_temporal_controlnet else rearrange(mid_block_res_sample, "(b f) c h w -> b c f h w", f=video_length),
                        return_dict=False,
                    )[0]

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                if is_main_process:
                    tb_writer.add_scalar("loss step", loss.item(), global_step)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if not args.is_second_stage:
                    torch.nn.utils.clip_grad_norm_(appearance_encoder.parameters(), config.max_grad_norm)
                    if not args.freeze_controlnet:
                        torch.nn.utils.clip_grad_norm_(controlnet.parameters(), config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()to_path

                reference_control_reader.clear()
                reference_control_writer.clear()
                
                loss_avgmeter.update(loss.item(), img.shape[0])
                
                if is_main_process and  (global_step % args.save_model_iter_freq == 0 or global_step in args.validation_step_lst):
                    if not args.is_second_stage:
                        to_path = opj(args.model_save_dir, f"[AppearanceEncoder]_[Epoch={epoch}]_[Iter={global_step}]_[loss={loss_avgmeter.avg:.4f}].ckpt")
                        torch.save(appearance_encoder.module.state_dict() if hasattr(appearance_encoder, "module") else appearance_encoder.state_dict(), to_path)
                        zero_rank_print(f"Save AppearanceNet to {to_path}")                    

                        if args.init_unet_lora:
                            to_path = opj(args.model_save_dir, f"[UNet]_[Epoch={epoch}]_[Iter={global_step}]_[loss={loss_avgmeter.avg:.4f}].ckpt")
                            torch.save(unet.module.state_dict() if hasattr(unet, "module") else unet.state_dict(), to_path)
                            zero_rank_print(f"Save UNet to {to_path}")
            
                    else:
                        to_path = opj(args.model_save_dir, f"[UNet]_[Epoch={epoch}]_[Iter={global_step}]_[loss={loss_avgmeter.avg:.4f}].ckpt")
                        torch.save(unet.module.state_dict() if hasattr(unet, "module") else unet.state_dict(), to_path)
                        zero_rank_print(f"Save UNet to {to_path}")

                        if args.use_temporal_controlnet:
                            to_path = opj(args.model_save_dir, f"[ControlNetT]_[Epoch={epoch}]_[Iter={global_step}]_[loss={loss_avgmeter.avg:.4f}].ckpt")
                            torch.save(controlnet.module.state_dict() if hasattr(controlnet, 'module') else controlnet.state_dict(), to_path)
                            zero_rank_print(f"Save ControlNet to {to_path}")
                    
                if global_step % args.validation_step == 0 or global_step in args.validation_step_lst:
                    image_transform = T.RandomResizedCrop(
                        size=(config.size, config.size),
                        scale=(1.0,1.0),
                        ratio=(1.0,1.0),
                        interpolation=T.InterpolationMode.BILINEAR,
                        antialias=True,
                    )
                    torch.cuda.empty_cache()
                    test_videos = config.video_path
                    source_images = config.source_image
                    
                    sizes = [config.size] * len(test_videos)
                    steps = [config.S] * len(test_videos)
                    num_actual_inference_steps = config.get("num_actual_inference_steps", config.steps)

                    if len(source_images) % args.n_gpus == 0:
                        n_infer = len(source_images) // args.n_gpus
                    else:
                        n_infer = len(source_images) // args.n_gpus + 1
                    start_idx = int(args.local_rank * n_infer)
                    end_idx = start_idx + n_infer
                    source_images = source_images[start_idx:end_idx]
                    
                    test_videos = test_videos[start_idx:end_idx]
                    sizes = sizes[start_idx:end_idx]
                    steps = steps[start_idx:end_idx]

                    if args.DDP: 
                        dist.barrier()

                    validation_pbar = tqdm(
                        enumerate(zip(source_images,                     
                        test_videos, sizes, steps)),
                        total=len(source_images),
                        desc=f"validation local rank{args.local_rank}",
                        ncols=75
                    )
                    save_dir = opj(args.sample_save_dir, f"Epoch{epoch}_Iter{global_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    
                    for idx, (source_image_p, test_video_p, size, step) in validation_pbar:
                        prompt = n_prompt = ""
                        if args.motion_type != "densepose_rgb":
                            test_video_p = test_video_p.replace("densepose_rgb", args.motion_type)

                        if test_video_p.endswith('.mp4'):
                            control = VideoReader(test_video_p).read()
                            if control[0].shape[0] != size:
                                control = [np.array(image_transform(Image.fromarray(c))) for c in control]
                            if config.max_length is not None:
                                control = control[config.offset: (config.offset+config.max_length)]
                            control = np.array(control)

                        if source_image_p.endswith(".mp4"):
                            source_image = np.array(image_transform(Image.fromarray(VideoReader(source_image_p).read()[0])))
                        else:
                            source_image = np.array(Image.open(source_image_p).resize((size, size)))
                        H, W, C = source_image.shape

                        init_latents = None
                        original_length = control.shape[0]
                        if control.shape[0] % config.L > 0:
                            control = np.pad(control, ((0, config.L-control.shape[0] % config.L), (0, 0), (0, 0), (0, 0)), mode='edge')

                        generator = torch.Generator(device=device)
                        generator.manual_seed(torch.initial_seed())

                        with torch.autocast("cuda"), torch.no_grad():
                            sample = validation_pipeline(
                                prompt,
                                negative_prompt         = n_prompt,
                                num_inference_steps     = config.steps,
                                guidance_scale          = config.guidance_scale,
                                width                   = W,
                                height                  = H,
                                video_length            = len(control),
                                controlnet_condition    = control,
                                context_frames          = config.L,
                                context_overlap         = config.L // 4,
                                init_latents            = init_latents,
                                generator               = generator,
                                num_actual_inference_steps = num_actual_inference_steps,
                                appearance_encoder       = appearance_encoder.module if hasattr(appearance_encoder, "module") else appearance_encoder, 
                                reference_control_writer = reference_control_writer,
                                reference_control_reader = reference_control_reader,
                                source_image             = source_image,
                                dist=True, rank=args.local_rank,
                                use_temporal_taumap      = True,                                
                                use_temporal_controlnet  = args.use_temporal_controlnet,
                                use_noisy_ref_img = args.use_noisy_ref_img,
                            ).videos

                        samples_per_video = []
                        save_source_images = np.array([source_image] * original_length)
                        save_source_images = rearrange(torch.from_numpy(save_source_images), "t h w c -> 1 c t h w") / 255.0
                        samples_per_video.append(save_source_images)
                        
                        control = control / 255.0
                        control = rearrange(control, "t h w c -> 1 c t h w")
                        control = torch.from_numpy(control)
                        samples_per_video.append(control[:, :, :original_length])
                        samples_per_video.append(sample[:, :, :original_length])
                        samples_per_video = torch.cat(samples_per_video)

                        source_dn = os.path.basename(os.path.dirname(source_image_p))
                        source_bn = os.path.splitext(os.path.basename(source_image_p))[0]
                        source_fn = source_bn if ("pose" in source_dn) or ("image" in source_dn) else source_dn

                        video_dn = os.path.basename(os.path.dirname(test_video_p))
                        video_bn = os.path.splitext(os.path.basename(test_video_p))[0]
                        video_fn = video_bn if ("pose" in video_dn) or ("pose" in video_dn) else video_dn

                        save_p = opj(save_dir, f"videos/{source_fn}__{video_fn}.mp4")
                        save_grid_p = opj(save_dir, f"videos/{source_fn}__{video_fn}_grid.mp4")
                        save_videos_grid(samples_per_video[-1:], save_p)
                        save_videos_grid(samples_per_video, save_grid_p)
                        print(f"save video to : {save_p}")
    
                    reference_control_writer = ReferenceAttentionControl(
                        appearance_encoder.module if hasattr(appearance_encoder, "module") else appearance_encoder, 
                        do_classifier_free_guidance=True, 
                        mode="write", 
                        fusion_blocks=config.fusion_blocks, 
                        batch_size=args.batch_size, 
                        is_train=True, 
                        is_second_stage=args.is_second_stage
                    )
                    reference_control_reader = ReferenceAttentionControl(
                        unet.module if hasattr(unet, "module") else unet,
                        do_classifier_free_guidance=True,
                        mode="read", 
                        fusion_blocks=config.fusion_blocks,
                        batch_size=args.batch_size, 
                        is_train=True, 
                        is_second_stage=args.is_second_stage
                    )
                    
                    if args.DDP:
                        dist.barrier()
                    if is_main_process:
                        torch.cuda.empty_cache()
                        os.system(f"./gen_eval_tiktok.sh {save_dir}")
                    if args.DDP:
                        dist.barrier()

                global_step += 1
                tl.set_postfix(loss=loss.item(), loss_avg=loss_avgmeter.avg)
        if is_main_process:
            tb_writer.add_scalar("loss epoch", loss_avgmeter.avg, epoch)
    if args.DDP:
        dist.destroy_process_group()  

if __name__=="__main__":
    args = parse_args()
    main(args)
