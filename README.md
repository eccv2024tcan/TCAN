# [ECCV2024] TCAN: Animating Human Images with Temporally Consistent Pose Guidance using Diffusion Models
This repository is the official implementation of [TCAN](https://arxiv.org/abs/2407.09012)

> **TCAN: Animating Human Images with Temporally Consistent Pose Guidance using Diffusion Models**<br>
> [Jeongho Kim*](https://scholar.google.co.kr/citations?user=4SCCBFwAAAAJ&hl=ko/), [Min-Jung Kim*](https://emjay73.github.io/), [Junsoo Lee](https://ssuhan.github.io/), [Jaegul Choo](https://sites.google.com/site/jaegulchoo/) 
(*: equal contribution)

[[arXiv Paper](https://arxiv.org/abs/2407.09012)]&nbsp;
[[Project Page](https://eccv2024tcan.github.io/)]&nbsp;


## TODO List
- [ ] Inference code
- [ ] Release model weights
- [x] Training code

## Dataset
Preprocessed TikTok: [Download](https://huggingface.co/datasets/rlawjdghek/TikTok/tree/main)

Unzip the donwnloaded dataset and set the path to the dataset as follows
```bash
cd TCAN
mkdir DATA
cd DATA
ln -s [data_path] TikTok 
```
```bash
TCAN/DATA/TikTok
L train
L valid_video
```

## Environment
```bash
conda create -n tcan python=3.10
conda activate tcan
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install diffusers==0.25.0
pip install xformers==0.0.22
pip install accelerate==0.22.0
pip install transformers==4.32.0
pip install omegaconf
pip install einops
pip install clean-fid
pip install tensorboard
pip install imageio==2.9.0
pip install opencv-python
pip install av==11.0.0
pip install matplotlib
pip install peft==0.9.0
pip install imageio_ffmpeg
pip install ffmpeg
pip install scikit-image==0.20.0
pip install lpips
pip install onnxruntime
pip install numpy==1.26.4
```

#### From the Links
Download pretrained motion module weights provided by [AnimateDiff](https://github.com/guoyww/animatediff/)
and RealisticVision UNet weights from the link.
Place the downloaded weights into the 'checkpoints' directory.

Download Link: 
[mm_sd_v15.ckpt](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15.ckpt), 
[mm_sd_v15_v2.ckpt](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt).
[realisticVision](https://huggingface.co/spaces/TianxingWu/FreeInit/resolve/09c34cd1aae3a2362d116970e60a9d4f0c562738/models/DreamBooth_LoRA/realisticVisionV51_v20Novae.safetensors?download=true)



## Train
🔥We trained our model using two A100🔥

### First Stage
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 3874 train.py \
 --config "./configs/train/first_stage.yaml" \
 --batch_size 2\
 --motion_type dwpose \
 --pretrained_unet_path "./checkpoints/realisticVisionV51_v20Novae.safetensors" \
 --pretrained_appearance_encoder_path "./checkpoints/realisticVisionV51_v20Novae.safetensors" \
 --pretrained_controlnet_path "./checkpoints/control_v11p_sd15_openpose_RenamedForMA.pth" \
 --freeze_controlnet \
 --init_unet_lora \
 --save_name First_Unetlora
```

### Second Stage
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 6836 train.py \
 --config ./configs/train/second_stage.yaml \
 --num_workers 2 \
 --batch_size 1 \
 --is_second_stage \
 --motion_type dwpose \
 --pretrained_unet_path "./logs/20240419_First_Unetlora/models/[UNet]_[Epoch=1]_[Iter=100]_[loss=0.1025].ckpt" \
 --pretrained_appearance_encoder_path "./logs/20240419_First_Unetlora/models/[AppearanceEncoder]_[Epoch=1]_[Iter=100]_[loss=0.1025].ckpt" \
 --pretrained_controlnet_path "./checkpoints/control_v11p_sd15_openpose_RenamedForMA.pth" \
 --init_unet_lora \
 --load_unet_lora_weight \
 --use_temporal_controlnet \
 --save_name SecondUnetloraTctrl
```
