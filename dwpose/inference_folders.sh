#### 126서버에서 돌렸었음.
CUDA_VISIBLE_DEVICES=0 python inference_folders.py --from_root_dir "/home/nas4_dataset/vision/ted_dataset/TED384-v2/train" --p_idx 0 --n_proc 3 &> ./exp_nohup/tedtalk_0.txt &
CUDA_VISIBLE_DEVICES=1 python inference_folders.py --from_root_dir "/home/nas4_dataset/vision/ted_dataset/TED384-v2/train" --p_idx 1 --n_proc 3 &> ./exp_nohup/tedtalk_1.txt &
CUDA_VISIBLE_DEVICES=2 python inference_folders.py --from_root_dir "/home/nas4_dataset/vision/ted_dataset/TED384-v2/train" --p_idx 2 --n_proc 3 &> ./exp_nohup/tedtalk_2.txt &
CUDA_VISIBLE_DEVICES=3 python inference_folders.py --from_root_dir "/home/nas4_dataset/vision/ted_dataset/TED384-v2/valid" --p_idx 0 --n_proc 1 &> ./exp_nohup/tedtalk_3.txt &


# CUDA_VISIBLE_DEVICES=0 python inference_folders.py --from_root_dir "/home/nas4_dataset/vision/TikTok/TikTok_dataset/train" --p_idx 0 --n_proc 3 &> ./exp_nohup/dummy_0.txt &
# CUDA_VISIBLE_DEVICES=1 python inference_folders.py --from_root_dir "/home/nas4_dataset/vision/TikTok/TikTok_dataset/train" --p_idx 1 --n_proc 3 &> ./exp_nohup/dummy_1.txt &
# CUDA_VISIBLE_DEVICES=2 python inference_folders.py --from_root_dir "/home/nas4_dataset/vision/TikTok/TikTok_dataset/train" --p_idx 2 --n_proc 3 &> ./exp_nohup/dummy_2.txt &
# CUDA_VISIBLE_DEVICES=3 python inference_folders.py --from_root_dir "/home/nas4_dataset/vision/TikTok/TikTok_dataset/valid" --p_idx 2 --n_proc 3 &> ./exp_nohup/dummy_3.txt &