#!/usr/bin/bash
#SBATCH --job-name=hrnet48_no_pretrained
#SBATCH --output=hrnet48_no_pretrained.out
#SBATCH --error=hrnet48_no_pretrained.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --time=40:00:00

conda activate opensim_mm
ml system
ml ffmpeg
python /scratch/users/yonigoz/opencap-core/benchmark.py \
    --model_config_pose configs/body_2d_keypoint/topdown_heatmap/infinity/hrnet48/td-hm_hrnet-w48_dark-8xb32-210e_no_pretrained.py \
    --model_ckpt_pose /scratch/users/yonigoz/mmpose_data/work_dirs/no_pretrained/HRNet/w48_dark_pretrained/epoch_30.pth \
    --dataName "hrnet48_no_pretrained" \
    --batch_size_det 2 \
    --batch_size_pose 4

