#!/usr/bin/bash
#SBATCH --job-name=coco_wholebody_rtmw-x
#SBATCH --output=coco_wholebody_rtmw-x.out
#SBATCH --error=coco_wholebody_rtmw-x.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --time=40:00:00

conda activate opensim_mm
ml system
ml ffmpeg
python /scratch/users/yonigoz/opencap-core/benchmark.py \
    --model_config_pose configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py \
    --model_ckpt_pose  /scratch/users/yonigoz/mmpose_data/ckpts/rtmwpose/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth \
    --dataName "coco_wholebody_rtmw-x" \
    --marker_set "Coco" \
    --batch_size_det 2 \
    --batch_size_pose 4

