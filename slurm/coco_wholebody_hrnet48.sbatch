#!/usr/bin/bash
#SBATCH --job-name=coco_wholebody_hrnet48
#SBATCH --output=coco_wholebody_hrnet48.out
#SBATCH --error=coco_wholebody_hrnet48.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --time=40:00:00

conda activate opensim_mm
ml system
ml ffmpeg
python /scratch/users/yonigoz/opencap-core/benchmark.py \
    --model_config_pose configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py \
    --model_ckpt_pose  /scratch/users/yonigoz/mmpose_data/ckpts/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
    --dataName "coco_wholebody_hrnet48" \
    --marker_set "Coco" \
    --batch_size_det 2 \
    --batch_size_pose 4

