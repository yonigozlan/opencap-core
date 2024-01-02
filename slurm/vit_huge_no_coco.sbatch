#!/usr/bin/bash
#SBATCH --job-name=vithuge_no_coco
#SBATCH --output=vithuge_no_coco.out
#SBATCH --error=vithuge_no_coco.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --time=40:00:00

conda activate opensim_mm
ml system
ml ffmpeg
python /scratch/users/yonigoz/opencap-core/benchmark.py \
    --model_config_pose configs/body_2d_keypoint/topdown_heatmap/infinity/vithuge/td-hm_ViTPose-huge_8xb64-210e_no_coco.py \
    --model_ckpt_pose  /scratch/users/yonigoz/mmpose_data/work_dirs/no_coco/ViT/huge_pretrained/epoch_30.pth \
    --dataName "vithuge_no_coco" \
    --batch_size_det 2 \
    --batch_size_pose 4

