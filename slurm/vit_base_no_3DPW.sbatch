#!/usr/bin/bash
#SBATCH --job-name=vitbase_no_3DPW
#SBATCH --output=vitbase_no_3DPW.out
#SBATCH --error=vitbase_no_3DPW.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --time=40:00:00

conda activate opensim_mm
ml system
ml ffmpeg
python /scratch/users/yonigoz/opencap-core/benchmark.py \
    --model_config_pose configs/body_2d_keypoint/topdown_heatmap/infinity/vitbase/td-hm_ViTPose-base_8xb64-210e_no_3DPW.py \
    --model_ckpt_pose  /scratch/users/yonigoz/mmpose_data/work_dirs/no_3DPW/ViT/base_pretrained/epoch_30.pth \
    --dataName "vitbase_no_3DPW" \
    --batch_size_det 2 \
    --batch_size_pose 4

