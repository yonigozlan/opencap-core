#!/usr/bin/bash
#SBATCH --job-name=hrnet32_final
#SBATCH --output=hrnet32_final.out
#SBATCH --error=hrnet32_final.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --time=40:00:00

conda activate opensim_mm
ml system
ml ffmpeg
python /scratch/users/yonigoz/opencap-core/benchmark.py \
    --model_config_pose configs/body_2d_keypoint/topdown_heatmap/infinity/hrnet32/td-hm_hrnet-w32_dark-8xb64-210e_merge_bedlam_infinity_coco_3DPW_eval_rich-384x288_pretrained.py \
    --model_ckpt_pose /scratch/users/yonigoz/mmpose_data/work_dirs/merge_bedlam_infinity_coco_3DPW_eval_bedlam_final/HRNet/w32_dark_bedlam/epoch_30.pth \
    --dataName "hrnet32_final" \
    --batch_size_det 2 \
    --batch_size_pose 8

