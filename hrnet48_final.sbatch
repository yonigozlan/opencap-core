#!/usr/bin/bash
#SBATCH --job-name=hrnet48_final
#SBATCH --output=hrnet48_final.out
#SBATCH --error=hrnet48_final.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --time=40:00:00

conda activate opensim_mm
ml system
ml ffmpeg
python /scratch/users/yonigoz/opencap-core/benchmark.py \
    --model_config_pose configs/body_2d_keypoint/topdown_heatmap/infinity/hrnet48/td-hm_hrnet-w48_dark-8xb32-210e_merge_bedlam_infinity_coco_3DPW_eval_rich-384x288_pretrained.py \
    --model_ckpt_pose  /scratch/users/yonigoz/mmpose_data/work_dirs/merge_bedlam_infinity_coco_3DPW_eval_bedlam_final/HRNet/w48_dark_pretrained/best_infinity_AP_epoch_21.pth \
    --dataName "hrnet48_final" \
    --batch_size_det 2 \
    --batch_size_pose 4

