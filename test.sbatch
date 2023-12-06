#!/usr/bin/bash
#SBATCH --job-name=opencap_bench_test
#SBATCH --output=opencap_bench_test.out
#SBATCH --error=opencap_bench_test.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1

conda activate opensim_mm
ml system
ml ffmpeg
python /scratch/users/yonigoz/opencap-core/benchmark \
    --model_config_pose configs/body_2d_keypoint/topdown_heatmap/infinity/hrnet32/td-hm_hrnet-w32_dark-8xb64-210e_merge_bedlam_infinity_coco_3DPW_eval_rich-384x288_pretrained.py \
    --model_ckpt_pose /scratch/users/yonigoz/mmpose_data/work_dirs/merge_bedlam_infinity_coco_3DPW_eval_bedlam_final/HRNet/w32_dark_bedlam/best_infinity_AP_epoch_21.pth \
    --dataName "Test_argparse"

