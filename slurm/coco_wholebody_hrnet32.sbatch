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
    --model_config_pose configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w32_8xb64-210e_coco-wholebody-256x192.py \
    --model_ckpt_pose  /scratch/users/yonigoz/mmpose_data/ckpts/hrnet/hrnet_w32_coco_wholebody_256x192_dark-469327ef_20200922.pth \
    --dataName "coco_wholebody_hrnet32" \
    --marker_set "Coco" \
    --batch_size_det 2 \
    --batch_size_pose 4

