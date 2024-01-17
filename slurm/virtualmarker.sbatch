#!/usr/bin/bash
#SBATCH --job-name=virtualmarker
#SBATCH --output=virtualmarker.out
#SBATCH --error=virtualmarker.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --time=40:00:00

conda activate opensim_mm
ml system
ml ffmpeg
python /scratch/users/yonigoz/opencap-core/benchmark.py \
    --dataName "virtualmarker" \
    --marker_set "Anatomical" \
    --batch_size_det 2 \
    --batch_size_pose 1000 \
    --alt_model "VirtualMarker"

