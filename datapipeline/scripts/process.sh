#!/usr/bin/bash

# run full pre-processing

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <seq_name> [CUDA_VISIBLE_DEVICES]"
    exit 1
fi

seq_name=$1
export CUDA_VISIBLE_DEVICES=${2:-0} # Default to 0 if not provided
HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEQ_P="$HOME/data/$seq_name"
HMR="$HOME/submodules/camera_motion"

echo " [DATA] HOME: $HOME"
echo "SEQUENCE DIR: $SEQ_P"

if [ ! -d "$SEQ_P" ]; then
    echo "Error: '$SEQ_P' does not exist"
    exit 1
fi

# environments
echo "Warning: Personalized CONDA ENV Paths!"
# pyhmr="/home/michel/.miniconda3/envs/preprocessing/bin/python"
pyhmr="/home/michel/.miniconda3/envs/handobjectreconstruction/bin/python"
pycolmap="/home/michel/.miniconda3/envs/hloc/bin/python"

# set up folder structure
SCRATCH_P="$SEQ_P/scratch"
MFV_P="$SCRATCH_P/mfv"
SAM_P="$SCRATCH_P/sam2"
COLMAP_P="$SCRATCH_P/colmap"
ALIGNMENT_P="$SCRATCH_P/alignment"

mkdir -p $MFV_P $SAM_P $COLMAP_P $ALIGNMENT_P "$HOME/.gradio_cache"
export GRADIO_TEMP_DIR="$HOME/.gradio_cache"

echo "--- Step 1: HMR - SMPLX Poses"
cd $HMR
$pyhmr apis/demo.py --afv_upload "$SEQ_P/video.mp4" \
    --output_folder $MFV_P --verbose --static_cam 1 --avatarTrackingLimit 1 --cfg_file cfg_files/mfv2-1_img_align.yaml

cd $HOME
echo "--- Step 2.1: Gradio - SAM2 PROMPTS"
$pyhmr src/create_mask_prompts.py --video_p "$SEQ_P/video.mp4" --out_p "$SEQ_P/scratch/sam2" # --online

echo "--- Step 2.2: SAM2 - Segmentations"
$pyhmr src/create_masks.py --seq_name $seq_name --use_cache

# TODO - add object foundation model!
# echo "--- Step X.Y: TRELLIS - 3D Mesh Reconstruction"
# pytrellis="/home/michel/.miniconda3/envs/trellis/bin/python"
# $pytrellis src/estimate_mesh.py --seq_name $seq_name

# echo "--- Step 2.3: DSINE - Normals"
# $pyhmr src/estimate_surface_normals.py ./configs/dsine.txt --seq_name $seq_name

echo "--- Step 3: COLMAP - Object Poses & SfM Model"
$pycolmap src/colmap_estimation.py --seq_name $seq_name --num_pairs 80 # --skip_sfm

echo "--- Step 4: Human-Object Alignment"
$pyhmr src/align_ho.py --seq_name $seq_name --side_view

echo "--- Step 5: CLEAN UP"
$pyhmr src/merge_videos.py --seq_name $seq_name
