#!/bin/bash
#
#SBATCH --job-name=fol_cmp # Job name
#SBATCH --time=180:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20G

module load Miniconda/3;           # Load module dependencies
module load ompi-cpu;

source activate tuna-project
python inference/inference_WISE_dataset.py \
--network_loc "/slimdata/tunadata/SAGE/saved_networks/network_states_saved_pretrain_syntho_run_WISE_800_train/00000-gpus1-batch4-offsetsFalse"  \
--cond_loc "/slimdata/tunadata/SAGE/test_dataset/rtms/" \
--back_loc "/slimdata/tunadata/SAGE/test_dataset/backs/" \
--gt_loc "/slimdata/tunadata/SAGE/test_dataset/gts/" \
--use_offsets \
--out_chan 1 \
--num_skip 1 \
--cond_norm 400000  \
--gt_norm 4.5 \
--outdir "WISE_sampling_800" 