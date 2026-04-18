#!/bin/bash
#SBATCH --job-name=geopixel_finetune
#SBATCH --account=nlp
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb
#SBATCH --mem=50gb
#SBATCH --time=216:00:00
#SBATCH --output=/logs/geopixel_%j.out
#SBATCH --error=/logs/geopixel_%j.err

module load python/3.11

source geopixel_venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=0

bash finetune_hpc.sh
