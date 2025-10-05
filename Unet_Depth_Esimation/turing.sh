#!/bin/bash

#SBATCH --mail-user=jjandus@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J window_seg
#SBATCH --output=/home/jjandus/logs/window_seg%j.out
#SBATCH --error=/home/jjandus/logs/window_seg%j.err

#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C H100|A100|V100|A30
#SBATCH -p academic
#SBATCH -t 24:00:00

#SBATCH -t 24:00:00
module load cuda
source /home/jjandus/.venv/bin/activate
python3 /home/jjandus/RBE474X/part2/train.py
