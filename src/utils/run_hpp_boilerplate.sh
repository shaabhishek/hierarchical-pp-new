#!/bin/bash
#SBATCH --job-name=hpp-lvm
#SBATCH --partition m40-short
#SBATCH --mem=4096
#SBATCH --gres gpu:1
#SBATCH --output=script-out-hpp.log

. /home/abhishekshar/anaconda3/etc/profile.d/conda.sh
conda activate hpp

cd /home/abhishekshar/hierarchichal_point_process/src

