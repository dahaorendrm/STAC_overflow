#!/bin/bash -l

#BATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=32

#SBATCH --mem=40G
# SBATCH --mem-per-cpu=10G

#SBATCH --job-name=STAC_train
#SBATCH --partition=gpu-t4
#SBATCH --gpus=1

#SBATCH --time=0-20:00:00
# SBATCH --output=ArraySCI%A-%a.out
#SBATCH --mail-user='xmdrm@udel.edu'
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90

# export OMP_NUM_THREADS=4
vpkg_require xm_pytorch/20210902-STAC
python3 -u stac_train.py
python3 -u submit.py
cd submit-pytorch 
zip -r submission_[$SLURM_JOB_ID].zip assets *.py
cd ..
