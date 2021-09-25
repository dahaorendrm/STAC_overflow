#!/bin/bash -l

#BATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

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
cd submit-pytorch
cp -f *.py codeexecution
cp -f -r assets codeexecution/assets
#cd codeexecution
python3 -u codeexecution/main.py
cd ../
