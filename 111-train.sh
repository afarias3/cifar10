#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --account=lonepeak-gpu
#SBATCH --partition=lonepeak-gpu
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --mem=16GB
#SBATCH -e 111train_slurm-%j.err-%N
#SBATCH -o 111train_slurm-%j.out-%N

source activate base
conda activate cifar10 
python  $HOME/py_projects/cifar10/train.py --data_root $HOME/py_projects/cifar10/data_root --logging_root $HOME/py_projects/cifar10/log_root --train_test train --sigma 0.3 2>&1 >> $HOME/py_projects/cifar10/111train-sigma-0.3.log
