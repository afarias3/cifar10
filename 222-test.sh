#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --account=lonepeak-gpu
#SBATCH --partition=lonepeak-gpu
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --mem=16GB
#SBATCH -e 222test_slurm-%j.err-%N
#SBATCH -o 222test_slurm-%j.out-%N
source activate base
conda activate cifar10 
python -u $HOME/py_projects/cifar10/test.py --data_root './data_root' --sigma 0.3 --checkpoint './log_root/logs/12_27/00-44-07_data_root_data_root_logging_root_log_root_experiment_name__checkpoint_None_sigma_0.3_lr_0.001_reg_weight_0.0_/model-epoch_0_iter_12500.pth' > 222test-sigma-0.3.log
