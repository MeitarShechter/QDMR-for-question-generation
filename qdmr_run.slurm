#! /bin/sh
#SBATCH --job-name=test1
#SBATCH --output=test1.out
#SBATCH --error=test1.err
#SBATCH --partition=studentbatch
#SBATCH --job-name testing
#SBATCH --time=100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
python main.py
