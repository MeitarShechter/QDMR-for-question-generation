#! /bin/sh
# conda activate /home/joberant/nlp_fall_2021/meitars/anaconda3/envs/QDMR
#SBATCH --job-name=QDMR_Q_to_D
#SBATCH --output=QDMR_Q_to_D.out
#SBATCH --error=QDMR_Q_to_D.err
#SBATCH --partition=studentbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
python main.py --Q2D
