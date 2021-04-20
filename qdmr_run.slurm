#! /bin/sh
#SBATCH --job-name=QDMR_Q2D_on_halves
#SBATCH --output=QDMR_Q2D_on_halves.out
#SBATCH --error=QDMR_Q2D_on_halves.err
#SBATCH --partition=studentbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
python main.py --name Q2D_on_halves --Q2D --augmentation_path '/home/joberant/nlp_fall_2021/omriefroni/qdmr_project/unite_datasets_omri/'
