#! /bin/sh
#SBATCH --job-name=QDMR_unite_data
#SBATCH --output=QDMR_unite_data.out
#SBATCH --error=QDMR_unite_data.err
#SBATCH --partition=studentbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
python unite_datasets.py --name unite_datasets --first_half_ckpt '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/log/main-12-04-2021__21:32:48-D2Q_trained_on_first_half/checkpoint-44000' --second_half_ckpt '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/log/main-12-04-2021__10:35:55-D2Q_trained_on_second_half/checkpoint-44000'
