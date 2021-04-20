#! /bin/sh
#SBATCH --job-name=QDMR_evaluate
#SBATCH --output=QDMR_evaluate.out
#SBATCH --error=QDMR_evaluate.err
#SBATCH --partition=studentbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
python evaluate.py --val_dataset_path '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/val_dataset.csv' --ckpt '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/log/main-19-04-2021__05:53:08-Q2D_on_halves/checkpoint-100000' --save_dir '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/evaluations/extra_data_2halves_total_of_2_epochs/'
# python evaluate.py --val_dataset_path '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/val_dataset.csv' --ckpt '/home/joberant/nlp_fall_2021/omriefroni/qdmr_project/log/main-16-04-2021__01:11:56/checkpoint-110000' --save_dir '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/evaluations/extra_data_2halves_total_of_2_epochs/'
# python evaluate.py --val_dataset_path '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/val_dataset.csv' --ckpt '/home/joberant/nlp_fall_2021/omriefroni/qdmr_project/log/main-16-04-2021__01:11:56/checkpoint-452000' --save_dir '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/evaluations/extra_data_2halves_2_epochs_later/'
# python evaluate.py --val_dataset_path '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/val_dataset.csv' --ckpt '/home/joberant/nlp_fall_2021/omriefroni/qdmr_project/log/main-16-04-2021__01:11:56/checkpoint-374000' --save_dir '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/evaluations/extra_data_2halves/'
# python evaluate.py --val_dataset_path '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/val_dataset.csv' --ckpt '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/log/main-08-04-2021__19:52:35/checkpoint-119500' --save_dir '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/evaluations/model_baseline/'
