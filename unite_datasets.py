import datetime
import os
import sys
import numpy as np
import traceback
import shutil
import argparse
import pickle

import torch
import torch.nn as nn

from transformers import BartModel, BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, TrainingArguments, default_data_collator, Trainer
# from transformers.modeling_bart import shift_tokens_right
from main import BreakDataset, shift_tokens_right

# user_name = 'meitars'
user_name = 'omriefroni'
cache_dir = '/home/joberant/nlp_fall_2021/' + user_name + '/.cache'
os.environ["TRANSFORMERS_CACHE"] = cache_dir
output_path = '/home/joberant/nlp_fall_2021/omriefroni/qdmr_project/unite_datasets_omri/'

def create_unite_dataset(trained_on_first_half_ckpt, trained_on_second_half_ckpt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if 'joberant' in os.path.abspath('./'):            
        model_d2q_first_half = BartForConditionalGeneration.from_pretrained(trained_on_first_half_ckpt, cache_dir=cache_dir)
        model_d2q_first_half = model_d2q_first_half.to(device)
        tokenizer_d2q_first_half = BartTokenizer.from_pretrained(trained_on_first_half_ckpt, cache_dir=cache_dir)

        model_d2q_second_half = BartForConditionalGeneration.from_pretrained(trained_on_second_half_ckpt, cache_dir=cache_dir)
        model_d2q_second_half = model_d2q_second_half.to(device)
        tokenizer_d2q_second_half = BartTokenizer.from_pretrained(trained_on_second_half_ckpt, cache_dir=cache_dir)
    else:
        model_d2q_first_half = BartForConditionalGeneration.from_pretrained(trained_on_first_half_ckpt)
        tokenizer_d2q_first_half = BartTokenizer.from_pretrained(trained_on_first_half_ckpt)

        model_d2q_second_half = BartForConditionalGeneration.from_pretrained(trained_on_second_half_ckpt)
        tokenizer_d2q_second_half = BartTokenizer.from_pretrained(trained_on_second_half_ckpt)

    train_dataset_first_half = BreakDataset('train', which_half='first')
    train_dataset_second_half = BreakDataset('train', which_half='second')

    def change_q_to_predict_first_half(examples):
        targets = examples['decomposition']

        model_inputs = tokenizer_d2q_second_half(targets, return_tensors="pt")
        model_inputs = model_inputs.to(device)
        greedy_output = model_d2q_second_half.generate(**model_inputs, max_length=256)
        question_text = tokenizer_d2q_second_half.decode(greedy_output[0], skip_special_tokens=True)

        new_pair = {}
        new_pair["question_text"] = question_text
        new_pair["decomposition"] = examples['decomposition']

        return new_pair

    def change_q_to_predict_second_half(examples):
        targets = examples['decomposition']

        model_inputs = tokenizer_d2q_first_half(targets, return_tensors="pt")
        model_inputs = model_inputs.to(device)
        greedy_output = model_d2q_first_half.generate(**model_inputs, max_length=256)
        question_text = tokenizer_d2q_first_half.decode(greedy_output[0], skip_special_tokens=True)

        new_pair = {}
        new_pair["question_text"] = question_text
        new_pair["decomposition"] = examples['decomposition']

        return new_pair


    train_dataset_first_half.map(change_q_to_predict_first_half)
    train_dataset_second_half.map(change_q_to_predict_second_half)
    # train_dataset = BreakDataset('train')

    output_path_1 = os.path.join(output_path + 'first_half_data')
    output_path_2 = os.path.join(output_path + 'second_half_data')

    train_dataset_first_half.data.save_to_disk(output_path_1)
    train_dataset_second_half.data.save_to_disk(output_path_2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='dynamic cage deformation')
    parser.add_argument("--name", help="experiment name")
    parser.add_argument("--first_half_ckpt", type=str, help="path to the checkpoint of the model that was trained on the first half of the dataset")
    parser.add_argument("--second_half_ckpt", type=str, help="path to the checkpoint of the model that was trained on the second half of the dataset")
    opt = parser.parse_args()

    trained_on_first_half_ckpt = opt.first_half_ckpt
    trained_on_second_half_ckpt = opt.second_half_ckpt
    #
    # trained_on_first_half_ckpt = '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/log/main-12-04-2021__21:32:48-D2Q_trained_on_first_half/checkpoint-44000'
    # trained_on_second_half_ckpt = '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/log/main-12-04-2021__10:35:55-D2Q_trained_on_second_half/checkpoint-44000'

    create_unite_dataset(trained_on_first_half_ckpt, trained_on_second_half_ckpt)
