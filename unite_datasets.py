import datetime
import os
import sys
import numpy as np
import traceback
import shutil
import argparse

import torch
import torch.nn as nn

from transformers import BartModel, BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, TrainingArguments, default_data_collator, Trainer
# from transformers.modeling_bart import shift_tokens_right
from main import BreakDataset, shift_tokens_right




def create_unite_dataset(trained_on_first_half_ckpt, trained_on_second_half_ckpt):

    if 'joberant' in os.path.abspath('./'):            
        cache_dir = '/home/joberant/nlp_fall_2021/omriefroni/.cache'

        model_d2q_first_half = BartForConditionalGeneration.from_pretrained(trained_on_first_half_ckpt, cache_dir=cache_dir)            
        tokenizer_d2q_first_half = BartTokenizer.from_pretrained(trained_on_first_half_ckpt, cache_dir=cache_dir)

        model_d2q_second_half = BartForConditionalGeneration.from_pretrained(trained_on_second_half_ckpt, cache_dir=cache_dir)
        tokenizer_d2q_second_half = BartTokenizer.from_pretrained(trained_on_second_half_ckpt, cache_dir=cache_dir)
    else:
        model_d2q_first_half = BartForConditionalGeneration.from_pretrained(trained_on_first_half_ckpt)
        tokenizer_d2q_first_half = BartTokenizer.from_pretrained(trained_on_first_half_ckpt)

        model_d2q_second_half = BartForConditionalGeneration.from_pretrained(trained_on_second_half_ckpt)
        tokenizer_d2q_second_half = BartTokenizer.from_pretrained(trained_on_second_half_ckpt)

    train_dataset_first_half = BreakDataset('train', which_half='first')
    train_dataset_second_half = BreakDataset('train', which_half='second')

    def change_q_to_predict_first_half(examples):
        inputs = examples['question_text']
        targets = examples['decomposition']

        model_inputs = tokenizer_d2q_second_half(targets, return_tensors="pt")
        greedy_output = model_d2q_second_half.generate(**model_inputs, max_length=256)
        question_text = tokenizer_d2q_second_half.decode(greedy_output[0], skip_special_tokens=True)

        new_pair = {}
        new_pair["question_text"] = question_text
        new_pair["decomposition"] = examples['decomposition']

        return new_pair

    def change_q_to_predict_second_half(examples):
        inputs = examples['question_text']
        targets = examples['decomposition']

        model_inputs = tokenizer_d2q_first_half(targets, return_tensors="pt")
        greedy_output = model_d2q_first_half.generate(**model_inputs, max_length=256)
        question_text = tokenizer_d2q_first_half.decode(greedy_output[0], skip_special_tokens=True)

        new_pair = {}
        new_pair["question_text"] = question_text
        new_pair["decomposition"] = examples['decomposition']

        return new_pair


    train_dataset_first_half.map(change_q_to_predict_first_half)
    train_dataset_second_half.map(change_q_to_predict_second_half)
    train_dataset = BreakDataset('train')
    train_dataset_unite = torch.utils.data.ConcatDataset([train_dataset, train_dataset_first_half, train_dataset_second_half])
    # TODO: save the preprocessed datasets as this processing takes a lot of time!!!

    return train_dataset_unite


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='dynamic cage deformation')
    parser.add_argument("--name", help="experiment name")
    parser.add_argument("--fist_half_ckpt", type=str, help="path to the checkpoint of the model that was trained on the first half of the dataset")
    parser.add_argument("--second_half_ckpt", type=str, help="path to the checkpoint of the model that was trained on the second half of the dataset")
    opt = parser.parse_args()

    trained_on_first_half_ckpt = opt.fist_half_ckpt
    trained_on_second_half_ckpt = opt.second_half_ckpt

    create_unite_dataset(trained_on_first_half_ckpt, trained_on_second_half_ckpt)
