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
from data_util import BreakDataset, shift_tokens_right
import datasets
from global_params import *

# user_name = 'meitars'
user_name = 'omriefroni'
cache_dir = '/home/joberant/nlp_fall_2021/' + user_name + '/.cache'
os.environ["TRANSFORMERS_CACHE"] = cache_dir
output_path = '/home/joberant/nlp_fall_2021/omriefroni/qdmr_project/unite_datasets_omri/'

def create_unite_dataset_inverse(q2d_ckpt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    if 'joberant' in os.path.abspath('./'):

        # q2d_ckpt = '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/log/main-08-04-2021__19:52:35/checkpoint-119500'
        model_q2d = BartForConditionalGeneration.from_pretrained(q2d_ckpt, cache_dir=cache_dir)
        model_q2d = model_q2d.to(device)
        tokenizer_q2d = BartTokenizer.from_pretrained(q2d_ckpt, cache_dir=cache_dir)
    # else:
    #     model_d2q_first_half = BartForConditionalGeneration.from_pretrained(trained_on_first_half_ckpt)
    #     tokenizer_d2q_first_half = BartTokenizer.from_pretrained(trained_on_first_half_ckpt)
    #
    #     model_d2q_second_half = BartForConditionalGeneration.from_pretrained(trained_on_second_half_ckpt)
    #     tokenizer_d2q_second_half = BartTokenizer.from_pretrained(trained_on_second_half_ckpt)


    train_dataset_q_augmented = BreakDataset('train', augmentation_path=output_path, without_regular=True)

    def change_d_to_predict(examples):
        inputs = examples['question_text']
        targets = examples['decomposition']


        model_inputs = tokenizer_q2d(inputs, return_tensors="pt")
        model_inputs = model_inputs.to(device)
        greedy_output = model_q2d.generate(**model_inputs, max_length=256)
        decomp_new = tokenizer_q2d.decode(greedy_output[-1], skip_special_tokens=True)
        # print('_____________________________________________________________________-')
        # print('_____________________________________________________________________-')
        # print('ORIGINAL : %s ' % inputs)
        # for i in range(3):
        #     print('###################%d###################' %i)
        #     print(tokenizer_d2q_second_half.decode(greedy_output[i], skip_special_tokens=True))

        new_pair = {}
        new_pair["question_text"] = examples['question_text']
        new_pair["decomposition"] = decomp_new

        return new_pair

    train_dataset_q_augmented.map(change_d_to_predict)

    output_path_1 = os.path.join(output_path + 'augmented_decomp_based_on_main-08-04-2021__19:52:35_checkpoint-119500')

    train_dataset_q_augmented.data.save_to_disk(output_path_1)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='dynamic cage deformation')
    parser.add_argument("--name", help="experiment name")
    parser.add_argument("--first_half_ckpt", type=str, help="path to the checkpoint of the model that was trained on the first half of the dataset")
    parser.add_argument("--second_half_ckpt", type=str, help="path to the checkpoint of the model that was trained on the second half of the dataset")
    opt = parser.parse_args()


    q2d_ckpt = '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/log/main-08-04-2021__19:52:35/checkpoint-119500'
    create_unite_dataset_inverse(q2d_ckpt)
