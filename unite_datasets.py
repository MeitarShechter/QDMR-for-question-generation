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



def create_unite_dataset(trained_on_first_half_ckpt, trained_on_second_half_ckpt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    if 'joberant' in os.path.abspath('./'):            
        model_d2q_first_half = BartForConditionalGeneration.from_pretrained(trained_on_first_half_ckpt, cache_dir=cache_dir)
        model_d2q_first_half = model_d2q_first_half.to(device)
        tokenizer_d2q_first_half = BartTokenizer.from_pretrained(trained_on_first_half_ckpt, cache_dir=cache_dir)

        # model_d2q_second_half = BartForConditionalGeneration.from_pretrained(trained_on_second_half_ckpt, cache_dir=cache_dir)
        # model_d2q_second_half = model_d2q_second_half.to(device)
        # tokenizer_d2q_second_half = BartTokenizer.from_pretrained(trained_on_second_half_ckpt, cache_dir=cache_dir)

        q2d_ckpt = '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/log/main-08-04-2021__19:52:35/checkpoint-119500'
        model_q2d = BartForConditionalGeneration.from_pretrained(q2d_ckpt, cache_dir=cache_dir)
        model_q2d = model_q2d.to(device)
        tokenizer_q2d = BartTokenizer.from_pretrained(q2d_ckpt, cache_dir=cache_dir)
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
        greedy_output = model_d2q_second_half.generate(**model_inputs, max_length=256,num_return_sequences=3)
        question_text = tokenizer_d2q_second_half.decode(greedy_output[-1], skip_special_tokens=True)
        # print('_____________________________________________________________________-')
        # print('_____________________________________________________________________-')
        # print('ORIGINAL : %s ' % inputs)
        # for i in range(3):
        #     print('###################%d###################' %i)
        #     print(tokenizer_d2q_second_half.decode(greedy_output[i], skip_special_tokens=True))

        new_pair = {}
        new_pair["question_text"] = question_text
        new_pair["decomposition"] = examples['decomposition']

        return new_pair

    output_text = os.path.join(output_path, 'last_3_questions_119500.txt')
    output_text_2 = os.path.join(output_path, 'reverse_decomp_119500.txt')
    output_text_f = open(output_text, 'a')
    output_text_2_f = open(output_text_2, 'a')

    def change_q_to_predict_second_half(examples):
        targets = examples['decomposition']

        model_inputs = tokenizer_d2q_first_half(targets, return_tensors="pt")
        model_inputs = model_inputs.to(device)
        greedy_output = model_d2q_first_half.generate(**model_inputs, max_length=256,num_return_sequences=3)
        question_text = tokenizer_d2q_first_half.decode(greedy_output[-1], skip_special_tokens=True)

        last_3_test = '______________________________________________________\n' \
                      'ORIGINAL : %s\n' % inputs
        for i in range(3):
            last_3_test = last_3_test + '#############%d##################\n %s \n' % (i,
                                                                                       tokenizer_d2q_first_half.decode(greedy_output[i], skip_special_tokens=True))
        output_text_f.write(last_3_test)

        q2d_inputs = tokenizer_q2d(question_text, return_tensors="pt")
        q2d_inputs = q2d_inputs.to(device)
        greedy_output_decomp = model_q2d.generate(**q2d_inputs, max_length=256)
        new_deccomp = tokenizer_q2d.decode(greedy_output_decomp[0], skip_special_tokens=True)
        reverse_decomp_txt = '______________________________________________________\n' \
                      'ORIGINAL_Q : %s\n ORIGINAL_D : %s\n PRED_Q: %s\n PRED_D: %s\n' % (inputs, targets, question_text, new_deccomp )
        print(reverse_decomp_txt)
        output_text_2_f.write(reverse_decomp_txt)

        new_pair = {}
        new_pair["question_text"] = question_text
        new_pair["decomposition"] = examples['decomposition']

        return new_pair


    # train_dataset_first_half.map(change_q_to_predict_first_half)
    train_dataset_second_half.map(change_q_to_predict_second_half)
    # train_dataset = BreakDataset('train')

    output_path_1 = os.path.join(output_path + 'first_half_data_third_best')
    output_path_2 = os.path.join(output_path + 'second_half_data_third_best_new')

    # train_dataset_first_half.data.save_to_disk(output_path_1)
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
    trained_on_first_half_ckpt = '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/log/main-12-04-2021__21:32:48-D2Q_trained_on_first_half/checkpoint-44000'
    # trained_on_second_half_ckpt = '/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/log/main-12-04-2021__10:35:55-D2Q_trained_on_second_half/checkpoint-44000'

    create_unite_dataset(trained_on_first_half_ckpt, trained_on_second_half_ckpt)
