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




def create_unite_dataset(cp_half_1, cp_half_2, train_tokenizer, train_model):

    if 'joberant' in os.path.abspath('./'):
        model_d2q_1 = BartForConditionalGeneration.from_pretrained(
            '/home/joberant/nlp_fall_2021/omriefroni/qdmr_project/log/' + cp_half_1 + '/',
            cache_dir='/home/joberant/nlp_fall_2021/omriefroni/.cache')
        tokenizer_d2q_1 = BartTokenizer.from_pretrained(
            '/home/joberant/nlp_fall_2021/omriefroni/qdmr_project/log/' + cp_half_1 + '/',
            cache_dir='/home/joberant/nlp_fall_2021/omriefroni/.cache')

        # model_d2q_2 = BartForConditionalGeneration.from_pretrained(
        #     '/home/joberant/nlp_fall_2021/omriefroni/qdmr_project/log/' + cp_half_2 + '/',
        #     cache_dir='/home/joberant/nlp_fall_2021/omriefroni/.cache')
        # tokenizer_d2q_2 = BartTokenizer.from_pretrained(
        #     '/home/joberant/nlp_fall_2021/omriefroni/qdmr_project/log/' + cp_half_2 + '/',
        #     cache_dir='/home/joberant/nlp_fall_2021/omriefroni/.cache')
    else:
        model_d2q_1 = BartForConditionalGeneration.from_pretrained(
            '/Users/omriefroni/Documents/master/nlp/QDMR_project/' + cp_half_1 + '/')
        tokenizer_d2q_1 = BartTokenizer.from_pretrained(
            '/Users/omriefroni/Documents/master/nlp/QDMR_project/' + cp_half_1 + '/')
        model_d2q_2 = BartForConditionalGeneration.from_pretrained(
            '/Users/omriefroni/Documents/master/nlp/QDMR_project/' + cp_half_2 + '/')
        tokenizer_d2q_2 = BartTokenizer.from_pretrained(
            '/Users/omriefroni/Documents/master/nlp/QDMR_project/' + cp_half_2 + '/')



    train_dataset_1 = BreakDataset('train', which_half=1)
    train_dataset_2 = BreakDataset('train', which_half=2)


    def change_q_to_predict_1(examples):
        inputs = examples['question_text']
        targets = examples['decomposition']

                # input_ids = tokenizer("when the soccer match between Liverpool and M.U. is starting", return_tensors="pt")
                # greedy_output = model_ft.generate(**input_ids, max_length=256)
                # print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

        model_inputs = tokenizer_d2q_1(targets, return_tensors="pt")
        greedy_output = model_d2q_1.generate(**model_inputs, max_length=256)
        q_test = tokenizer_d2q_1.decode(greedy_output[0], skip_special_tokens=True)
        # model_inputs = tokenizer(inputs, max_length=256, padding='max_length', truncation=True)

        # Setup the tokenizer for targets
        with train_tokenizer.as_target_tokenizer():
            labels = train_tokenizer(targets, max_length=256, padding='max_length', truncation=True, return_tensors="pt")
            # labels = tokenizer(targets, max_length=256, padding='max_length', truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        decoder_input_ids = shift_tokens_right(labels['input_ids'], train_model.config.pad_token_id)
        # labels["input_ids"] = [
        labels["input_ids"][labels["input_ids"][:, :] == train_model.config.pad_token_id] = -100
        # labels["input_ids"] = \
        #     [(l if l != tokenizer.pad_token_id else -100) for l in labels["input_ids"][0]]
            # [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        # ]

        try:
            model_inputs["labels"] = labels["input_ids"][0,:]
        except:
            print('Omri')
        model_inputs["decoder_input_ids"] = decoder_input_ids[0,:]
        model_inputs["input_ids"] = model_inputs["input_ids"][0,:]
        model_inputs["attention_mask"] = model_inputs["attention_mask"][0,:]
        model_inputs["question_text"] = q_test
        return model_inputs


    def change_q_to_predict_2(examples):
        inputs = examples['question_text']
        targets = examples['decomposition']

                # input_ids = tokenizer("when the soccer match between Liverpool and M.U. is starting", return_tensors="pt")
                # greedy_output = model_ft.generate(**input_ids, max_length=256)
                # print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

        model_inputs = tokenizer_d2q_2(targets, return_tensors="pt")
        greedy_output = model_d2q_2.generate(**model_inputs, max_length=256)
        q_test = tokenizer_d2q_2.decode(greedy_output[0], skip_special_tokens=True)
        # model_inputs = tokenizer(inputs, max_length=256, padding='max_length', truncation=True)

        # Setup the tokenizer for targets
        with train_tokenizer.as_target_tokenizer():
            labels = train_tokenizer(targets, max_length=256, padding='max_length', truncation=True, return_tensors="pt")
            # labels = tokenizer(targets, max_length=256, padding='max_length', truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        decoder_input_ids = shift_tokens_right(labels['input_ids'], train_model.config.pad_token_id)
        # labels["input_ids"] = [
        labels["input_ids"][labels["input_ids"][:, :] == train_model.config.pad_token_id] = -100
        # labels["input_ids"] = \
        #     [(l if l != tokenizer.pad_token_id else -100) for l in labels["input_ids"][0]]
            # [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        # ]

        try:
            model_inputs["labels"] = labels["input_ids"][0,:]
        except:
            print('Omri')
        model_inputs["decoder_input_ids"] = decoder_input_ids[0,:]
        model_inputs["input_ids"] = model_inputs["input_ids"][0,:]
        model_inputs["attention_mask"] = model_inputs["attention_mask"][0,:]
        model_inputs["question_text"] = q_test
        return model_inputs



    train_dataset_1.map(change_q_to_predict_1)
    train_dataset_2.map(change_q_to_predict_2)

    def preprocess_function(examples):
        inputs = examples['question_text']
        targets = examples['decomposition']
        model_inputs = tokenizer(inputs, max_length=256, padding='max_length', truncation=True, return_tensors="pt")
        # model_inputs = tokenizer(inputs, max_length=256, padding='max_length', truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=256, padding='max_length', truncation=True, return_tensors="pt")
            # labels = tokenizer(targets, max_length=256, padding='max_length', truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        decoder_input_ids = shift_tokens_right(labels['input_ids'], model.config.pad_token_id)
        # labels["input_ids"] = [
        labels["input_ids"][labels["input_ids"][:, :] == model.config.pad_token_id] = -100
        # labels["input_ids"] = \
        #     [(l if l != tokenizer.pad_token_id else -100) for l in labels["input_ids"][0]]
            # [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        # ]

        model_inputs["labels"] = labels["input_ids"][0,:]
        model_inputs["decoder_input_ids"] = decoder_input_ids[0,:]
        model_inputs["input_ids"] = model_inputs["input_ids"][0,:]
        model_inputs["attention_mask"] = model_inputs["attention_mask"][0,:]
        return model_inputs


    train_dataset_unite = BreakDataset('train')
    train_dataset_unite.map(preprocess_function)

    train_dataset_unite.is_unite = True
    train_dataset_unite.data_1 = train_dataset_1.data
    train_dataset_unite.data_2 = train_dataset_2.data

    return train_dataset_unite





if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='dynamic cage deformation')
    parser.add_argument("--name", help="experiment name")
    parser.add_argument("--ckpt", type=str, help="path to model to test")
    parser.add_argument("--log_dir", type=str, help="log directory", default="./log")
    parser.add_argument("--device", type=str, help="device", default="cpu", choices=["cpu", "cuda:0"])
    opt = parser.parse_args()

    if opt.ckpt is not None:
        opt.log_dir = os.path.dirname(opt.ckpt) # log dir will be in the same dir as the checkpoint
    else:
        opt.log_dir = os.path.join(opt.log_dir, "-".join(filter(None, [os.path.basename(__file__)[:-3],
                                                                    datetime.datetime.now().strftime("%d-%m-%Y__%H:%M:%S"),
                                                                    opt.name]))) # log dir will be the file name + date_time + expirement name

    os.makedirs(opt.log_dir, exist_ok=True)
    log_file = open(os.path.join(opt.log_dir, "training_log.txt"), "a")
    log_file.close()

    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large',  cache_dir='/home/joberant/nlp_fall_2021/omriefroni/.cache')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large',  cache_dir='/home/joberant/nlp_fall_2021/omriefroni/.cache')


    create_unite_dataset('main-07-04-2021__21:29:25/checkpoint-9500', '',tokenizer, model )
