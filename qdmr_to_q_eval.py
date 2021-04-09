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

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state

from datasets import load_dataset
from main import BreakDataset, shift_tokens_right


def eval_qdmr_to_q(opt, checkpoint_path_name):

    ## choose device ###
    device = torch.device(opt.device)
    opt.device = device

    ### model declaration ###
    if 'joberant' in os.path.abspath('./'):
        model_ft = BartForConditionalGeneration.from_pretrained(
            '/home/joberant/nlp_fall_2021/omriefroni/QDMR-for-question-generation/log/main-08-04-2021__19:52:35/' + checkpoint_path_name + '/',
            cache_dir='/home/joberant/nlp_fall_2021/meitars/.cache')
        tokenizer = BartTokenizer.from_pretrained(
            '/home/joberant/nlp_fall_2021/omriefroni/QDMR-for-question-generation/log/main-08-04-2021__19:52:35/' + checkpoint_path_name + '/',
            cache_dir='/home/joberant/nlp_fall_2021/meitars/.cache')
    else:
        model_ft = BartForConditionalGeneration.from_pretrained(
            '/Users/omriefroni/Documents/master/nlp/QDMR_project/' + checkpoint_path_name + '/')
        tokenizer = BartTokenizer.from_pretrained(
            '/Users/omriefroni/Documents/master/nlp/QDMR_project/' + checkpoint_path_name + '/')

    train_dataset = BreakDataset('train')

    def preprocess_function(examples):
        inputs = examples['decomposition']
        targets = examples['question_text']
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

    train_dataset.map(preprocess_function)

    training_args = TrainingArguments(
        output_dir=opt.log_dir,          # output directory
        num_train_epochs=2000,              # total number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=50,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=opt.log_dir,         # directory for storing logs
        logging_steps=1,
        evaluation_strategy='epoch',
    )


    data_collator = default_data_collator

    trainer = Trainer(
        model=model,
        args=training_args,
        test_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    result = trainer.predict()

    input_ids = tokenizer("when the soccer match between Liverpool and M.U. is starting", return_tensors="pt")
    greedy_output = model_ft.generate(**input_ids, max_length=256)
    print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))


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
    eval_qdmr_to_q(opt, 'checkpoint-99500')
