import datetime
import os
import sys
# import numpy as np
import traceback
import shutil
import argparse

import torch
import torch.nn as nn

from transformers import BartModel, BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, TrainingArguments, default_data_collator, Trainer
# from transformers.modeling_bart import shift_tokens_right

from datasets import load_dataset

# user_name = 'meitars'
user_name = 'omriefroni'
cache_dir = '/home/joberant/nlp_fall_2021/' + user_name + '/.cache'
os.environ["TRANSFORMERS_CACHE"] = cache_dir 

def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


class BreakDataset(torch.utils.data.Dataset):
    def __init__(self, split='traiיn', which_half='all'):
        super().__init__()
        if 'joberant' in os.path.abspath('./'):
            data = load_dataset('break_data', 'QDMR', cache_dir=cache_dir)[split]
        else:
            data = load_dataset('break_data', 'QDMR')[split]

        # data = data.select(range(10))
        print("Creating a BreakDataset instance for {} half(s)".format(which_half))
        if which_half == 'first':                
            # data = torch.utils.data.Subset(data, range(0, len(data)//2))
            data = data.select(range(0, len(data)//2))
        elif which_half == 'second':
            # data = torch.utils.data.Subset(data, range(len(data)//2, len(data)))
            data = data.select(range(len(data)//2, len(data)))

        self.split = split
        self.data = data
        self.which_half = which_half

        if True: #which_half == 'all':
            self.data = self.data.map(self.prepare_data)
        else:
            self.data.dataset = self.data.dataset.map(self.prepare_data)

    def prepare_data(self, example):
        decomp = example['decomposition']

        decomp = decomp.replace(';', '@@')
        decomp = decomp.replace('#', '##')
        example['decomposition'] = decomp
        return example

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def map(self, preprocess_function):
        if type(self.data) is torch.utils.data.dataset.Subset:
            self.data.dataset = self.data.dataset.map(preprocess_function)
        else:
            self.data = self.data.map(preprocess_function)
        # self.data = self.data.map(preprocess_function, batched=True)


def train(opt):
    ### randomization-related stuff ###
    # random.seed(0)
    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)
    # torch.cuda.manual_seed(0)

    ### choose device ###
    # device = torch.device(opt.device)
    # opt.device = device

    ### model declaration ###
    if 'joberant' in os.path.abspath('./'):
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large', cache_dir=cache_dir)
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', cache_dir=cache_dir)
    else:
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    ### declare dataset ###
    train_dataset = BreakDataset('train', which_half=opt.dataset_half)
    val_dataset = BreakDataset('validation', which_half=opt.dataset_half)

    def preprocess_function(examples):
        if opt.Q2D:
            inputs = examples['question_text']
            targets = examples['decomposition']
        else:
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
    val_dataset.map(preprocess_function)

    training_args = TrainingArguments(
        output_dir=opt.log_dir,          # output directory
        num_train_epochs=4,              # total number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=2,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=opt.log_dir,         # directory for storing logs
        logging_steps=100,
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=2000,
        evaluation_strategy='epoch',
        # eval_steps=500,
        learning_rate=5e-5,              # default
        # no_cuda=True
    )

    data_collator = default_data_collator

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()    


def test(opt, net, tokenizer, save_subdir="test"):
    opt.phase = "test"
    opt.batch_size = 1
    # test_dataset = BreakDataset(tokenizer, 'test')

    net.eval()

    QDMR_TO_GENERATE = "return organizations @@ return number of ##1"
    inputs = tokenizer([QDMR_TO_GENERATE], max_length=1024, return_tensors='pt')
    question_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
    print([tokenizer.decode(q, skip_special_tokens=True, clean_up_tokenization_spaces=False) for q in question_ids])

    # test_output_dir = os.path.join(opt.log_dir, save_subdir)
    # os.makedirs(test_output_dir, exist_ok=True)
    # with open(os.path.join(test_output_dir, "eval.txt"), "w") as f:

    model_ft = BartForConditionalGeneration.from_pretrained('/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/log/main-08-04-2021__19:52:35/checkpoint-60500/', cache_dir='/home/joberant/nlp_fall_2021/meitars/.cache')
    tokenizer = BartTokenizer.from_pretrained('/home/joberant/nlp_fall_2021/meitars/QDMR-for-question-generation/log/main-08-04-2021__19:52:35/checkpoint-60500/', cache_dir='/home/joberant/nlp_fall_2021/meitars/.cache')

    input_ids = tokenizer("when the soccer match between Liverpool and M.U. is starting", return_tensors="pt")
    greedy_output = model_ft.generate(**input_ids, max_length=256)
    print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='dynamic cage deformation')
    parser.add_argument("--name", help="experiment name")
    parser.add_argument("--ckpt", type=str, help="path to model to test")
    parser.add_argument("--log_dir", type=str, help="log directory", default="./log")
    parser.add_argument("--device", type=str, help="device", default="cpu", choices=["cpu", "cuda:0"])
    parser.add_argument("--Q2D", action="store_true", help="which direction we want our transformer, if true then Question to Decomposition")
    parser.add_argument("--dataset_half", type=str, choices=["first", "second", "all"], default="all")
    opt = parser.parse_args()

    if opt.ckpt is not None:
        opt.log_dir = os.path.dirname(opt.ckpt) # log dir will be in the same dir as the checkpoint
    else:
        opt.log_dir = os.path.join(opt.log_dir, "-".join(filter(None, [os.path.basename(__file__)[:-3],
                                                                    datetime.datetime.now().strftime("%d-%m-%Y__%H:%M:%S"),
                                                                    opt.name]))) # log dir will be the file name + date_time + expirement name

    os.makedirs(opt.log_dir, exist_ok=True)
    # log_file = open(os.path.join(opt.log_dir, "training_log.txt"), "a")
    # log_file.close()
    train(opt)


