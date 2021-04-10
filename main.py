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


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


class BreakDataset(torch.utils.data.Dataset):
    def __init__(self, split='train'):
        super().__init__()
        if 'joberant' in os.path.abspath('./'):
            data = load_dataset('break_data', 'QDMR', cache_dir='/home/joberant/nlp_fall_2021/meitars/.cache')[split]
        else:
            data = load_dataset('break_data', 'QDMR')[split]
        data = data.select(range(10))


        self.split = split
        self.data = data
        self.data = self.data.map(self.prepare_data)

    def prepare_data(self, example):
        decomp = example['decomposition']

        decomp = decomp.replace(';', '@@')
        decomp = decomp.replace('#', '##')
        example['decomposition'] = decomp
        return example

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        return self.data.__getitem__(idx)

    def map(self, preprocess_function):
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
    device = torch.device(opt.device)
    opt.device = device

    ### model declaration ###
    if 'joberant' in os.path.abspath('./'):
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large', cache_dir='/home/joberant/nlp_fall_2021/meitars/.cache')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', cache_dir='/home/joberant/nlp_fall_2021/meitars/.cache')
    else:
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    ### declare dataset ###
    train_dataset = BreakDataset('train')
    val_dataset = BreakDataset('validation')

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
    val_dataset.map(preprocess_function)

    training_args = TrainingArguments(
        output_dir=opt.log_dir,          # output directory
        num_train_epochs=20,              # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=opt.log_dir,         # directory for storing logs
        logging_steps=100,
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        evaluation_strategy='steps',
        eval_steps=500,
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


    # ### load checkpoint if in need ###
    # epoch = None
    # if opt.ckpt:
    #     net, epoch = load_network(net, opt.ckpt, device)

    # ### declare on all relevant losses ###
    # all_losses = AllLosses(opt)

    # ### optimizer ###
    # optimizer = torch.optim.Adam([
    #     {"params": net.encoder.parameters()},
    #     {"params": net.decoder.parameters()}], lr=opt.lr)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(opt.nepochs*0.8), gamma=0.1, last_epoch=-1)

    # ### train ###
    # log_file = open(os.path.join(opt.log_dir, "training_log.txt"), "a")
    # log_file.write("----- Starting training process -----\n")
    # net.train()
    # start_epoch = 0 if epoch is None else epoch
    # t = 0 if epoch is None else start_epoch*len(dataloader) 

    # log_interval = max(len(dataloader)//5, 100)
    # save_interval = max(opt.nepochs//10, 1)
    # running_avg_loss = -1


    # log_file.close()
    # writer.close()
    # save_network(net, opt.log_dir, network_label="net", epoch_label="final")

    # test(opt, net=model, tokenizer=tokenizer)


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
    train(opt)


