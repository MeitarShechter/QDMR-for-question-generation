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
# from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

# model = BartModel.from_pretrained('facebook/bart-large')

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state

from datasets import load_dataset

# # @dataclass
# class DataCollatorForSeq2Seq:
#     """
#     Data collator that will dynamically pad the inputs received, as well as the labels.
#     Args:
#         tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
#             The tokenizer used for encoding the data.
#         model (:class:`~transformers.PreTrainedModel`):
#             The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
#             prepare the `decoder_input_ids`
#             This is useful when using `label_smoothing` to avoid calculating loss twice.
#         padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
#             Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
#             among:
#             * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
#               sequence is provided).
#             * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
#               maximum acceptable input length for the model if that argument is not provided.
#             * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
#               different lengths).
#         max_length (:obj:`int`, `optional`):
#             Maximum length of the returned list and optionally padding length (see above).
#         pad_to_multiple_of (:obj:`int`, `optional`):
#             If set will pad the sequence to a multiple of the provided value.
#             This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
#             7.5 (Volta).
#         label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
#             The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
#     """

#     tokenizer: PreTrainedTokenizerBase
#     model = None
#     padding = True
#     max_length: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None
#     label_pad_token_id: int = -100

#     def __call__(self, features):
#         labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
#         # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
#         # same length to return tensors.
#         if labels is not None:
#             max_label_length = max(len(l) for l in labels)
#             padding_side = self.tokenizer.padding_side
#             for feature in features:
#                 remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
#                 feature["labels"] = (
#                     feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
#                 )

#         features = self.tokenizer.pad(
#             features,
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors="pt",
#         )

#         # prepare decoder_input_ids
#         if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
#             decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
#             features["decoder_input_ids"] = decoder_input_ids

#         return features

def prepare_data(example):
    decomp = example['decomposition']

    decomp = decomp.replace(';', '@@')
    decomp = decomp.replace('#', '##')
    example['decomposition'] = decomp
    return example


class BreakDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, split='train'):
        super().__init__()
        data = load_dataset('break_data', 'QDMR')[split]
        # data = self.prepare_data(data, tokenizer)

        self.split = split
        self.data = data
        self.data = self.data.map(prepare_data)

    # def prepare_data(self, example):
    #     decomp = example['decomposition']
    #
    #     decomp = decomp.replace(';', '@@')
    #     decomp = decomp.replace('#', '##')
    #     example['decomposition'] = decomp
    # def prepare_data(self, data, tokenizer):
    #     # dataset = []
    #     for i, example in enumerate(data):
    #         # question = example['question_text']
    #         decomp = example['decomposition']
    #
    #         decomp = decomp.replace(';', '@@')
    #         decomp = decomp.replace('#', '##')
    #
    #         # q_encoding = tokenizer(question, return_tensors='pt', padding=True, truncation=True)
    #         # d_encoding = tokenizer(decomp, return_tensors='pt', padding=True, truncation=True)
    #
    #         # dataset.append({'input_ids':d_encoding['input_ids'], 'attention_mask':d_encoding['attention_mask'], 'decoder_input_ids':q_encoding['input_ids'], 'labels':q_encoding['input_ids']})
    #         # dataset.append({'question_text':question, 'decomposition':decomp})
    #         example['decomposition'] = decomp
    #
    #     return data

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        return self.data.__getitem__(idx)

    def map(self, preprocess_function):
        self.data = self.data.map(preprocess_function, batched=True)


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
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', )

    ### declare dataset ###
    # dataset = load_dataset('break_data', 'QDMR')
    train_dataset = BreakDataset(tokenizer, 'train')
    val_dataset = BreakDataset(tokenizer, 'validation')
    def preprocess_function(examples):
        inputs = examples['decomposition']
        targets = examples['question_text']
        model_inputs = tokenizer(inputs, max_length=256, padding='max_length', truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=256, padding='max_length', truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.    
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset.map(preprocess_function)
    val_dataset.map(preprocess_function)

    training_args = TrainingArguments(
        output_dir=opt.log_dir,          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=opt.log_dir,         # directory for storing logs
        logging_steps=10,
        evaluation_strategy='epoch',
    )

    data_collator = default_data_collator

    # trainer = Trainer(
    #     model=model,                         # the instantiated 🤗 Transformers model to be trained
    #     args=training_args,                  # training arguments, defined above
    #     train_dataset=train_dataset,         # training dataset
    #     eval_dataset=val_dataset             # evaluation dataset
    # )

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

    # # with torch.autograd.detect_anomaly():
    # if opt.epoch:
    #     start_epoch = opt.epoch % opt.nepochs
    #     t += start_epoch*len(dataloader)

    # for epoch in range(start_epoch, opt.nepochs):
    #     for t_epoch, data in enumerate(dataloader):
    #         warming_up = epoch < opt.warmup_epochs

    #         ############# get data ###########
    #         data["source_shape"] = data["source_shape"].detach().to(device) 
    #         data["target_shape"] = data["target_shape"].detach().to(device) 

    #         ############# run network ###########
    #         optimizer.zero_grad()
    #         outputs = net(source_shape_t, target_shape_t)

    #         ############# get losses ###########            
    #         current_loss = all_losses(data, outputs, progress)

    #         if running_avg_loss < 0:
    #             running_avg_loss = loss_sum
    #         else:
    #             running_avg_loss = running_avg_loss + (loss_sum.item() - running_avg_loss)/(t+1)

    #         if (t % log_interval == 0):                
    #             log_str = "Epoch: {:03d}. t: {:05d}: {}.".format(epoch+1, t+1, ", ".join(["{} {:.3g}".format(k, v.mean().item()) for k, v in current_loss.items()]))
    #             print(log_str)
    #             log_file.write(log_str+"\n")
    #             log_outputs(opt, t, outputs, data, running_avg_loss, current_loss, writer=writer)

    #         loss_sum.backward()
    #         optimizer.step()

    #         if (t + 1) % 500 == 0:
    #             save_network(net, opt.log_dir, network_label="net", epoch_label="latest", epoch=epoch)

    #         t += 1

    #     print("--- Done epoch {} --- loss: {}".format(epoch+1, ", ".join(["{} {:.3g}".format(k, v.mean().item()) for k, v in current_loss.items()])))

    #     if (epoch + 1) % save_interval == 0:
    #         save_network(net, opt.log_dir, network_label="net", epoch_label=epoch+1, epoch=epoch)

    #     scheduler.step()
    #     # test(opt, net=net, save_subdir="epoch_{}".format(epoch+1))

    # log_file.close()
    # writer.close()
    # save_network(net, opt.log_dir, network_label="net", epoch_label="final")
    test(opt, net=model, tokenizer=tokenizer)


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
    #     with torch.no_grad():
    #         for i, data in enumerate(dataloader):
    #             source_shape_t = data["source_shape"].to(opt.device).contiguous()
    #             target_shape_t = data["target_shape"].to(opt.device).contiguous()
    #             outputs = net(source_shape_t, target_shape_t)


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


