import datetime
import os
import sys
import numpy as np
import traceback
import shutil

import torch
import torch.nn as nn

from modules import NetworkFull, weights_init
from utils import BaseOptions, load_network, constructTBPath, clamp_gradient, save_network, deform_with_MLS, build_dataloader
from utils import error, warn, info, success, save_pts, center_bounding_box, crisscross_input, log_outputs
from mylosses import AllLosses
import warnings

from transformers import BartModel, BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments

# model = BartModel.from_pretrained('facebook/bart-large')

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state
ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

from datasets import load_dataset

class BreakDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, split='train'):
        super().__init__()
        data = load_dataset('break_data', 'QDMR')[split]
        data = prepare_data(data, tokenizer)

        self.split = split
        self.data = data

    def prepare_data(data, tokenizer):
        dataset = []
        for i, example in enumerate(data):
            question = example['question_text']
            decomp = example['decomposition']

            decomp = decomp.replace(';', '@@')
            decomp = decomp.replace('#', '##')

            # q_encoding = tokenizer([question, decomp], return_tensors='pt', padding=True, truncation=True)
            q_encoding = tokenizer(question, return_tensors='pt', padding=True, truncation=True)
            d_encoding = tokenizer(decomp, return_tensors='pt', padding=True, truncation=True)

            dataset.append({'input_ids':d_encoding['input_ids'], 'attention_mask':d_encoding['attention_mask'], 'decoder_input_ids':q_encoding['input_ids'], 'labels':q_encoding['input_ids']})
        
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    ### declare dataset ###
    # dataset = load_dataset('break_data', 'QDMR')
    train_dataset = BreakDataset(tokenizer, 'train')
    val_dataset = BreakDataset(tokenizer, 'validation')

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )

    trainer.train()




    ### load checkpoint if in need ###
    epoch = None
    if opt.ckpt:
        net, epoch = load_network(net, opt.ckpt, device)

    ### declare on all relevant losses ###
    all_losses = AllLosses(opt)

    ### optimizer ###
    optimizer = torch.optim.Adam([
        {"params": net.encoder.parameters()},
        {"params": net.decoder.parameters()}], lr=opt.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(opt.nepochs*0.8), gamma=0.1, last_epoch=-1)

    ### train ###
    log_file = open(os.path.join(opt.log_dir, "training_log.txt"), "a")
    log_file.write("----- Starting training process -----\n")
    net.train()
    start_epoch = 0 if epoch is None else epoch
    t = 0 if epoch is None else start_epoch*len(dataloader) 

    log_interval = max(len(dataloader)//5, 100)
    save_interval = max(opt.nepochs//10, 1)
    running_avg_loss = -1

    # with torch.autograd.detect_anomaly():
    if opt.epoch:
        start_epoch = opt.epoch % opt.nepochs
        t += start_epoch*len(dataloader)

    for epoch in range(start_epoch, opt.nepochs):
        for t_epoch, data in enumerate(dataloader):
            warming_up = epoch < opt.warmup_epochs

            ############# get data ###########
            data["source_shape"] = data["source_shape"].detach().to(device) 
            data["target_shape"] = data["target_shape"].detach().to(device) 

            ############# run network ###########
            optimizer.zero_grad()
            outputs = net(source_shape_t, target_shape_t)

            ############# get losses ###########            
            current_loss = all_losses(data, outputs, progress)

            if running_avg_loss < 0:
                running_avg_loss = loss_sum
            else:
                running_avg_loss = running_avg_loss + (loss_sum.item() - running_avg_loss)/(t+1)

            if (t % log_interval == 0):                
                log_str = "Epoch: {:03d}. t: {:05d}: {}.".format(epoch+1, t+1, ", ".join(["{} {:.3g}".format(k, v.mean().item()) for k, v in current_loss.items()]))
                print(log_str)
                log_file.write(log_str+"\n")
                log_outputs(opt, t, outputs, data, running_avg_loss, current_loss, writer=writer)

            loss_sum.backward()
            optimizer.step()

            if (t + 1) % 500 == 0:
                save_network(net, opt.log_dir, network_label="net", epoch_label="latest", epoch=epoch)

            t += 1

        print("--- Done epoch {} --- loss: {}".format(epoch+1, ", ".join(["{} {:.3g}".format(k, v.mean().item()) for k, v in current_loss.items()])))

        if (epoch + 1) % save_interval == 0:
            save_network(net, opt.log_dir, network_label="net", epoch_label=epoch+1, epoch=epoch)

        scheduler.step()
        # test(opt, net=net, save_subdir="epoch_{}".format(epoch+1))

    log_file.close()
    writer.close()
    save_network(net, opt.log_dir, network_label="net", epoch_label="final")
    test(opt, net=net)


def test(opt, net=None, save_subdir="test"):
    opt.phase = "test"
    opt.batch_size = 1
    dataloader = build_dataloader(opt)

    if net is None:
        # network
        net = NetworkFull(opt, dim=opt.dim, bottleneck_size=opt.bottleneck_size).to(opt.device)
        net.eval()
        load_network(net, opt.ckpt)
    else:
        net.eval()

    test_output_dir = os.path.join(opt.log_dir, save_subdir)
    os.makedirs(test_output_dir, exist_ok=True)
    with open(os.path.join(test_output_dir, "eval.txt"), "w") as f:
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                source_shape_t = data["source_shape"].to(opt.device).contiguous()
                target_shape_t = data["target_shape"].to(opt.device).contiguous()
                outputs = net(source_shape_t, target_shape_t)



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


