from datasets import load_dataset
import datasets
import torch
import torch.nn as nn
import os
from global_params import *


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


class BreakDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', which_half='all', augmentation_path=None, without_regular=False):
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

        data = data.map(self.prepare_data)
        if augmentation_path is not None:
            # data_first_half = datasets.load_from_disk(os.path.join(augmentation_path, 'first_half_data'))
            # data_second_half = datasets.load_from_disk(os.path.join(augmentation_path, 'second_half_data'))
            # data_first_half_3 = datasets.load_from_disk(os.path.join(augmentation_path, 'first_half_data_third_best'))
            # data_second_half_3 = datasets.load_from_disk(os.path.join(augmentation_path, 'second_half_data_third_best_new'))
            inverse_data = datasets.load_from_disk(os.path.join(augmentation_path, 'augmented_decomp_based_on_main-08-04-2021__19:52:35_checkpoint-119500'))
            print('Len data before: %d' % len(data))
            if without_regular:
                # data = datasets.concatenate_datasets([data_first_half, data_second_half])
                print('Load only 2 datasets')
            else:
                data = datasets.concatenate_datasets([data, inverse_data])
                    # [data, data_first_half, data_second_half, data_first_half_3, data_second_half_3])


            print('Len data after (with inverse only): %d' %len(data))

        self.split = split
        self.data = data
        self.which_half = which_half

        # if True: #which_half == 'all':
        #     self.data = self.data.map(self.prepare_data)
        # else:
        #     self.data.dataset = self.data.dataset.map(self.prepare_data)

    def prepare_data(self, example):
        decomp = example['decomposition']

        decomp = decomp.replace(';', '@@')
        decomp = decomp.replace('#', '##')
        decomp = decomp.replace('####', '##')

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
