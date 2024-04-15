import argparse
import json
import os
import sys

from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class LoRA_Dataset(Dataset):

    def __init__(self, args, tokenizer):
        self.data = list()
        self.args = args
        self.tokenizer = tokenizer
        self.get_dataset()

    def get_dataset(self):
        #构建jsonl文件列表
        data_path = Path(os.path.join(self.args.data_path))
        jsonl_list = [str(p) for p in data_path.glob('*.jsonl')]
        raw_data = load_dataset('json', data_files=jsonl_list)
        for i, item in enumerate(raw_data['train']):
            src = item['func_src_before']
            diffs = item['char_changes']['deleted']
            data = self.add_data(src, diffs, i, item['file_name'].split('.')[-1])
            if data is not None:
                self.data.append(data)


    def add_data(self, src, changes, vul_id, lang):

        encoded = self.tokenizer.encode_plus(src)
        if len(encoded['input_ids']) > self.args.max_num_tokens: return None
        min_changed_tokens = (2 if self.args.vul_type in ('cwe-invalid', 'cwe-valid') else 1)

        if len(changes) == 0:
            weights = [1] * len(encoded['input_ids'])
        else:
            weights = [0] * len(encoded['input_ids'])
        for change in changes:
            char_start = change['char_start']
            char_start_idx = encoded.char_to_token(char_start)
            char_end = change['char_end']
            char_end_idx = encoded.char_to_token(char_end - 1)
            for char_idx in range(char_start_idx, char_end_idx + 1):
                weights[char_idx] = 1
        if sum(weights) < min_changed_tokens: return None
        if len(encoded['input_ids']) - sum(weights) < min_changed_tokens: return None

        return encoded['input_ids'], weights, vul_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        return {
            'input_ids': torch.tensor(self.data[item][0]),
            'weights': torch.tensor(self.data[item][1]),
            'vul_id': torch.tensor(self.data[item][2]),
        }



