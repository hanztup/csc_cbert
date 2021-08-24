#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file  : csc_dataset.py
@author: hanzhang yang
@contact : you don't have to
@date  : 2021/07/16 17:07
@version: 1.0
@desc  :
"""

import os
import json
import numpy as np
from typing import List
from pprint import pprint
from functools import partial
from pypinyin import pinyin, Style

import torch
import tokenizers
from torch.utils.data import Dataset, DataLoader
from tokenizers import BertWordPieceTokenizer


class DetectorDataset(object):
    def __init__(self, vocab_file, config_path, max_length=512):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = BertWordPieceTokenizer(vocab_file)
        self.label_to_idx = {label_item: label_idx for label_idx, label_item in enumerate(DetectorDataset.get_labels())}

        # load pinyin map dict
        with open(os.path.join(config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map tensor
        with open(os.path.join(config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # load pinyin map tensor
        with open(os.path.join(config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

    @classmethod
    def get_labels(cls, ):
        """gets the list of labels for this data set."""
        return ["0", "1"]

    def arrange_inputs(self, sentences):
        input_ids, pinyin_ids = [], []
        max_sent_length = min(max([len(sent) for sent in sentences]) + 2, self.max_length)
        for sentence in sentences:
            # convert sentence to ids
            tokenizer_output = self.tokenizer.encode(sentence)
            bert_tokens = tokenizer_output.ids
            pinyin_tokens = self.convert_sentence_to_pinyin_ids(sentence, tokenizer_output)

            assert len(bert_tokens) == len(pinyin_tokens)
            while len(bert_tokens) < max_sent_length:
                bert_tokens += [0]
                pinyin_tokens.append([0] * 8)

            # assert，token nums should be same as pinyin token nums
            assert len(bert_tokens) <= self.max_length
            assert len(bert_tokens) == len(pinyin_tokens)
            bert_tokens = bert_tokens
            pinyin_tokens = np.array(pinyin_tokens).flatten().tolist()

            input_ids.append(bert_tokens)
            pinyin_ids.append(pinyin_tokens)

        # convert list to tensor
        input_ids = torch.LongTensor(input_ids)
        pinyin_ids = torch.LongTensor(pinyin_ids)
        return input_ids, pinyin_ids

    def convert_sentence_to_pinyin_ids(self, sentence: str, tokenizer_output: tokenizers.Encoding) -> List[List[int]]:
        # get pinyin of a sentence
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        # find chinese character location, and generate pinyin ids
        pinyin_ids = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)

        return pinyin_ids


class CSCDataset(Dataset):
    """the Dataset Class for Chinese Spell Detection Dataset."""
    def __init__(self, directory, prefix, vocab_file, config_path, max_length=512, file_name="all.txt"):
        """
        Args:
            directory: str, path to data directory.
            prefix: str, one of [train/dev/test]
            vocab_file: str, path to the vocab file for model pre-training.
            config_path: str, config_path must contain [pinyin_map.json, id2pinyin.json, pinyin2tensor.json]
            max_length: int,
        """
        super().__init__()
        self.max_length = max_length
        data_file_path = os.path.join(directory, "{}.{}".format(prefix, file_name))
        self.data_items = CSCDataset._read_conll(data_file_path)
        self.tokenizer = BertWordPieceTokenizer(vocab_file)
        self.label_to_idx = {label_item: label_idx for label_idx, label_item in enumerate(CSCDataset.get_labels())}

        # get pinyin of a sentence
        with open(os.path.join(config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map tensor
        with open(os.path.join(config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # load pinyin map tensor
        with open(os.path.join(config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        data_item = self.data_items[idx]
        token_sequence, label_sequence = data_item[0], data_item[1]
        label_sequence = [self.label_to_idx[label_item] for label_item in label_sequence]
        token_sequence = "".join(token_sequence[: self.max_length - 2])
        label_sequence = label_sequence[: self.max_length - 2]
        # convert sentence to ids
        tokenizer_output = self.tokenizer.encode(token_sequence)
        # example of tokenizer_output ->
        # Encoding(num_tokens=77, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
        bert_tokens = tokenizer_output.ids
        label_sequence = self._update_labels_using_tokenize_offsets(tokenizer_output.offsets, label_sequence)
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(token_sequence, tokenizer_output)
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        assert len(bert_tokens) == len(label_sequence)
        input_ids = torch.LongTensor(bert_tokens)
        pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        label = torch.LongTensor(label_sequence)
        return input_ids, pinyin_ids, label

    def _update_labels_using_tokenize_offsets(self, offsets, original_sequence_labels):
        """part of offset sequence [(51, 52), (52, 54)] -> (token index after tokenized, original token index)"""
        update_sequence_labels = []
        for offset_idx, offset_item in enumerate(offsets):
            if offset_idx == 0 or offset_idx == (len(offsets) - 1):
                continue
            update_index, origin_index = offset_item
            current_label = original_sequence_labels[origin_index-1]
            update_sequence_labels.append(current_label)
        update_sequence_labels = [self.label_to_idx["0"]] + update_sequence_labels + [self.label_to_idx["0"]]
        return update_sequence_labels

    @classmethod
    def get_labels(cls, ):
        """gets the list of labels for this data set."""
        return ["0", "1"]

    @staticmethod
    def _read_conll(input_file, delimiter=" "):
        """load ner dataset from CoNLL-format files."""
        dataset_item_lst = []
        with open(input_file, "r", encoding="utf-8") as r_f:
            datalines = r_f.readlines()

        cached_token, cached_label = [], []
        for idx, data_line in enumerate(datalines):
            data_line = data_line.strip()
            if idx != 0 and len(data_line) == 0:
                dataset_item_lst.append([cached_token, cached_label])
                cached_token, cached_label = [], []
            else:
                token_label = data_line.split(delimiter)
                token_data_line, label_data_line = token_label[0], token_label[1]
                cached_token.append(token_data_line)
                cached_label.append(label_data_line)
        return dataset_item_lst

    def convert_sentence_to_pinyin_ids(self, sentence: str, tokenizer_output: tokenizers.Encoding) -> List[List[int]]:
        # get pinyin of a sentence
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):  # 一个字符可能有多个读音，如果某个读音在提前保存的拼音里就使用
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        pinyin_ids = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)

        return pinyin_ids


class CSC_Correct_Dataset(Dataset):
    """the Dataset Class for Chinese Spell Correction Dataset."""
    def __init__(self, directory, prefix, vocab_file, config_path, max_length=512, file_name="all.txt"):
        """
        Args:
            directory: str, path to data directory.
            prefix: str, one of [train/dev/test]
            vocab_file: str, path to the vocab file for model pre-training.
            config_path: str, config_path must contain [pinyin_map.json, id2pinyin.json, pinyin2tensor.json]
            max_length: int,
        """
        super().__init__()
        self.max_length = max_length
        data_file_path = os.path.join(directory, "{}.{}".format(prefix, file_name))
        self.data_items = CSC_Correct_Dataset._read_conll(data_file_path)
        self.tokenizer = BertWordPieceTokenizer(vocab_file)
        self.label_to_idx = self.tokenizer.get_vocab()  # by yuang: tag list为整个词表

        # get pinyin of a sentence
        with open(os.path.join(config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map tensor
        with open(os.path.join(config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # load pinyin map tensor
        with open(os.path.join(config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        data_item = self.data_items[idx]
        token_sequence, label_sequence = data_item[0], data_item[1]

        # 参考TtT实现思路：label句子的结尾是[SEP][SEP]
        token_sequence = token_sequence[: self.max_length - 2]
        label_sequence = label_sequence[: self.max_length - 3] + ['[SEP]']
        if len(label_sequence) > len(token_sequence):
            token_sequence += ['[MASK]'] * (len(label_sequence) - len(token_sequence))
        elif len(label_sequence) < len(token_sequence):
            label_sequence += ['[SEP]'] + ['[PAD]'] * (len(token_sequence) - len(label_sequence))
        else:
            pass
        token_sequence = "".join(token_sequence)
        label_sequence = "".join(label_sequence)

        # convert input sentence to ids
        # example of tokenizer_output -> Encoding(num_tokens=77, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
        tokenizer_output = self.tokenizer.encode(token_sequence)
        bert_tokens = tokenizer_output.ids

        # convert label sentence to ids
        tokenizer_label = self.tokenizer.encode(label_sequence)
        label_sequence = tokenizer_label.ids

        if len(label_sequence) < len(bert_tokens):
            label_sequence += [0] * (len(bert_tokens) - len(label_sequence))  # 0 means [PAD], pad to same length

        # convert input sentence to pinyin ids
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(token_sequence, tokenizer_output)
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        assert len(bert_tokens) == len(label_sequence)
        input_ids = torch.LongTensor(bert_tokens)
        pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        label = torch.LongTensor(label_sequence)
        return input_ids, pinyin_ids, label

    @classmethod
    def get_labels(cls, vocab_file):
        """gets the list of labels for this data set."""
        vocab_dict = BertWordPieceTokenizer(vocab_file).get_vocab()
        return vocab_dict

    @staticmethod
    def _read_conll(input_file, delimiter=" "):
        """load ner dataset from CoNLL-format files."""
        dataset_item_lst = []
        with open(input_file, "r", encoding="utf-8") as r_f:
            datalines = r_f.readlines()

        cached_token, cached_label = [], []
        for idx, data_line in enumerate(datalines):
            data_line = data_line.strip()
            if idx != 0 and len(data_line) == 0:
                dataset_item_lst.append([cached_token, cached_label])
                cached_token, cached_label = [], []
            else:
                token_label = data_line.split(delimiter)
                token_data_line, label_data_line = token_label[0], token_label[1]
                if label_data_line == '[PAD]':  # 当target tag为[PAD]时，不添加tag
                    cached_token.append(token_data_line)
                else:
                    cached_token.append(token_data_line)  # 其余情况（包括input text中包含有[MASK]的情形），均添加
                    cached_label.append(label_data_line)
        return dataset_item_lst

    def convert_sentence_to_pinyin_ids(self, sentence: str, tokenizer_output: tokenizers.Encoding) -> List[List[int]]:
        # get pinyin of a sentence
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):  # 一个字符可能有多个读音，如果某个读音在提前保存的拼音里就使用
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        pinyin_ids = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)

        return pinyin_ids


class CSC_Detect_Dataset(Dataset):
    """the Dataset Class for Chinese Spell Detection Dataset."""
    def __init__(self, directory, prefix, vocab_file, config_path, max_length=512, file_name="all.txt"):
        """
        Args:
            directory: str, path to data directory.
            prefix: str, one of [train/dev/test]
            vocab_file: str, path to the vocab file for model pre-training.
            config_path: str, config_path must contain [pinyin_map.json, id2pinyin.json, pinyin2tensor.json]
            max_length: int,
        """
        super().__init__()
        self.max_length = max_length
        data_file_path = os.path.join(directory, "{}.{}".format(prefix, file_name))
        self.data_items = CSC_Detect_Dataset._read_conll(data_file_path)
        self.tokenizer = BertWordPieceTokenizer(vocab_file)
        self.label_to_idx = {label_item: label_idx for label_idx, label_item in enumerate(CSC_Detect_Dataset.get_labels())}

        # get pinyin of a sentence
        with open(os.path.join(config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map tensor
        with open(os.path.join(config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # load pinyin map tensor
        with open(os.path.join(config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        data_item = self.data_items[idx]
        token_sequence, label_sequence = data_item[0], data_item[1]
        label_sequence = [self.label_to_idx[label_item] for label_item in label_sequence]
        token_sequence = "".join(token_sequence[: self.max_length - 2])
        label_sequence = label_sequence[: self.max_length - 2]
        # convert sentence to ids
        tokenizer_output = self.tokenizer.encode(token_sequence)
        # example of tokenizer_output ->
        # Encoding(num_tokens=77, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
        bert_tokens = tokenizer_output.ids
        label_sequence = self._update_labels_using_tokenize_offsets(tokenizer_output.offsets, label_sequence)
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(token_sequence, tokenizer_output)
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        assert len(bert_tokens) == len(label_sequence)
        input_ids = torch.LongTensor(bert_tokens)
        pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        label = torch.LongTensor(label_sequence)
        return input_ids, pinyin_ids, label

    def _update_labels_using_tokenize_offsets(self, offsets, original_sequence_labels):
        """part of offset sequence [(51, 52), (52, 54)] -> (token index after tokenized, original token index)"""
        update_sequence_labels = []
        for offset_idx, offset_item in enumerate(offsets):
            if offset_idx == 0 or offset_idx == (len(offsets) - 1):
                continue
            update_index, origin_index = offset_item
            current_label = original_sequence_labels[origin_index-1]
            update_sequence_labels.append(current_label)
        update_sequence_labels = [self.label_to_idx["O"]] + update_sequence_labels + [self.label_to_idx["O"]]
        return update_sequence_labels

    @classmethod
    def get_labels(cls, ):
        """gets the list of labels for this data set."""
        return ["O",
                "B-S", "I-S",
                "B-M", "I-M",
                "B-W", "I-W",
                "B-R", "I-R"
                ]

    @staticmethod
    def _read_conll(input_file, delimiter=" "):
        """load ner dataset from CoNLL-format files."""
        dataset_item_lst = []
        with open(input_file, "r", encoding="utf-8") as r_f:
            datalines = r_f.readlines()

        cached_token, cached_label = [], []
        for idx, data_line in enumerate(datalines):
            data_line = data_line.strip()
            if idx != 0 and len(data_line) == 0:
                dataset_item_lst.append([cached_token, cached_label])
                cached_token, cached_label = [], []
            else:
                token_label = data_line.split(delimiter)
                token_data_line, label_data_line = token_label[0], token_label[1]
                cached_token.append(token_data_line)
                cached_label.append(label_data_line)
        return dataset_item_lst

    def convert_sentence_to_pinyin_ids(self, sentence: str, tokenizer_output: tokenizers.Encoding) -> List[List[int]]:
        # get pinyin of a sentence
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):  # 一个字符可能有多个读音，如果某个读音在提前保存的拼音里就使用
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        pinyin_ids = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)

        return pinyin_ids


if __name__ == '__main__':

    from collate_functions import collate_to_max_length

    '''
    for detector task
    '''
    # dataset = CSCDataset(directory="/Users/yuang/PA_tech/text_corrector/ChineseBert/csc_correct_task_yuang/data/sighan/transformed",
    #                      prefix="tff",
    #                      vocab_file="/Users/yuang/PA_tech/text_corrector/ChineseBert/ChineseBERT-base/vocab.txt",
    #                      max_length=256,
    #                      config_path="/Users/yuang/PA_tech/text_corrector/ChineseBert/ChineseBERT-base/config")
    #
    # dataloader = DataLoader(dataset=dataset,
    #                         batch_size=2,
    #                         collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0]),
    #                         drop_last=False)
    #
    # for i, batch in enumerate(dataloader):
    #     print(i)
    #     break



    '''
    for correct task
    '''
    dataset = CSC_Correct_Dataset(
        directory="/Users/yuang/PA_tech/text_corrector/ChineseBert/csc_correct_task_yuang/data/sighan/transformed",
        prefix="tff",
        vocab_file="/Users/yuang/PA_tech/text_corrector/ChineseBert/ChineseBERT-base/vocab.txt",
        max_length=256,
        config_path="/Users/yuang/PA_tech/text_corrector/ChineseBert/ChineseBERT-base/config")

    dataloader = DataLoader(dataset=dataset,
                            batch_size=3,
                            collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0]),
                            drop_last=False)

    for i, batch in enumerate(dataloader):
        print(i)
        input_ids, pinyin_ids, label = batch
        attention_mask = (input_ids != 0).long()
        print(attention_mask)
        print(label)
        break



