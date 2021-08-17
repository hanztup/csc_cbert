#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file  : corrector_trainer.py
@author: hanzhang yang
@contact : you don't have to
@date  : 2021/08/02 09:49
@version: 1.0
@desc  : Develop function of CSC correct
"""
import sys
sys.path.append('..')

import os
import re
import json
import argparse
import logging
from pprint import pprint
from functools import partial
from collections import namedtuple

from datasets.collate_functions import collate_to_max_length
from datasets.csc_dataset import CSC_Correct_Dataset, DetectorDataset
from models.modeling_glycebert import GlyceBertForTokenClassification
from models.crf_layer import DynamicCRF
from utils.random_seed import set_random_seed
from metrics.csc_metric import MetricForCSC_Corrector

# enable reproducibility
# https://pytorch-lightning.readthedocs.io/en/latest/trainer.html
set_random_seed(2333)

import torch
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
# from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class CSCCorrectTask(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        """Initialize a model, tokenizer and config"""
        super().__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        else:
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)

        self.detect_labels = CSC_Correct_Dataset.get_labels(os.path.join(self.args.bert_path, "vocab.txt"))
        self.bert_dir = args.bert_path
        self.num_labels = len(self.detect_labels)
        self.bert_config = BertConfig.from_pretrained(self.bert_dir, output_hidden_states=False,
                                                      num_labels=self.num_labels,
                                                      hidden_dropout_prob=self.args.hidden_dropout_prob)
        self.model = GlyceBertForTokenClassification.from_pretrained(self.bert_dir,
                                                                     config=self.bert_config,
                                                                     mlp=False if self.args.classifier=="single" else True)
        self.CRF_layer = DynamicCRF(self.num_labels)
        self.loss_type = args.loss_type  # todo: 补充args中的参数

        self.ner_evaluation_metric = MetricForCSC_Corrector(entity_labels=self.detect_labels, save_prediction=self.args.save_ner_prediction)

        format = '%(asctime)s - %(name)s - %(message)s'
        logging.basicConfig(format=format, filename=os.path.join(self.args.save_path, "eval_result_log.txt"), level=logging.INFO)
        self.result_logger = logging.getLogger(__name__)
        self.result_logger.setLevel(logging.INFO)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        # todo: 更新crf层参数
        no_decay = ["bias", "LayerNorm.weight"]
        model_params = list(self.model.named_parameters()) + \
                       list(self.CRF_layer.named_parameters()) if 'CRF' in self.loss_type else list(self.model.named_parameters())
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model_params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.args.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999), lr=self.args.lr, eps=self.args.adam_epsilon, )
        elif self.args.optimizer == "torch.adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          lr=self.args.lr,
                                          eps=self.args.adam_epsilon,
                                          weight_decay=self.args.weight_decay)
        else:
            raise ValueError("Please import the Optimizer first. ")
        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (
                self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        warmup_steps = int(self.args.warmup_proportion * t_total)
        if self.args.no_lr_scheduler:
            return [optimizer]
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, pinyin_ids):
        attention_mask = (input_ids != 0).long()
        sequence_logits = self.model(input_ids, pinyin_ids, attention_mask=attention_mask)[0]  # also note as the "sequence_emissions"
        decode_result = self.CRF_layer.decode(sequence_logits, mask=attention_mask.byte())

        return sequence_logits, decode_result

    def compute_loss(self, logits, labels, loss_mask=None):
        """
        Desc:
            compute cross entropy loss
            todo: add focal loss !
        Args:
            logits: FloatTensor, shape of [batch_size, sequence_len, num_labels]
            labels: LongTensor, shape of [batch_size, sequence_len, num_labels]
            loss_mask: Optional[LongTensor], shape of [batch_size, sequence_len].
                1 for non-PAD tokens, 0 for PAD tokens.
        """
        loss_fct = CrossEntropyLoss()
        if loss_mask is not None:
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # crf loss
        loss_crf = -self.CRF_layer(logits, labels, mask=loss_mask.byte(), reduction='token_mean')

        if 'CRF' in self.loss_type:
            return loss + loss_crf
        else:
            return loss

    def training_step(self, batch, batch_idx):
        input_ids, pinyin_ids, labels = batch
        loss_mask = (input_ids != 0).long()
        batch_size, seq_len = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, seq_len, 8)
        sequence_logits, decode_result = self.forward(input_ids=input_ids, pinyin_ids=pinyin_ids,)
        loss = self.compute_loss(sequence_logits, labels, loss_mask=loss_mask)

        tf_board_logs = {
            "train_loss": loss,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"]
        }
        return {"loss": loss, "log": tf_board_logs}

    def validation_step(self, batch, batch_idx):
        input_ids, pinyin_ids, gold_labels = batch
        batch_size, seq_len = input_ids.shape
        loss_mask = (input_ids != 0).long()
        pinyin_ids = pinyin_ids.view(batch_size, seq_len, 8)
        sequence_logits, decode_result = self.forward(input_ids=input_ids, pinyin_ids=pinyin_ids,)
        loss = self.compute_loss(sequence_logits, gold_labels, loss_mask=loss_mask)
        confusion_matrix = self.ner_evaluation_metric(decode_result[1], gold_labels, sequence_mask=loss_mask)
        return {"val_loss": loss, "confusion_matrix": confusion_matrix}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        confusion_matrix = torch.stack([x[f"confusion_matrix"] for x in outputs]).sum(0)
        all_pp, all_rr, all_ff = confusion_matrix
        sample_nums = len(outputs) * self.args.eval_batch_size
        precision = all_pp / sample_nums
        recall = all_rr / sample_nums
        f1 = all_ff / sample_nums

        self.result_logger.info(f"EVAL INFO -> current_epoch is: {self.trainer.current_epoch}, current_global_step is: {self.trainer.global_step} ")
        self.result_logger.info(f"EVAL INFO -> valid_f1 is: {f1}")
        tensorboard_logs = {"val_loss": avg_loss, "val_f1": f1, }
        return {"val_loss": avg_loss, "val_log": tensorboard_logs, "val_f1": f1, "val_precision": precision, "val_recall": recall}

    def train_dataloader(self,) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self, ) -> DataLoader:
        return self.get_dataloader("test")

    def _load_dataset(self, prefix="test"):
        dataset = CSC_Correct_Dataset(directory=self.args.data_dir,
                                      prefix=prefix,
                                      vocab_file=os.path.join(self.args.bert_path, "vocab.txt"),
                                      max_length=self.args.max_length,
                                      config_path=os.path.join(self.args.bert_path, "config"))
        return dataset

    def get_dataloader(self, prefix="train", limit=None) -> DataLoader:
        """return {train/dev/test} dataloader"""
        dataset = self._load_dataset(prefix=prefix)

        # if prefix == "train":
        if prefix.startswith("train"):
            batch_size = self.args.train_batch_size
            # small dataset like weibo ner, define data_generator will help experiment reproducibility.
            data_generator = torch.Generator()
            data_generator.manual_seed(self.args.seed)
            data_sampler = RandomSampler(dataset, generator=data_generator)
        else:
            batch_size = self.args.eval_batch_size
            data_sampler = SequentialSampler(dataset)

        # sampler option is mutually exclusive with shuffle
        dataloader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size,
                                num_workers=self.args.workers, collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0]),
                                drop_last=False)

        return dataloader

    def test_dataloader(self, ) -> DataLoader:
        return self.get_dataloader("test")

    def test_step(self, batch, batch_idx):
        input_ids, pinyin_ids, gold_labels = batch
        sequence_mask = (input_ids != 0).long()
        batch_size, seq_len = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, seq_len, 8)
        sequence_logits, decode_result = self.forward(input_ids=input_ids, pinyin_ids=pinyin_ids,)
        confusion_matrix = self.ner_evaluation_metric(decode_result[1], gold_labels, sequence_mask=sequence_mask)
        return {"confusion_matrix": confusion_matrix}

    def test_epoch_end(self, outputs):
        confusion_matrix = torch.stack([x[f"confusion_matrix"] for x in outputs]).sum(0)
        all_pp, all_rr, all_ff = confusion_matrix
        sample_nums = len(outputs) * self.args.eval_batch_size
        precision = all_pp / sample_nums
        recall = all_rr / sample_nums
        f1 = all_ff / sample_nums

        if self.args.save_ner_prediction:
            gold_entity_lst, pred_entity_lst = self.ner_evaluation_metric.gold_entity_lst, self.ner_evaluation_metric.pred_entity_lst
            self.save_predictions_to_file(gold_entity_lst, pred_entity_lst)

        tensorboard_logs = {"test_f1": f1,}
        self.result_logger.info(f"TEST RESULT -> TEST F1: {f1}, Precision: {precision}, Recall: {recall} ")
        return {"test_log": tensorboard_logs, "test_f1": f1, "test_precision": precision, "test_recall": recall}

    def save_predictions_to_file(self, gold_entity_lst, pred_entity_lst, prefix="test"):
        save_file_path = os.path.join(self.args.save_path, "test_predictions.txt")
        print(f"INFO -> write predictions to {save_file_path}")
        with open(save_file_path, "w") as f:
            for gold_label_item, pred_label_item in zip(gold_entity_lst, pred_entity_lst):
                f.write(str(pred_label_item)+"\t" + str(gold_label_item) + '\n')
        return


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_path", type=str, help="bert config file")
    parser.add_argument("--train_batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--use_memory", action="store_true", help="load dataset to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of dataset")
    parser.add_argument("--data_dir", type=str, help="train data path")
    parser.add_argument("--save_path", type=str, help="train data path")
    parser.add_argument("--save_topk", default=1, type=int, help="save topk checkpoint")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, )
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--classifier", type=str, default="single")
    parser.add_argument("--no_lr_scheduler", action="store_true")
    parser.add_argument("--train_file_name", default="", type=str, help="use for truncated train sets.")
    parser.add_argument("--save_ner_prediction", action="store_true", help="only work for test.")
    parser.add_argument("--path_to_model_hparams_file", default="", type=str, help="use for evaluation")
    parser.add_argument("--checkpoint_path", default="", type=str, help="use for evaluation.")
    parser.add_argument("--loss_type", default="CRF", type=str, help="use for loss type chosen.")

    return parser


def main():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    model = CSCCorrectTask(args)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_path, "checkpoint", "{epoch}",),
        save_top_k=args.save_topk,
        save_last=False,
        monitor="val_f1",
        mode="max",
        verbose=True,
        period=-1,
    )

    logger = TensorBoardLogger(
        save_dir=args.save_path,
        name='log')

    # save args
    with open(os.path.join(args.save_path, "checkpoint", "args.json"), "w") as f:
        args_dict = args.__dict__
        del args_dict["tpu_cores"]
        json.dump(args_dict, f, indent=4)

    trainer = Trainer.from_argparse_args(args,
                                         checkpoint_callback=checkpoint_callback,
                                         logger=logger,
                                         deterministic=True)
    trainer.fit(model)

    # after training, use the model checkpoint which achieves the best f1 score on dev set to compute the f1 on test set.
    best_f1_on_dev, path_to_best_checkpoint = find_best_checkpoint_on_dev(args.save_path)
    model.result_logger.info("=&"*20)
    model.result_logger.info(f"Best F1 on DEV is {best_f1_on_dev}")
    model.result_logger.info(f"Best checkpoint on DEV set is {path_to_best_checkpoint}")
    checkpoint = torch.load(path_to_best_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    trainer.test(model)
    model.result_logger.info("=&"*20)


def find_best_checkpoint_on_dev(output_dir: str, log_file: str = "eval_result_log.txt"):
    with open(os.path.join(output_dir, log_file)) as f:
        log_lines = f.readlines()

    F1_PATTERN=re.compile(r"val_f1 reached \d+\.\d* \(best")
    # val_f1 reached 0.00000 (best 0.00000)
    CKPT_PATTERN=re.compile(r"saving model to \S+ as top")
    checkpoint_info_lines = []
    for log_line in log_lines:
        if "saving model to" in log_line:
            checkpoint_info_lines.append(log_line)
    # example of log line
    # Epoch 00000: val_f1 reached 0.00000 (best 0.00000), saving model to /data/xiaoya/outputs/glyce/0117/debug_5_12_2e-5_0.001_0.001_275_0.1_1_0.25/checkpoint/epoch=0.ckpt as top 20
    best_f1_on_dev = 0
    best_checkpoint_on_dev = 0
    for checkpoint_info_line in checkpoint_info_lines:
        current_f1 = float(re.findall(F1_PATTERN, checkpoint_info_line)[0].replace("val_f1 reached ", "").replace(" (best", ""))
        current_ckpt = re.findall(CKPT_PATTERN, checkpoint_info_line)[0].replace("saving model to ", "").replace(" as top", "")

        if current_f1 >= best_f1_on_dev:
            best_f1_on_dev = current_f1
            best_checkpoint_on_dev = current_ckpt

    return best_f1_on_dev, best_checkpoint_on_dev


def evaluate():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = CSCCorrectTask.load_from_checkpoint(checkpoint_path=args.checkpoint_path,
                                                map_location=None,
                                                batch_size=1,
                                                save_ner_prediction=args.save_ner_prediction)

    trainer = Trainer.from_argparse_args(args, deterministic=True)

    trainer.test(model)


def infer(sentences, mask_nums=3):
    '''
    todo: 输入部分添加冗余[MASK]字符
    ----------------------------
    inputs:
    - sentences: input sentence list
    - mask_nums: number of [MASK] that appends to each sentence

    output:
    - res: list of predict sentence
    '''

    # 加载参数
    parser = get_parser()
    args = parser.parse_args()

    # 加载tokenizer
    tokenizer = DetectorDataset(vocab_file=os.path.join(args.bert_path, "vocab.txt"),
                                config_path=os.path.join(args.bert_path, "config"),
                                max_length=args.max_length)

    # 恢复模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSCCorrectTask.load_from_checkpoint(checkpoint_path=args.checkpoint_path,
                                                map_location=None,)
    model.to(device)
    model.eval()

    # 加载数据
    # 按照TtT论文思路，且保证DetectorDataset一致性，先对sentence进行MASK扩充
    sentences = [sent + '[MASK]' * mask_nums for sent in sentences]
    input_ids, pinyin_ids = tokenizer.arrange_inputs(sentences)
    sequence_mask = (input_ids != 0).long()
    batch_size, seq_len = input_ids.shape
    pinyin_ids = pinyin_ids.view(batch_size, seq_len, 8)
    input_ids, pinyin_ids = input_ids.to(device), pinyin_ids.to(device)  # 送到GPU上
    with torch.no_grad():
        _, decode_result = model(input_ids=input_ids, pinyin_ids=pinyin_ids,)

    pred_sequence_labels = decode_result[1].to("cpu").numpy().tolist()
    sequence_mask = sequence_mask.numpy().tolist()
    refine_pred_labels = []
    for item_idx, pred_label_item in enumerate(pred_sequence_labels):
        sequence_mask_item = sequence_mask[item_idx]
        try:
            token_end_pos = sequence_mask_item.index(0) - 1  # before [PAD] always has an [SEP] token.
        except:
            token_end_pos = len(sequence_mask_item)
        refine_pred_labels.append(pred_label_item[1: token_end_pos])

    # 转换为中文序列
    output_sequences = tokenizer.tokenizer.decode_batch(refine_pred_labels)
    output_sequences = [sent.replace('[SEP]', '').strip() for sent in output_sequences]
    pprint(output_sequences)
    return


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    # for training
    main()

    # for evaluating
    # evaluate()

    # for inferring
    # infer(['张爱文很聪明，老师教他英文、地理什么得，他很快明白了。',
    #        '吃了早菜以后他去上课。',
    #        '下个星期，我跟我朋唷打算去法国玩儿。',
    #        '真麻烦你了。希望你们好好的跳无。',
    #        '所以我先去看医生，再去你的祝庆会。',
    #        '坐路差不多十分钟，我们到了。'])