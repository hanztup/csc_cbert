#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file  : metrics/ner.py
@author: xiaoya li
@contact : xiaoya_li@shannonai.com
@date  : 2021/01/14 16:13
@version: 1.0
@desc  :
"""

import torch
from typing import Any, List
from pytorch_lightning.metrics.metric import TensorMetric


class MetricForCSC(TensorMetric):
    """
    compute span-level F1 scores for named entity recognition task.
    """
    def __init__(self, entity_labels: List[str] = None, reduce_group: Any = None, reduce_op: Any = None, save_prediction = False):
        super(MetricForCSC, self).__init__(name="metric_for_csc", reduce_group=reduce_group, reduce_op=reduce_op)
        self.num_labels = len(entity_labels)
        self.entity_labels = entity_labels
        self.tags2label = {label_idx: label_item for label_idx, label_item in enumerate(entity_labels)}
        self.save_prediction = save_prediction
        if save_prediction:
            self.pred_entity_lst = []
            self.gold_entity_lst = []

    def forward(self, pred_sequence_labels, gold_sequence_labels, sequence_mask=None):
        """
        Args:
            pred_sequence_labels: torch.LongTensor, shape of [batch_size, sequence_len]
            gold_sequence_labels: torch.LongTensor, shape of [batch_size, sequence_len]
            sequence_mask: Optional[torch.LongTensor], shape of [batch_size, sequence_len].
                        1 for non-[PAD] tokens; 0 for [PAD] tokens
        """
        true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0
        pred_sequence_labels = pred_sequence_labels.to("cpu").numpy().tolist()
        gold_sequence_labels = gold_sequence_labels.to("cpu").numpy().tolist()
        if sequence_mask is not None:
            sequence_mask = sequence_mask.to("cpu").numpy().tolist()
            # [1, 1, 1, 0, 0, 0]

        for item_idx, (pred_label_item, gold_label_item) in enumerate(zip(pred_sequence_labels, gold_sequence_labels)):
            if sequence_mask is not None:
                sequence_mask_item = sequence_mask[item_idx]
                try:
                    token_end_pos = sequence_mask_item.index(0) - 1 # before [PAD] always has an [SEP] token.
                except:
                    token_end_pos = len(sequence_mask_item)
            else:
                token_end_pos = len(gold_label_item)

            pred_label_item = [self.tags2label[tmp] for tmp in pred_label_item[1:token_end_pos]]
            gold_label_item = [self.tags2label[tmp] for tmp in gold_label_item[1:token_end_pos]]

            pred_entities = collect_labels(pred_label_item)
            gold_entities = collect_labels(gold_label_item)

            if self.save_prediction:
                self.pred_entity_lst.append(pred_entities)
                self.gold_entity_lst.append(gold_entities)

            tp, fp, fn = count_confusion_matrix(pred_label_item, gold_label_item)
            true_positive += tp
            false_positive += fp
            false_negative += fn

        batch_confusion_matrix = torch.LongTensor([true_positive, false_positive, false_negative])
        return batch_confusion_matrix

    def compute_f1_using_confusion_matrix(self, true_positive, false_positive, false_negative, prefix="dev"):
        """
        compute f1 scores.
        Description:
            f1: 2 * precision * recall / (precision + recall)
                - precision = true_positive / true_positive + false_positive
                - recall = true_positive / true_positive + false_negative
        Returns:
            precision, recall, f1
        """
        precision = true_positive / (true_positive + false_positive + 1e-13)
        recall = true_positive / (true_positive + false_negative + 1e-13)
        f1 = 2 * precision * recall / (precision + recall + 1e-13)

        if self.save_prediction and prefix == "test":
            entity_tuple = (self.gold_entity_lst, self.pred_entity_lst)
            return precision, recall, f1, entity_tuple

        return precision, recall, f1


class MetricForCSC_Corrector(TensorMetric):
    """
    compute span-level F1 scores for named entity recognition task.
    """
    def __init__(self, entity_labels, reduce_group=None, reduce_op=None, save_prediction=False):
        super(MetricForCSC_Corrector, self).__init__(name="metric_for_csc_corrector", reduce_group=reduce_group, reduce_op=reduce_op)
        self.num_labels = len(entity_labels)
        self.entity_labels = entity_labels
        self.tags2label = {label_idx: label_item for label_item, label_idx in entity_labels.items()}
        self.save_prediction = save_prediction
        if save_prediction:
            self.pred_entity_lst = []
            self.gold_entity_lst = []

    def forward(self, pred_sequence_labels, gold_sequence_labels, sequence_mask=None):
        """
        Args:
            pred_sequence_labels: torch.LongTensor, shape of [batch_size, sequence_len]
            gold_sequence_labels: torch.LongTensor, shape of [batch_size, sequence_len]
            sequence_mask: Optional[torch.LongTensor], shape of [batch_size, sequence_len].
                        1 for non-[PAD] tokens; 0 for [PAD] tokens
        """
        # true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0
        pp, rr, ff = 0., 0., 0.
        pred_sequence_labels = pred_sequence_labels.to("cpu").numpy().tolist()
        gold_sequence_labels = gold_sequence_labels.to("cpu").numpy().tolist()
        if sequence_mask is not None:
            sequence_mask = sequence_mask.to("cpu").numpy().tolist()

        for item_idx, (pred_label_item, gold_label_item) in enumerate(zip(pred_sequence_labels, gold_sequence_labels)):
            if sequence_mask is not None:
                sequence_mask_item = sequence_mask[item_idx]
                try:
                    token_end_pos = sequence_mask_item.index(0) - 1  # before [PAD] always has an [SEP] token.
                except:
                    token_end_pos = len(sequence_mask_item)
            else:
                token_end_pos = len(gold_label_item)

            pred_label_item = pred_label_item[1:token_end_pos]
            gold_label_item = gold_label_item[1:token_end_pos]

            pred_entities = ''.join([self.tags2label[tmp] for tmp in pred_label_item])
            gold_entities = ''.join([self.tags2label[tmp] for tmp in gold_label_item])

            if self.save_prediction:
                self.pred_entity_lst.append(pred_entities)
                self.gold_entity_lst.append(gold_entities)

            pi, ri, fi = count_confusion_matrix_for_corrector(pred_label_item, gold_label_item)
            pp += pi
            rr += ri
            ff += fi

        batch_confusion_matrix = torch.tensor([pp, rr, ff])
        return batch_confusion_matrix

    def compute_f1_using_confusion_matrix(self, true_positive, false_positive, false_negative, prefix="dev"):
        """
        compute f1 scores.
        Description:
            f1: 2 * precision * recall / (precision + recall)
                - precision = true_positive / true_positive + false_positive
                - recall = true_positive / true_positive + false_negative
        Returns:
            precision, recall, f1
        """
        precision = true_positive / (true_positive + false_positive + 1e-13)
        recall = true_positive / (true_positive + false_negative + 1e-13)
        f1 = 2 * precision * recall / (precision + recall + 1e-13)

        if self.save_prediction and prefix == "test":
            entity_tuple = (self.gold_entity_lst, self.pred_entity_lst)
            return precision, recall, f1, entity_tuple

        return precision, recall, f1


def count_confusion_matrix_for_corrector(pred_labels, gold_labels):
    acc = 0.
    for i, (pred, gold) in enumerate(zip(pred_labels, gold_labels)):
        if pred == gold:
            acc += 1
    pi = acc / (len(pred_labels) + 1e-13)
    ri = acc / (len(gold_labels) + 1e-13)
    fi = 2 * pi * ri / (pi + ri + 1e-13)

    return pi, ri, fi


def count_confusion_matrix(pred_labels, gold_labels):
    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0

    for i, (pred, gold) in enumerate(zip(pred_labels, gold_labels)):
        if gold == "1":
            if pred == gold:
                true_positive += 1
            else:
                false_negative += 1
        elif gold == "0":
            if pred == gold:
                true_negative += 1
            else:
                false_positive += 1
        else:
            raise ValueError("Wrong label value.")

    return true_positive, false_positive, false_negative


def collect_labels(label_sequence):
    positive_locations = []
    for idx, label in enumerate(label_sequence):
        if label != '1':
            continue
        positive_locations.append(idx)

    return positive_locations
