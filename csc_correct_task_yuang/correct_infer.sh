#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# author: Hanzhang Yang

REPO_PATH=/Users/yuang/PA_tech/text_corrector/ChineseBert/csc_correct_task_yuang
BERT_PATH=/Users/yuang/PA_tech/text_corrector/ChineseBert/ChineseBERT-base
CHECKPOINT_PATH=/Users/yuang/PA_tech/text_corrector/ChineseBert/csc_correct_task_yuang/results/0810/corrector_glyce_base_1000_2_3e-5_0.002_0.02_256_0.1_1_0.1/checkpoint/epoch=29_v1.ckpt

MAX_LEN=256
CLASSIFIER=multi

CUDA_VISIBLE_DEVICES="" python $REPO_PATH/corrector_trainer.py \
--max_length ${MAX_LEN} \
--bert_path ${BERT_PATH} \
--classifier ${CLASSIFIER} \
--checkpoint_path ${CHECKPOINT_PATH}