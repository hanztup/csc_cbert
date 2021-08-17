#!/usr/bin/env bash
# -*- coding: utf-8 -*-

REPO_PATH=/Users/yuang/PA_tech/text_corrector/ChineseBert
BERT_PATH=/Users/yuang/PA_tech/text_corrector/ChineseBert/ChineseBERT-base
CHECKPOINT_PATH=/Users/yuang/PA_tech/text_corrector/ChineseBert/csc_data/output/0714/weibo_glyce_base_5_2_3e-5_0.002_0.02_256_0.2_1_0.25/checkpoint/epoch=4_v0.ckpt

MAX_LEN=256
CLASSIFIER=multi

CUDA_VISIBLE_DEVICES="" python $REPO_PATH/detector_trainer.py \
--max_length ${MAX_LEN} \
--bert_path ${BERT_PATH} \
--classifier ${CLASSIFIER} \
--checkpoint_path ${CHECKPOINT_PATH}