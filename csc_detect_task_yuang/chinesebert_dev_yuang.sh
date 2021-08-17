#!/usr/bin/env bash
# -*- coding: utf-8 -*-

TIME=0714
FILE_NAME=weibo_glyce_base
REPO_PATH=/Users/yuang/PA_tech/text_corrector/ChineseBert
BERT_PATH=/Users/yuang/PA_tech/text_corrector/ChineseBert/ChineseBERT-base
DATA_DIR=/Users/yuang/PA_tech/text_corrector/ChineseBert/csc_data
CHECKPOINT_PATH=/Users/yuang/PA_tech/text_corrector/ChineseBert/csc_data/output/0714/weibo_glyce_base_5_2_3e-5_0.002_0.02_256_0.2_1_0.25/checkpoint/epoch=4_v0.ckpt
HPARAMS_PATH=/Users/yuang/PA_tech/text_corrector/ChineseBert/csc_data/output/0714/weibo_glyce_base_5_2_3e-5_0.002_0.02_256_0.2_1_0.25/log/version_0/hparams.yaml

SAVE_TOPK=2
TRAIN_BATCH_SIZE=2
LR=3e-5
WEIGHT_DECAY=0.002
WARMUP_PROPORTION=0.02
MAX_LEN=256
MAX_EPOCH=5
DROPOUT=0.2
ACC_GRAD=1
VAL_CHECK_INTERVAL=0.25
CLASSIFIER=multi


OUTPUT_DIR=/Users/yuang/PA_tech/text_corrector/ChineseBert/csc_data/output/${TIME}/${FILE_NAME}_${MAX_EPOCH}_${TRAIN_BATCH_SIZE}_${LR}_${WEIGHT_DECAY}_${WARMUP_PROPORTION}_${MAX_LEN}_${DROPOUT}_${ACC_GRAD}_${VAL_CHECK_INTERVAL}
mkdir -p $OUTPUT_DIR
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

CUDA_VISIBLE_DEVICES="" python3 $REPO_PATH/detector_trainer.py \
--lr ${LR} \
--max_epochs ${MAX_EPOCH} \
--max_length ${MAX_LEN} \
--weight_decay ${WEIGHT_DECAY} \
--hidden_dropout_prob ${DROPOUT} \
--warmup_proportion ${WARMUP_PROPORTION} \
--train_batch_size ${TRAIN_BATCH_SIZE} \
--accumulate_grad_batches ${ACC_GRAD} \
--save_topk ${SAVE_TOPK} \
--bert_path ${BERT_PATH} \
--data_dir ${DATA_DIR} \
--save_path ${OUTPUT_DIR} \
--val_check_interval ${VAL_CHECK_INTERVAL} \
--classifier ${CLASSIFIER} \
--precision=32 \
--gpus="0" \
--save_ner_prediction \
--checkpoint_path ${CHECKPOINT_PATH} \
--path_to_model_hparams_file ${HPARAMS_PATH}