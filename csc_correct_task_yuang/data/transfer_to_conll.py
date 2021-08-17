# coding: utf-8

'''
date: 2021/08/02
content: 将CSC dataset转换为CoNLL格式，借鉴TtT思路，兼容句长不相等的情况
'''

import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default='./sighan/origin/test_ofsighan.txt', help="file to be transferred")
    parser.add_argument("--output_path", type=str, default='./sighan/transformed/tff.all.txt', help="file to be output")
    parser.add_argument("--type", type=str, default='correct', help="model type, one of detect/correct")

    args = parser.parse_args()
    return args


def load_origin(input_path, model_type):
    data_lines = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            wrong_num, wrong, correct = line.split('\t')
            wrong = wrong.strip().replace('　', '')
            correct = correct.strip().replace('　', '')

            if len(wrong) != len(correct) and model_type == 'detect':
                continue

            data_lines.append((wrong, correct))
    return data_lines


def trans_and_store_detect(output_path, data_lines):
    with open(output_path, 'w', encoding='utf-8') as f:
        for wrong, correct in data_lines:
            for w_c, c_c in zip(wrong, correct):
                if w_c == c_c:
                    f.write(w_c + ' ' + "0" + '\n')
                else:
                    f.write(w_c + ' ' + "1" + '\n')
            f.write('\n')
    return


def trans_and_store_correct(output_path, data_lines):
    all_text, all_tag = [], []
    for wrong, correct in data_lines:

        text_list = [w for w in wrong.strip()]
        tag_name_list = [w for w in correct.strip()]

        if len(tag_name_list) > len(text_list):
            text_list += ['[MASK]'] * (len(tag_name_list) - len(text_list))  # 在输入句子后追加MASK

            # text_list += ['<-MASK->'] * (len(tag_name_list) - len(text_list))  # 在输入句子后追加MASK
            # text_list += ['<-SEP->']
            # tag_name_list += ['<-SEP->']
        elif len(tag_name_list) < len(text_list):
            tag_name_list += ['[PAD]'] * (len(text_list) - len(tag_name_list))

            # tag_name_list += ['<-SEP->'] + ['<-PAD->'] * (len(text_list) - len(tag_name_list))
            # text_list += ['<-SEP->']
        else:
            pass

            # tag_name_list += ['<-SEP->']
            # text_list += ['<-SEP->']
        assert len(text_list) == len(tag_name_list)
        # tag_list = list()
        # for token in tag_name_list:
        #     tag_list.append(self.label_dict.token2idx(token))
        # return text_list, tag_list

        all_text.append(text_list)
        all_tag.append(tag_name_list)

    with open(output_path, 'w', encoding='utf-8') as f:
        for wrong, correct in zip(all_text, all_tag):
            for w_c, c_c in zip(wrong, correct):
                f.write(w_c + ' ' + c_c + '\n')
            f.write('\n')

    return





def load_and_store(args):
    '''load the SIGHAN format data, and output CoNLL format data'''

    # 加载原始格式数据
    data_lines = load_origin(args.input_path, model_type=args.type)

    # 转换并保存为detector/corrector所需的数据
    if args.type == 'detect':
        trans_and_store_detect(args.output_path, data_lines)
    elif args.type == 'correct':
        trans_and_store_correct(args.output_path, data_lines)
    else:
        raise ValueError("Wrong model type.")

    return


def main():
    args = get_parser()
    load_and_store(args)


if __name__ == '__main__':
    main()