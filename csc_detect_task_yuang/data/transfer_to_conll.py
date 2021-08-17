# coding: utf-8

'''
date: 2021/7/19
content: 将CSC dataset转换为CoNLL格式
'''

import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default='./test_ofsighan15.txt', help="file to be transferred")
    parser.add_argument("--output_path", type=str, default='./test_sighan15.all.txt', help="file to be output ")
    args = parser.parse_args()
    return args


def load_and_store(args):
    '''load the SIGHAN format data, and output CoNLL format data'''

    data_lines = []
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            wrong_num, wrong, correct = line.split('\t')
            wrong = wrong.strip().replace('　', '')
            correct = correct.strip().replace('　', '')
            assert len(wrong) == len(correct), "Have different length"
            data_lines.append((wrong, correct))

    with open(args.output_path, 'w', encoding='utf-8') as f:
        for wrong, correct in data_lines:
            for w_c, c_c in zip(wrong, correct):
                if w_c == c_c:
                    f.write(w_c + ' ' + "0" + '\n')
                else:
                    f.write(w_c + ' ' + "1" + '\n')
            f.write('\n')
    return


def main():
    args = get_parser()

    load_and_store(args)


if __name__ == '__main__':
    main()