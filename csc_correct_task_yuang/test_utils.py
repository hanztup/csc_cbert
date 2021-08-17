# coding:utf-8

'''
date: 2021/08/16
content: 利用FASPell的逻辑，编写csc的测试代码
'''

import json


def test_unit(res, test_path, out_name, strict=True):
    '''
    测试单元 用验证CSC的效果
    --------------------
    inputs:
    - res:
    - test_path:
    - out_name:

    outputs:
    -
    '''

    out = open(f'{out_name}.txt', 'w', encoding='utf-8')  # 存入对每一个待CSC句子的性能检查结果

    corrected_char = 0         # 统计在CSC句子中 所有纠正过的char数目
    wrong_char = 0             # 统计在CSC句子中 所有存在错误的char数目
    corrected_sent = 0
    wrong_sent = 0
    true_corrected_char = 0    # 正确纠正的char数目
    true_corrected_sent = 0    # 正确纠正的sent数目
    true_detected_char = 0
    true_detected_sent = 0     # 句子级别 检错正确的句子数目
    accurate_detected_sent = 0
    accurate_corrected_sent = 0
    all_sent = 0               # 所有句子的数目

    for idx, line in enumerate(open(test_path, 'r', encoding='utf-8')):
        all_sent += 1

        num, wrong, correct = line.strip().split('\t')
        predict = res[idx]

        wrong_num = 0                       # 记录一个句子中 错误纠正的char数目
        corrected_num = 0                   # 记录一个句子中 纠正的char数目
        original_wrong_num = 0              # 记录一个句子中 存在错误的char数目
        true_detected_char_in_sentence = 0  # 记录一个句子中 正确检查的char数目

        # 通过循环的方式 确定各个指标的具体值
        for c, w, p in zip(correct, wrong, predict):
            if c != p:          # 正确的char不等于模型输出的char
                wrong_num += 1  # 错误纠正的char数目+1
            if w != p:              # 原句的char不等于模型输出的char
                corrected_num += 1  # 纠正过的char次数+1
                if c == p:
                    true_corrected_char += 1 # 正确纠正char数目+1
                if w != c:
                    true_detected_char += 1  # 正确检查char+1 正确检查句子中的char+1
                    true_detected_char_in_sentence += 1
            if c != w:              # 正确的char不等于原始的char
                original_wrong_num += 1  # 原句中char错误的数目+1

        # 写入对该句子的性能检查结果
        # out.write('\t'.join([str(original_wrong_num), wrong, correct, predict, str(wrong_num)]) + '\n')
        corrected_char += corrected_num            # 统计所有纠正过的char数目
        wrong_char += original_wrong_num           # 统计所有存在错误的char数目
        if original_wrong_num != 0:                # 检查该句子中是否有错误的char
            wrong_sent += 1                        # 错误句子数目+1 统计所有存在错误的句子数目
        if corrected_num != 0 and wrong_num == 0:  # 二者均满足的时候 句子中纠正过且纠正全部正确
            true_corrected_sent += 1               # 正确纠正的句子数目+1
        if corrected_num != 0:                     # 只满足纠正了句子
            corrected_sent += 1                    # 纠正的句子数目+1
        if strict:                                 # 是否严格检查句子级别检错结果 严格：所有检查出的位置是存在错误的位置
            true_detected_flag = (
                        true_detected_char_in_sentence == original_wrong_num and original_wrong_num != 0 and corrected_num == true_detected_char_in_sentence)
        else:                                      # 不严格检查： 句子存在错误 且有检查痕迹
            true_detected_flag = (corrected_num != 0 and original_wrong_num != 0)

        if true_detected_flag:                     # 检错结果  句子级别检错 只要句子中存在错误 且判断模型有改动 则认为是正确的检查出了错误的句子
            true_detected_sent += 1                # 字词级别检错 句子中存在错误 且所有错误的地方均被正确的检查出来
        if correct == predict:                     # 满足模型输出句子 等于 正确句子
            accurate_corrected_sent += 1           # 输出句子等于correct的数目+1（就是正确的句子不做改变+错误的句子纠正正确 Tp + Tn）
        if correct == predict or true_detected_flag:  # 满足模型输出句子 等于 正确句子 或者检错结果正确
            accurate_detected_sent += 1               # 正确检查出的句子数目+1

    print("corretion:")
    print(f'char_p={true_corrected_char}/{corrected_char}')  # 正确纠正的char数目/所有纠正的char数目
    print(f'char_r={true_corrected_char}/{wrong_char}')      # 正确纠正的char数目/所有存在错误的char数目
    print(f'sent_p={true_corrected_sent}/{corrected_sent}')  # 正确纠正的句子数目/所有纠正的句子数目
    print(f'sent_r={true_corrected_sent}/{wrong_sent}')      # 正确纠正的句子数目/所有存在错误的句子数目
    print(f'sent_a={accurate_corrected_sent}/{all_sent}')    # （Tp +Tn）/ all_sents
    print("detection:")
    print(f'char_p={true_detected_char}/{corrected_char}')   # 正确检查出的char数目 / 所有检查出的char数目
    print(f'char_r={true_detected_char}/{wrong_char}')       # 正确检查出的char数目 / 所有存在错误的char数目
    print(f'sent_p={true_detected_sent}/{corrected_sent}')   # 正确检查出错误的句子数目 / 所有纠正的句子数目
    print(f'sent_r={true_detected_sent}/{wrong_sent}')       # 正确检查出错误的句子数目 / 所有存在错误的句子数目
    print(f'sent_a={accurate_detected_sent}/{all_sent}')     # （Tp + Tn )/ all_sentences

    # 将模型CSC后的结果又写入到out_name.json文件中 方便后续试验观察
    # w = open(f'{out_name}.json', 'w', encoding='utf-8')
    # w.write(json.dumps(res, ensure_ascii=False, indent=4, sort_keys=False))
    # w.close()


def get_model_output(path):
    outputs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            predict, gold = line.split('\t')
            outputs.append(predict)

    return outputs


if __name__ == '__main__':

    model_output_path = '/Users/yuang/PA_tech/text_corrector/ChineseBert/csc_correct_task_yuang/results/0810/corrector_glyce_base_1000_2_3e-5_0.002_0.02_256_0.1_1_0.1/test_predictions.txt'
    origin_test_path = '/Users/yuang/PA_tech/text_corrector/ChineseBert/csc_correct_task_yuang/data/sighan/origin/test_ofsighan15.txt'

    model_outputs = get_model_output(model_output_path)
    test_unit(model_outputs, origin_test_path, 'what_ever')
