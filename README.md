# Chinese Spell/Grammar Detection(Correction) based on ChineseBERT

This repository is built on the [repository][1] for [ChineseBERT]() at ACL2021.

**[ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information](https://arxiv.org/pdf/2106.16038.pdf)**
*Zijun Sun, Xiaoya Li, Xiaofei Sun, Yuxian Meng, Xiang Ao, Qing He, Fei Wu and Jiwei Li*

[1]: https://github.com/ShannonAI/ChineseBert

## Guide  

| Task | Description |
|  ----  | ----  |
| [csc_detect_task][2] | using chinesebert and softmax to predict char label (binary token classification) |
| [csc_correct_task][3] | using chinesebert and [dynamic crf][4] to predict char label (tags equals to vocabulary) |
| [csc_detect_promotion_task][5](on going) | using chinesebert and chainer crf to to predict char label (aims to tackle the error type of missing/redundant) |

[2]: https://github.com/hanztup/csc_cbert/tree/main/csc_detect_task_yuang
[3]: https://github.com/hanztup/csc_cbert/tree/main/csc_correct_task_yuang
[4]: https://arxiv.org/abs/2106.01609
[5]: https://github.com/hanztup/csc_cbert/tree/main/csc_detect_task_promotion_yuang