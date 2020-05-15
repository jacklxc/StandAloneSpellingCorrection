# StandAloneSpellingCorrection
Repository for EMNLP 2020 submission "Context-aware Stand-alone Neural Spelling Correction". This work is done by Xiangci Li as research scientist intern at IDL of SVAIL, Baidu USA supervised by Dr. Hairong Liu. If you have any question, please contact lixiangci8@gmail.com. The repository will be maintained [here](https://github.com/jacklxc/StandAloneSpellingCorrection).

The following are the explanation to each of the folders.

## Creating Dataset
We construct the misspelling dataset from [1-Billion-Word-Language-Model-Benchmark](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark). The natural misspelings are from `missp.dat.txt` ([link](https://www.dcs.bbk.ac.uk/~roger/corpora.html)) and `en.natural.txt` ([link](https://github.com/ybisk/charNMT-noise)). We treat sentences 1B-dataset as gold tokens, and randomly replace correct words with candidate misspellings. 

`dataset_train.py` and `dataset_train_random.py` creates datsets with natural misspellings and synthetic (random character) misspellings respectively, and store them in a input sentence file and label file. Other files convert these dataset files into corresponding format for different models. `train_replacement.json` saves the sampled "known misspelling vocabulary" described in the paper.

## Word-wise Spelling Correction
This is an early exploration of the spelling corrector, which is not covered in the paper. This directly originated from Dr. Hairong Liu's idea of using two Transformer encoders to encode characters and words, which is similar to Word+Char encoder in the paper. The difference is the model is only able to correct spellings at a given position, without being able to detect real-word misspellings.

The model is implemented with PaddlePaddle 1.6 DyGraph. The backbone code is from [PaddlePaddle's benchmark code](https://github.com/PaddlePaddle/benchmark/blob/master/dygraph/transformer/train.py). `configs/` includes configuration files for different settings, where `standard` means Word+Char model, `mix` means natural+synthetic misspellings and `char` means character encoder only.

To train the model, run `python -u model.py configs/standard_config.json`. Add `--test` for inference only. You can also run `tune_model.py` to evoke multiple jobs in SLURM system to train models with different configuations.

## LM Spelling Correction
This is the model introduced in the paper. Essentially it's a modification of [ERNIE 2.0 implementation](https://github.com/PaddlePaddle/ERNIE/tree/develop/ernie). Run `tune_xxx.py` to evoke multiple jobs in SLURM system and run corresponding `test_xxx.sh` for inference. The usage is the same as ERNIE 2.0.

## Baseline 1: ScRNN
ScRNN is the model proposed by [Sakaguchi et al. (2017)](https://github.com/keisks/robsut-wrod-reocginiton). This is the re-implementation using PaddlePaddle 1.6 DyGraph. Run `tune_robust_model.py` to evoke multiple jobs in SLURM system to tune learning rate and the hidden size of LSTM. You can run `python robust_model.py --test` to make inference once you have the model weights.

## Baseline 2: MUDE
[MUDE](https://github.com/zw-s-github/MUDE) is the previous SOTA model for stand-alone spelling correction. It's based on PyTorch. The version here is a minimal revision of the original code so that it can take our dataset as input.
