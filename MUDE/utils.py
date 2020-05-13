#!/usr/bin/env python
#encoding: utf-8
import string
import numpy as np
import random

def get_batch(X_source, mask_source, Y_source, batch_size, i):
    seq_len = min(batch_size, len(X_source) - 1 - i)
    X = X_source[i:i+seq_len]
    mask = mask_source[i:i+seq_len]
    Y = Y_source[i:i+seq_len]
    return X, mask, Y


def hasnum(w):
    for c_i in w:
        if c_i.isdigit():
            return True
    return False


def vec_char(w, alph, max_char_num):
    pad_idx = len(alph) + 4
    bin_all = [pad_idx]*max_char_num
    bin_all[0] = len(alph)
    mask_all = [0]*max_char_num
    mask_all[0]=1
    if w == '<eos>':
        bin_all[0]=len(alph)+1
    elif w == '<unk>':
        bin_all[0]=len(alph)+2
    elif hasnum(w):
        bin_all[0]=len(alph)+3
    else:
        for i in range(min(len(w),max_char_num-1)):
            try:
                bin_all[i+1] = alph.index(w[i])
            except ValueError:
                bin_all[i+1] = len(alph)+2
            mask_all[i+1] = 1
    return np.array(bin_all), np.array(mask_all)