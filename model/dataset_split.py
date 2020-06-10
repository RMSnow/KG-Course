# -*- coding: utf-8 -*-
""" 
@author: RMSnow 
@file: dataset_split.py 
@time: 2020/6/10 23:37
@contact: xueyao_98@foxmail.com

# 数据集划分
1. 根据句子中出现的event triggers词的数量来分层抽样
2. Train:Test = 4:1
"""

import json
from sklearn.model_selection import train_test_split
import numpy as np

RATIO = 0.2
SEED = 0

with open('../data/preprocess/dataset.json', 'r') as f:
    dataset = json.load(f)

triggers_num = [len(p['triggers']) for p in dataset]
train_index, test_index = train_test_split(
    np.arange(len(dataset)), test_size=RATIO, stratify=triggers_num, random_state=SEED)


def split_dataset(arr, use_dev=False):
    assert len(arr) == len(dataset)

    train_arr = arr[train_index]
    test_arr = arr[test_index]

    print(train_arr.shape, test_arr.shape)
    return train_arr, test_arr
