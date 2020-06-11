# -*- coding: utf-8 -*-
""" 
@author: RMSnow 
@file: train.py 
@time: 2020/6/10 23:42
@contact: xueyao_98@foxmail.com

# 记录模型训练过程、保存模型文件等
"""

import numpy as np
import os
import json

from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
import dataset_split

import matplotlib.pyplot as plt


def loss_plot(hist_logs):
    plt.plot()

    # keras 版本问题
    try:
        plt.plot(hist_logs['acc'], marker='*')
    except KeyError:
        plt.plot(hist_logs['accuracy'], marker='*')
    try:
        plt.plot(hist_logs['val_acc'], marker='*')
    except KeyError:
        plt.plot(hist_logs['val_accuracy'], marker='*')
    plt.plot(hist_logs['loss'], marker='*')
    plt.plot(hist_logs['val_loss'], marker='*')

    plt.title('model accuracy/loss')
    plt.ylabel('accuracy/loss')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'val_acc', 'train_loss', 'val_loss'], loc='upper left')
    plt.show()


def train(model, model_name, data, label,
          epochs=20, batch_size=128, early_stop=True,
          monitor='val_loss', monitor_mode='auto', to_predict=True, use_dev=False):
    train_data, dev_data, test_data = [[], [], []]
    train_label, dev_label, test_label = [[], [], []]

    for X in data:
        Xs = dataset_split.split_dataset(X, use_dev)
        train_data.append(Xs[0])
        test_data.append(Xs[-1])
        if use_dev:
            dev_data.append(Xs[1])

    ys = dataset_split.split_dataset(label, use_dev)
    train_label.append(ys[0])
    test_label.append(ys[-1])
    if use_dev:
        dev_label.append(ys[1])

    print(model.summary())
    print()

    if use_dev:
        model_file = './model/{}_useDev.hdf5'.format(model_name)
    else:
        model_file = './model/{}.hdf5'.format(model_name)

    checkpoint = ModelCheckpoint(model_file, monitor=monitor, verbose=0,
                                 save_best_only=True, mode=monitor_mode, save_weights_only=True)
    if early_stop:
        early_stop = EarlyStopping(
            monitor=monitor, patience=10, verbose=1, mode=monitor_mode)
    else:
        early_stop = EarlyStopping(
            monitor=monitor, patience=epochs, verbose=1, mode=monitor_mode)

    if use_dev:
        model_history = model.fit(train_data, train_label, epochs=epochs,
                                  batch_size=batch_size, validation_data=(dev_data, dev_label),
                                  shuffle=True, callbacks=[checkpoint, early_stop])
    else:
        model_history = model.fit(train_data, train_label, epochs=epochs,
                                  batch_size=batch_size, validation_data=(test_data, test_label),
                                  shuffle=True, callbacks=[checkpoint, early_stop])

    loss_plot(model_history.history)

    if to_predict:
        predict(model, model_file, test_data, test_label)


def predict_single_classification(y_pred, test_label, labels_names):
    # y_pred, test_label: (test_sz, 85, 8) -> (test_sz*85, 8)
    test_sz, steps, labels_sz = y_pred.shape
    y_pred = y_pred.reshape(test_sz * steps, labels_sz)
    test_label = test_label.reshape(test_sz * steps, labels_sz)

    arg = y_pred.argmax(axis=1)

    y_pred_label = np.zeros(y_pred.shape)
    for i, index in enumerate(arg):
        y_pred_label[i][index] = 1

    accuracy = accuracy_score(test_label, y_pred_label)

    print()
    print('TEST_sz:', len(test_label))
    print()
    print('Accuracy: {}'.format(accuracy))
    print()
    print(classification_report(test_label, y_pred_label,
                                labels=[i for i in range(len(labels_names))],
                                target_names=labels_names, digits=4))
    print()


def predict_single_identification(y_pred, test_label, labels_names):
    # y_pred, test_label: (test_sz, 85, 8) -> (test_sz*85, 2)
    test_sz, steps, labels_sz = y_pred.shape
    y_pred = y_pred.reshape(test_sz * steps, labels_sz)
    test_label = test_label.reshape(test_sz * steps, labels_sz)

    def transfer2identification(y):
        assert y.shape == (test_sz * steps, labels_sz)

        arg = y.argmax(axis=1)
        y_label = np.zeros(test_sz * steps)
        for i, label in enumerate(arg):
            if label != 0:
                y_label[i] = 1

        return y_label

    test_label = transfer2identification(test_label)
    y_pred_label = transfer2identification(y_pred)

    accuracy = accuracy_score(test_label, y_pred_label)

    print()
    print('TEST_sz:', len(test_label))
    print()
    print('Accuracy: {}'.format(accuracy))
    print()
    print(classification_report(test_label, y_pred_label,
                                labels=[i for i in range(len(labels_names))],
                                target_names=labels_names, digits=4))
    print()


def predict_multiple_identification(y_pred, test_label):
    # y_pred, test_label: (test_sz, 85, 8)
    test_sz, steps, labels_sz = y_pred.shape

    # -> (test_sz, 85)
    def transfer2identification(y):
        # (test_sz, 85)
        arg = y.argmax(axis=-1)

        y_label = np.zeros((test_sz, steps))
        for i, steps_labels in enumerate(arg):
            for j, label in enumerate(steps_labels):
                if label != 0:
                    y_label[i][j] = 1

        return y_label

    test_label = transfer2identification(test_label)
    y_pred_label = transfer2identification(y_pred)

    correct = 0
    for i in range(test_sz):
        if (test_label[i] == y_pred_label[i]).all():
            correct += 1

    print('TEST_sz:', len(test_label))
    print('Accuracy: {}'.format(correct / test_sz))


def predict_multiple_classification(y_pred, test_label, ):
    # y_pred, test_label: (test_sz, 85, 8)
    test_sz, steps, labels_sz = y_pred.shape

    test_label = test_label.argmax(axis=-1)
    y_pred_label = y_pred.argmax(axis=-1)

    correct = 0
    for i in range(test_sz):
        if (test_label[i] == y_pred_label[i]).all():
            correct += 1

    print('TEST_sz:', len(test_label))
    print('Accuracy: {}'.format(correct / test_sz))


def predict(model, model_file, test_data, test_label):
    model.load_weights(model_file)
    y_pred = model.predict(test_data)

    model_name = model_file.split('/')[-1].split('.')[0]
    np.save('./predict/y_pred_{}.npy'.format(model_name), y_pred)
