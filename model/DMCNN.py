# -*- coding: utf-8 -*-
""" 
@author: RMSnow 
@file: DMCNN.py 
@time: 2020/6/2 16:03
@contact: xueyao_98@foxmail.com

# Based on the DMCNN model, Paper:
# Event Extraction via Dynamic Multi-Pooling Convolutional Neural Networks
# Yubo Chen, Liheng Xu, Kang Liu, Daojian Zeng and Jun Zhao
# ACL-2015

# 与DMCNN相比：
# 模型的输出为3d：(batch_size, max_sequence_length, event_type)，即为一个序列标注问题，
# 对于每个词语，标注其是否属于event trigger，其中 event_type 共有8类
"""

from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.initializers import Constant
from keras import backend as K
from keras.layers.core import Lambda
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, TimeDistributed, Flatten
from keras.layers import Dropout, Dense, Concatenate, Reshape, Multiply


class DMCNN:
    def __init__(self, max_sequence_length, embedding_matrix,
                 window_size=3, filters_num=200, pf_dim=5, invalid_flag=-1,
                 output=8, l2_param=0.01, lr_param=0.001):
        self.steps = max_sequence_length
        self.embedding_matrix = embedding_matrix
        self.window = window_size
        self.filters = filters_num
        self.dim = embedding_matrix.shape[1]
        self.pf_dim = pf_dim
        self.invalid_flag = invalid_flag
        self.output = output

        self.l2_param = l2_param

        self.model = self.build()
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_param, beta_1=0.8),
                           metrics=['accuracy'])

    def build(self):
        # [n, steps]
        cwf_input = Input(shape=(self.steps,), name='Word2Vec')
        # [n, steps, steps, 2]
        pf_input = Input(shape=(self.steps, self.steps, 2), name='PositionFeatures')
        # [n, steps, 6]
        lexical_level_input = Input(shape=(self.steps, 6), name='LexicalLevelFeatures')

        # [n, steps, 3, steps]
        event_words_mask_input = Input(shape=(self.steps, 3, self.steps), name='EventWordsMask')

        # ----------------------------------------------------------------------- #

        # [n, steps, dim]
        cwf_emb = Embedding(self.embedding_matrix.shape[0],
                            self.embedding_matrix.shape[1],
                            embeddings_initializer=Constant(self.embedding_matrix),
                            input_length=self.steps,
                            trainable=False, name='cwf_embedding')(cwf_input)
        # [n, steps, steps, dim]
        cwf_repeat = Lambda(lambda x: K.repeat_elements(x[:, None, :, :], rep=self.steps, axis=1),
                            name='max_sequence_repeat')(cwf_emb)

        # [n, steps, steps, pf_dim]
        pf_emb = TimeDistributed(Dense(self.pf_dim), name='pf_embedding')(pf_input)

        # [n, steps, steps, dim + pf_dim]
        sentence_level = Concatenate(name='SentenceLevel')([cwf_repeat, pf_emb])

        sentence_masks = []
        for i in range(3):
            # [n, steps, 3, steps] -> [n, steps, steps] -> [n, steps, steps, 1]
            sentence_mask = Lambda(
                lambda x: K.expand_dims(x[0][:, :, i, :], axis=-1) * x[1],
                name='mask{}'.format(i))([event_words_mask_input, sentence_level])

            # [n, steps, steps, dim + pf_dim] -> [n, steps, 1, steps, dim + pf_dim]
            sentence_mask_reshape = Lambda(lambda x: K.expand_dims(x, axis=2),
                                           name='sentence_mask_reshape{}'.format(i))(sentence_mask)

            sentence_masks.append(sentence_mask_reshape)

        # [n, steps, 3, steps, dim + pf_dim]
        sentence = Concatenate(name='SentenceLevelMask', axis=2)(sentence_masks)

        # [n, steps, 3, steps - window + 1, filters]
        conv = TimeDistributed(
            TimeDistributed(Conv1D(filters=self.filters, kernel_size=self.window, activation='relu')),
            name='conv')(sentence)

        # [n, steps, 3, 1, filters]
        conv_pool = TimeDistributed(
            TimeDistributed(MaxPooling1D(self.steps - self.window + 1)),
            name='max_pooling')(conv)
        # [n, steps, 3 * filters]
        conv_flatten = TimeDistributed(Flatten(), name='flatten')(conv_pool)
        cnn = TimeDistributed(Dropout(0.5), name='dropout')(conv_flatten)

        # ----------------------------------------------------------------------- #

        lexical_level_embeddings = []
        for i in range(6):
            # [n, steps, dim]
            lexical_level_emb = TimeDistributed(
                Lambda(lambda x: self.get_embedding(x[:, i])),
                name='LexicalEmbedding{}'.format(i))(lexical_level_input)
            lexical_level_embeddings.append(lexical_level_emb)
        # [n, steps, 6 * dim]
        lexical_level = Concatenate(name='LexicalLevel')(lexical_level_embeddings)

        # ----------------------------------------------------------------------- #

        # [n, steps, 3 * filters + 6 * dim]
        fusion = Concatenate(name='LexicalAndSentence')([cnn, lexical_level])

        # [n, steps, 32]
        dense = TimeDistributed(
            Dense(32, activation='relu', kernel_regularizer=l2(self.l2_param)),
            name='fc')(fusion)
        # [n, steps, output]
        output = TimeDistributed(
            Dense(self.output, activation='softmax', kernel_regularizer=l2(self.l2_param)),
            name='output')(dense)

        model = Model(inputs=[cwf_input, pf_input, lexical_level_input, event_words_mask_input], outputs=output)
        return model

    def get_embedding(self, x):
        # x: [n, ] -> [n, 1] -> [n, 1, dim] -> [n, dim]
        x = K.expand_dims(x, axis=-1)
        emb = Embedding(self.embedding_matrix.shape[0],
                        self.embedding_matrix.shape[1],
                        embeddings_initializer=Constant(self.embedding_matrix),
                        input_length=1,
                        trainable=False)(x)
        flat = Flatten()(emb)
        return flat


class CNN:
    def __init__(self, max_sequence_length, embedding_matrix,
                 window_size=3, filters_num=200, pf_dim=5, invalid_flag=-1,
                 output=8, l2_param=0.01, lr_param=0.001):
        self.steps = max_sequence_length
        self.embedding_matrix = embedding_matrix
        self.window = window_size
        self.filters = filters_num
        self.dim = embedding_matrix.shape[1]
        self.pf_dim = pf_dim
        self.invalid_flag = invalid_flag
        self.output = output

        self.l2_param = l2_param

        self.model = self.build()
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_param, beta_1=0.8),
                           metrics=['accuracy'])

    def build(self):
        # [n, steps]
        cwf_input = Input(shape=(self.steps,), name='Word2Vec')
        # [n, steps, steps, 2]
        pf_input = Input(shape=(self.steps, self.steps, 2), name='PositionFeatures')
        # [n, steps, 6]
        lexical_level_input = Input(shape=(self.steps, 6), name='LexicalLevelFeatures')

        # ----------------------------------------------------------------------- #

        # [n, steps, dim]
        cwf_emb = Embedding(self.embedding_matrix.shape[0],
                            self.embedding_matrix.shape[1],
                            embeddings_initializer=Constant(self.embedding_matrix),
                            input_length=self.steps,
                            trainable=False, name='cwf_embedding')(cwf_input)
        # [n, steps, steps, dim]
        cwf_repeat = Lambda(lambda x: K.repeat_elements(x[:, None, :, :], rep=self.steps, axis=1),
                            name='max_sequence_repeat')(cwf_emb)

        # [n, steps, steps, pf_dim]
        pf_emb = TimeDistributed(Dense(self.pf_dim), name='pf_embedding')(pf_input)

        # [n, steps, steps, dim + pf_dim]
        sentence_level = Concatenate(name='SentenceLevel')([cwf_repeat, pf_emb])

        # [n, steps, steps - window + 1, filters]
        conv = TimeDistributed(
            Conv1D(filters=self.filters, kernel_size=self.window, activation='relu'), name='conv')(sentence_level)

        # [n, steps, 1, filters]
        conv_pool = TimeDistributed(MaxPooling1D(self.steps - self.window + 1), name='max_pooling')(conv)
        # [n, steps, filters]
        conv_flatten = TimeDistributed(Flatten(), name='flatten')(conv_pool)
        cnn = TimeDistributed(Dropout(0.5), name='dropout')(conv_flatten)

        # ----------------------------------------------------------------------- #

        lexical_level_embeddings = []
        for i in range(6):
            # [n, steps, dim]
            lexical_level_emb = TimeDistributed(
                Lambda(lambda x: self.get_embedding(x[:, i])),
                name='LexicalEmbedding{}'.format(i))(lexical_level_input)
            lexical_level_embeddings.append(lexical_level_emb)

        # [n, steps, 6 * dim]
        lexical_level = Concatenate(name='LexicalLevel')(lexical_level_embeddings)

        # ----------------------------------------------------------------------- #

        # [n, steps, filters + 6 * dim]
        fusion = Concatenate(name='LexicalAndSentence')([cnn, lexical_level])

        # [n, steps, 32]
        dense = TimeDistributed(
            Dense(32, activation='relu', kernel_regularizer=l2(self.l2_param)),
            name='fc')(fusion)

        # [n, steps, output]
        output = TimeDistributed(
            Dense(self.output, activation='softmax', kernel_regularizer=l2(self.l2_param)),
            name='output')(dense)

        model = Model(inputs=[cwf_input, pf_input, lexical_level_input], outputs=output)

        return model

    def get_embedding(self, x):
        # x: [n, ] -> [n, 1] -> [n, 1, dim] -> [n, dim]
        x = K.expand_dims(x, axis=-1)
        emb = Embedding(self.embedding_matrix.shape[0],
                        self.embedding_matrix.shape[1],
                        embeddings_initializer=Constant(self.embedding_matrix),
                        input_length=1,
                        trainable=False)(x)
        flat = Flatten()(emb)
        return flat
