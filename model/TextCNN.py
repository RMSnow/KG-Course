# -*- coding: utf-8 -*-
""" 
@author: RMSnow 
@file: TextCNN.py 
@time: 2020/6/11 11:58
@contact: xueyao_98@foxmail.com

# Baseline: TextCNN
"""

from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.initializers import Constant
from keras import backend as K
from keras.layers.core import Lambda
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, TimeDistributed, Flatten
from keras.layers import Dropout, Dense, Concatenate, Reshape, Multiply


class TextCNN:
    def __init__(self, max_sequence_length, embedding_matrix,
                 window_size=3, filters_num=200,
                 output=8, l2_param=0.01, lr_param=0.001):
        self.steps = max_sequence_length
        self.embedding_matrix = embedding_matrix
        self.window = [window_size - 1, window_size, window_size + 1]
        self.filters = filters_num
        self.dim = embedding_matrix.shape[1]
        self.output = output

        self.l2_param = l2_param

        self.model = self.build()
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_param, beta_1=0.8),
                           metrics=['accuracy'])

    def build(self):
        # [n, steps]
        cwf_input = Input(shape=(self.steps,), name='Word2Vec')

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

        convs = []
        for window in self.window:
            # [n, steps, steps - window + 1, filters]
            conv = TimeDistributed(Conv1D(filters=self.filters, kernel_size=window, activation='relu'),
                                   name='conv_window{}'.format(window))(cwf_repeat)

            # [n, steps, 1, filters]
            conv_pool = TimeDistributed(MaxPooling1D(self.steps - window + 1),
                                        name='max_pooling_window{}'.format(window))(conv)
            # [n, steps, filters]
            conv_flatten = TimeDistributed(Flatten(), name='flatten_window{}'.format(window))(conv_pool)

            convs.append(conv_flatten)

        # [n, steps, filters * 3]
        convs = Concatenate(name='cnn', axis=-1)(convs)
        cnn = TimeDistributed(Dropout(0.5), name='dropout')(convs)

        # ----------------------------------------------------------------------- #

        # [n, steps, 32]
        dense = TimeDistributed(
            Dense(32, activation='relu', kernel_regularizer=l2(self.l2_param)),
            name='fc')(cnn)

        # [n, steps, output]
        output = TimeDistributed(
            Dense(self.output, activation='softmax', kernel_regularizer=l2(self.l2_param)),
            name='output')(dense)

        model = Model(inputs=[cwf_input], outputs=output)
        return model
