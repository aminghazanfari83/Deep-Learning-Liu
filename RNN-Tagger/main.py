#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:32:58 2019

@author: aminghazanfari
"""
import utils
import keras
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, InputLayer, Bidirectional,TimeDistributed
from keras.layers import Embedding
from keras.layers import LSTM
from keras.optimizers import Adam


training_data = list(utils.read_training_data())
print('Number of sentences in the training data: {}'.format(len(training_data)))

development_data = list(utils.read_development_data())
print('Number of sentences in the development data: {}'.format(len(development_data)))

test_data = list(utils.read_test_data())
print('Number of sentences in the test data: {}'.format(len(test_data)))

w2i = dict()
t2i = dict()
w2i[b'UNW123'] = 1
w2i[b'PAD123'] = 0
t2i[b'PAD123'] = 0

for tagged_sentence in training_data:
    for word, tag in tagged_sentence:
        if word not in w2i:
            w2i[word] = len(w2i)    # assign next available index
        if tag not in t2i:
            t2i[tag] = len(t2i)    # assign next available index
print('Number of unique words in the training data: {}'.format(len(w2i)))
print('Number of unique tags in the training data: {}'.format(len(t2i)))

class Tagger(object):

    def __init__(self):
        self.model = Sequential() 
        self.n = 1
        
    def train(self,training_data):
        self.n= len(max(training_data, key=len))
        self.N = len(training_data)
        input_size = np.zeros((self.N,self.n),dtype='int32')
        output_size = np.zeros((self.N,self.n),dtype='int32')

        iter = 0
        for tagged_sentence in training_data:
            train_sentences_num, train_tags_num = [], []
            for word, tag in tagged_sentence:
                try:
                    train_sentences_num.append(w2i[word])
                except KeyError:
                    train_sentences_num.append(w2i[b'UNW123'])
        
                train_tags_num.append(t2i[tag])
            train_sentences_num_paded = sequence.pad_sequences([train_sentences_num], maxlen = self.n) 
            train_tags_num_padded = sequence.pad_sequences([train_tags_num], maxlen = self.n) 
            input_size[iter,:] = train_sentences_num_paded
            output_size[iter,:] = train_tags_num_padded
    
            iter = iter + 1
        output_size = keras.utils.to_categorical(output_size,num_classes=len(t2i)) 

        self.model.add(InputLayer(input_shape=(self.n, )))
        self.model.add(Embedding(len(w2i), 100,mask_zero= True))
        self.model.add(Bidirectional(LSTM(50, return_sequences=True)))
        self.model.add(Dense(len(t2i),activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())
        self. model.fit(input_size, output_size, batch_size=32, epochs=4)
        pass

    def evaluate(self,gold_data):
        input_size_eval = [] 
        output_size_eval = []

        for tagged_sentence_eval in gold_data:
            train_sentences_num_eval, train_tags_num_eval = [], []
            for word, tag in tagged_sentence_eval:
                try:
                    train_sentences_num_eval.append(w2i[word])
                except KeyError:
                    train_sentences_num_eval.append(w2i[b'UNW123'])
        
                train_tags_num_eval.append(t2i[tag])
            input_size_eval.append(train_sentences_num_eval)
            output_size_eval.append(train_tags_num_eval)
            #print(len(output_size_eval))
            #print(len(output_size_eval[0]))
            #print(len(input_size_eval))
            #print(len(input_size_eval[0]))
        
        
        train_sentences_num_paded_eval = sequence.pad_sequences(input_size_eval, maxlen = self.n) 
        train_tags_num_padded_eval = sequence.pad_sequences(output_size_eval, maxlen = self.n) 
        output_size_eval = keras.utils.to_categorical(train_tags_num_padded_eval,num_classes=len(t2i)) 
        acc = self.model.evaluate(train_sentences_num_paded_eval, output_size_eval)
        return  acc


tagger = Tagger()
tagger.train(training_data)
print('Start working on development data')
acc_developmentdata = tagger.evaluate(development_data)
print('Accuracy on development data with padding {:.2%}'.format(acc_developmentdata[1]))
print('Start working on test data')
acc_test_data = tagger.evaluate(test_data)
print('Accuracy on test data with padding {:.2%}'.format(acc_test_data[1]))