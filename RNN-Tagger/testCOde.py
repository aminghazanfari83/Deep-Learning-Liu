#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:59:11 2019

@author: aminghazanfari
"""

# After running, if you get the "using TensorFlow backend" message, please run again.
import utils
import keras
import numpy as np
from keras.preprocessing import sequence

training_data = list(utils.read_training_data())
print('Number of sentences in the training data: {}'.format(len(training_data)))

development_data = list(utils.read_development_data())
print('Number of sentences in the development data: {}'.format(len(development_data)))

test_data = list(utils.read_test_data())
print('Number of sentences in the test data: {}'.format(len(test_data)))



# Construct a simple index for words

w2i = dict()
tag2i = dict()
for tagged_sentence in training_data:
    for word, tag in tagged_sentence:
        #print('The content of tag {}'.format(tag))
        #print('The content of word {}'.format(word))
        if word not in w2i:
            w2i[word] = len(w2i) + 2    # assign next available index
        if tag not in tag2i:
            tag2i[tag] = len(tag2i) + 1

w2i = {w: i + 2 for i, w in enumerate(list(w2i))}
w2i['-PAD-'] = 0  # The special value used for padding
w2i['-OOV-'] = 1  # The special value used for OOVs

tag2i = {t: i + 1 for i, t in enumerate(list(tag2i))}
tag2i['-PAD-'] = 0  # The special value used to padding

#print('InTensor {}'.format(InTensor[0,:]))
print('Number of unique words in the training data: {}'.format(len(w2i)))
print('Index of the word "hates": {}'.format(w2i[b'hates']))
print('Number of unique tags in the training data:: {}'.format(len(tag2i)))

#from keras import backend as KVal
#def NoPadding_accuracy(ignoreIndx=0):
#    def NoPadding_acc(y_true, y_pred):
#        y_true_class = KVal.argmax(y_true, axis=-1)
#        y_pred_class = KVal.argmax(y_pred, axis=-1)
# 
#        ignore_mask = KVal.cast(KVal.not_equal(y_pred_class, ignoreIndx), 'int32')
#        matches = KVal.cast(KVal.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
#        accVal = KVal.sum(matches) / KVal.maximum(KVal.sum(ignore_mask), 1)
#        return accVal
#    return NoPadding_acc

#class Tagger(object):

#    def __init__(self):
        #self.model = None
#        self.model = Sequential()

#    def train(self, training_data):
        # Pre-process the training data
N = len(training_data)
n = max(len(tagged_sentence) for tagged_sentence in training_data)
InTensor = np.zeros((N,self.n),dtype='int32')
OutTensor = np.zeros((N,self.n),dtype='int32')
m = 0
for tagged_sentence in training_data:
    TempSeq = []
    TempTag  = []
    for word, tag in tagged_sentence:
        try:
            TempSeq.append(w2i[word])
        except KeyError:
            TempSeq.append(1)
                
        TempTag.append(tag2i[tag])
                
            #print('TempSeq {}'.format(TempSeq))
        PadSeq = sequence.pad_sequences([TempSeq], maxlen = self.n) 
            #print('PadSeq {}'.format(PadSeq))
        PadTag = sequence.pad_sequences([TempTag], maxlen = self.n) 
        InTensor[m,:] = PadSeq
        OutTensor[m,:] = PadTag
        m = m + 1
OutTensor = keras.utils.to_categorical(OutTensor)
            
        # Construct the network, add layers, compile, and fit
        self.model.add(InputLayer(input_shape=(self.n, )))
        self.model.add(Embedding(len(w2i), 100))
        self.model.add(Bidirectional(LSTM(100, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(len(tag2i))))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy',optimizer=Adam(0.001), metrics=['accuracy', NoPadding_accuracy(0)])
        self.model.summary()
        print('Start working on training data \n')
        print('Accuracy on training data')
        self.model.fit(InTensor, OutTensor, batch_size=100, epochs=10, validation_split=0.0)
        pass

    def evaluate(self, gold_data):
        # Compute the accuracy of the tagger relative to the gold data
        N_gold_data = len(gold_data)
        n_gold_data = max(len(tagged_sentence) for tagged_sentence in gold_data)
        InTensor_gold_data  = np.zeros((N_gold_data,self.n),dtype='int32')
        OutTensor_gold_data  = np.zeros((N_gold_data,self.n),dtype='int32')
        m = 0
        for tagged_sentence in gold_data:
            TempSeq_gold_data = []
            TempTag_gold_data  = []
            for word, tag in tagged_sentence:
                try:
                    TempSeq_gold_data.append(w2i[word])
                except KeyError:
                    TempSeq_gold_data.append(1)
                    
                TempTag_gold_data.append(tag2i[tag])
            PadSeq_gold_data = sequence.pad_sequences([TempSeq_gold_data], maxlen = self.n) 
            PadTag_gold_data = sequence.pad_sequences([TempTag_gold_data], maxlen = self.n) 
            InTensor_gold_data[m,:] = PadSeq_gold_data
            OutTensor_gold_data[m,:] = PadTag_gold_data
            m = m + 1
        OutTensor_gold_data = keras.utils.to_categorical(OutTensor_gold_data)
        acc = self.model.evaluate(InTensor_gold_data, OutTensor_gold_data)
        return acc
    
    
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam

tagger = Tagger()
tagger.train(training_data)
print('Start working on development data')
acc_developmentdata = tagger.evaluate(development_data)
print('Accuracy on development data with padding {:.2%}'.format(acc_developmentdata[1]))
print('Accuracy on development data without padding {:.2%}'.format(acc_developmentdata[2]))
print('Start working on test data')
acc_test_data = tagger.evaluate(test_data)
print('Accuracy on test data with padding {:.2%}'.format(acc_test_data[1]))
print('Accuracy on test data without padding {:.2%}'.format(acc_test_data[2]))

