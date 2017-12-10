# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 15:12:58 2017

@author: jxtan
"""
from __future__ import print_function
#import os
import re
import string
#import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from gensim.models.doc2vec import LabeledSentence
from gensim import utils
from gensim.models import Doc2Vec

input_ID = "0"
input_text = "ewrsfdg e#$5w sd2!fgxc s#$dfg wert sdfg"
input_size = 1

'''data cleaning'''
def textClean(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]    
    text = " ".join(text)
    return(text)

def cleanup(text):
    text = textClean(text)
    text = text.translate(str.maketrans("","", string.punctuation))
    return text

inputText = cleanup(input_text)

sentences = inputText.split()
a = []
for i in sentences:
    a.append(LabeledSentence(utils.to_unicode(i).split(), ['Text' + '_%s' % str('0')]))

sentences = a
#Doc2vec model for new data
INPUT_DIMENSION=300
inputText_arrays = np.zeros((input_size, INPUT_DIMENSION))
text_model = Doc2Vec(min_count=1, window=5, size=INPUT_DIMENSION, sample=1e-4, negative=5, workers=4, iter=5,seed=1)
text_model.build_vocab(sentences)
text_model.train(sentences, total_examples=text_model.corpus_count, epochs=text_model.iter)

#text arrays initialize
for i in range(input_size):
    inputText_arrays[i] = text_model.docvecs['Text_'+str(i)]
''' Merge Input features'''
input_set=inputText_arrays

#predict the model
from sklearn.externals import joblib
clf = joblib.load("train_model.m")
result=clf.predict_proba(input_set)

mystring = str(input_ID) + " | "
for digit in result[0]:
    mystring += str(digit) + ' '
print(mystring)

