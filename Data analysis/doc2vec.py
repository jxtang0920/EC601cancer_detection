# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 15:12:58 2017

@author: jxtan
"""
from __future__ import print_function
import os
import re
import string
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from gensim.models.doc2vec import LabeledSentence
from gensim import utils

train_variant = pd.read_csv("training_variants")
train_text = pd.read_csv("training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
train = pd.merge(train_variant, train_text, how='left', on='ID')
train_y = train['Class'].values
train_x = train.drop('Class', axis=1)
train_size=len(train_x)

test_variant  = pd.read_csv("test_variants")
test_text  = pd.read_csv("test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test_x = pd.merge(test_variant, test_text, how='left', on='ID')
test_size=len(test_x)

'''
print("Number of training variants:%d" % (train_size))
print("Number of training variants:%d" % (test_size))
'''

test_index = test_x['ID'].values
all_data = np.concatenate((train_x, test_x), axis=0)
all_data = pd.DataFrame(all_data)
all_data.columns = ["ID", "Gene", "Variation", "Text"]

'''data cleaning'''
class datacleaning():
    def constructLabeledSentences(data):
        sentences=[]
        for index, row in data.iteritems():
            sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
        return sentences

    def textClean(text):
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = text.lower().split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]    
        text = " ".join(text)
        return(text)

    def cleanup(text):
        text = datacleaning.textClean(text)
        text= text.translate(str.maketrans("","", string.punctuation))
        return text

allText = all_data['Text'].apply(datacleaning.cleanup)
sentences = datacleaning.constructLabeledSentences(allText)

INPUT_DIMENSION=300
''''Doc2Vec_model training'''
from gensim.models import Doc2Vec
text_model=None
filename='Doc2vec_model.d2v'
if os.path.isfile(filename):
    text_model = Doc2Vec.load(filename)
else:
    text_model = Doc2Vec(min_count=1, window=5, size=INPUT_DIMENSION, sample=1e-4, negative=5, workers=4, iter=5,seed=1)
    text_model.build_vocab(sentences)
    text_model.train(sentences, total_examples=text_model.corpus_count, epochs=text_model.iter)
    text_model.save(filename)

#text arrays initialize
text_train_arrays = np.zeros((train_size, INPUT_DIMENSION))
text_test_arrays = np.zeros((test_size, INPUT_DIMENSION))
for i in range(train_size):
    text_train_arrays[i] = text_model.docvecs['Text_'+str(i)]
j=0
for i in range(train_size,train_size+test_size):
    text_test_arrays[j] = text_model.docvecs['Text_'+str(i)]
    j=j+1

#print(text_train_arrays[0][:50])
'''TruncatedSVD:gene and vvaraitions'''
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=25, n_iter=25, random_state=12)
one_hot_gene = pd.get_dummies(all_data['Gene'])
truncated_one_hot_gene = svd.fit_transform(one_hot_gene.values)

one_hot_variation = pd.get_dummies(all_data['Variation'])
truncated_one_hot_variation = svd.fit_transform(one_hot_variation.values)

''' Merge Input features'''
train_set=np.hstack((truncated_one_hot_gene[:train_size],truncated_one_hot_variation[:train_size],text_train_arrays))
test_set=np.hstack((truncated_one_hot_gene[train_size:],truncated_one_hot_variation[train_size:],text_test_arrays))
#print(train_set[0][:50])


'''evaluate_feature'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(train_set,train_y,test_size=0.7, random_state=12)
clf = RandomForestClassifier(n_estimators=1000, max_depth=18, verbose=1).fit(X_train, Y_train)
clf.score(X_test, Y_test)

