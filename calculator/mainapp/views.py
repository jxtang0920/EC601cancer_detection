from __future__ import print_function
from django.shortcuts import render
from django.http import HttpResponse
from .forms import UploadFileForm
from .models import *


import os
import re
import string
import pandas as pd
import numpy as np
import warnings
from sklearn.externals import joblib
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from nltk.corpus import stopwords
from gensim.models.doc2vec import LabeledSentence
from gensim import utils
from gensim.models import Doc2Vec


def completeData(request):
    return 'ID' in request.GET


        

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

def index(request):
    if completeData(request):
        P = request.GET['ID']
        Y = request.GET['des']
        form = UploadFileForm(None)
        result=1
        context = {
            'ID' : P,
            'des' : Y,
            'result' : result,
            'final' : 1
        }
    else:
        form = UploadFileForm(None)
        context = {
            'final' : -1
        }
    return render(request, 'mainapp/index.html', context)


def upload(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        result = handle_upload_file(request.FILES['file'],str(request.FILES['file']))
        #result = "1234"
        #return HttpResponse('Successful')
        context = {
            'result' : result,
            'form' : form,
        }
        return render(request, 'mainapp/upload.html', context)
    else:
        form = UploadFileForm()
        result = "abc"
    return render(request, 'mainapp/upload.html', {'form': form}, {"result",result})
  
def handle_upload_file(file,filename):  
    path='media/uploads/'     #上传文件的保存路径，可以自己指定任意的路径  
    if not os.path.exists(path):  
        os.makedirs(path)  
    with open(path+filename,'wb+')as destination:  
        for chunk in file.chunks():  
            destination.write(chunk)
    input_text = pd.read_csv("media/uploads/input_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
    input_size = len(input_text)
    inputText = input_text['Text'].apply(datacleaning.cleanup)
    sentences = datacleaning.constructLabeledSentences(inputText)
    INPUT_DIMENSION=300
    inputText_arrays = np.zeros((input_size, INPUT_DIMENSION))
    text_model = Doc2Vec(min_count=1, window=5, size=INPUT_DIMENSION, sample=1e-4, negative=5, workers=4, iter=5,seed=1)
    text_model.build_vocab(sentences)
    text_model.train(sentences, total_examples=text_model.corpus_count, epochs=text_model.iter)
    for i in range(input_size):
        inputText_arrays[i] = text_model.docvecs['Text_'+str(i)]
    input_set=inputText_arrays
    from sklearn.externals import joblib
    clf = joblib.load("mainapp/train_model.m")
    result=clf.predict_proba(input_set)
    mystring = ""
    for digit in result:
        mystring += str(digit)
    return mystring
