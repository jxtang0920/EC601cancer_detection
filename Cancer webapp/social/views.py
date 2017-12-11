from django.shortcuts import render
from django.http import HttpResponse, Http404
from django.template import RequestContext, loader

from social.models import Member, Profile
from .forms import UploadFileForm
# REST imports
from rest_framework import viewsets
from .serializers import ProfileSerializer, MemberSerializer
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

class ProfileViewSet(viewsets.ModelViewSet):
    # API endpoint for listing and creating profiles
    queryset = Profile.objects.order_by('text')
    serializer_class = ProfileSerializer

class MemberViewSet(viewsets.ModelViewSet):
    # API endpoint for listing and creating members
    queryset = Member.objects.order_by('username')
    serializer_class = MemberSerializer

appname = 'Cancer classification'

def test(request):
   # response = HttpResponse("{name : 'Paulo'}", content_type="application/json")
   # response['ETag'] = '134112341324'
   # return HttpResponse("", status=404)
   from django.http import JsonResponse
   response = JsonResponse({'foo': 'bar'})
   return response

def index(request):
   context = { 'appname': appname }
   return render(request, 'social/index.html', context)
'''
def messages(request):
   if 'username' in request.session:
      username = request.session['username']
      context = {
         'appname': appname,
         'username': username,
         'loggedin': True
      }
      return render(request, 'social/messages.html', context)
   else:
      raise Http404("User is not logged it, no access to messages page!")
'''
def signup(request):
   context = { 'appname': appname }
   return render(request, 'social/signup.html', context)

def register(request):
   u = request.POST['user']
   p = request.POST['pass']
   user = Member(username=u, password=p)
   user.save()
   context = {
     'appname': appname,
     'username' : u
   }
   return render(request, 'social/user-registered.html', context)

def login(request):
   if 'username' not in request.POST:
      context = { 'appname': appname }
      return render(request, 'social/login.html', context)
   else:
      u = request.POST['username']
      p = request.POST['password']
      try:
         member = Member.objects.get(pk=u)
      except Member.DoesNotExist:
         raise Http404("User does not exist")
      if member.password == p:
         request.session['username'] = u
         request.session['password'] = p
         return render(request, 'social/login.html', {
             'appname': appname,
             'username': u,
             'loggedin': True}
             )
      else:
         raise Http404("Incorrect password")

def logout(request):
   if 'username' in request.session:
      u = request.session['username']
      request.session.flush()        
      context = {
         'appname': appname,
         'username': u
      }
      return render(request, 'social/logout.html', context)
   else:
      raise Http404("Can't logout, you are not logged in")

def member(request):
   if 'username' in request.session:
      username = request.session['username']
      #member = Member.objects.get(pk=view_user)

      if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        result = handle_upload_file(request.FILES['file'],str(request.FILES['file']))
        context = {
          'result' : result,
          'form' : form,
          'username' : username,
          'loggedin': True
        }
        return render(request, 'social/member.html', context)
      else:
        form = UploadFileForm()
        result = "Choose the file and upload, then you can get the result!"
        context = {
          'result' : result,
          'form' : form,
          'username' : username,
          'loggedin': True
        }
      return render(request, 'social/member.html', context)
 
   else:
      raise Http404("User is not logged it, no access to members page!")
  
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
    clf = joblib.load("social/train_model.m")
    result=clf.predict_proba(input_set)
    mystring = ""
    for digit in result:
        mystring += str(digit)
    return mystring

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

'''
def friends(request):
   if 'username' in request.session:
      username = request.session['username']
      member_obj = Member.objects.get(pk=username)
      # list of people I'm following
      following = member_obj.following.all()
      # list of people that are following me
      followers = Member.objects.filter(following__username=username)
      # render reponse
      return render(request, 'social/friends.html', {
         'appname': appname,
         'username': username,
         'members': members,
         'following': following,
         'followers': followers,
         'loggedin': True}
         )
   else:
      raise Http404("User is not logged it, no access to members page!")

def members(request):
   if 'username' in request.session:
      username = request.session['username']
      member_obj = Member.objects.get(pk=username)
      # follow new friend
      if 'add' in request.GET:
         friend = request.GET['add']
         friend_obj = Member.objects.get(pk=friend)
         member_obj.following.add(friend_obj)
         member_obj.save()
      # unfollow a friend
      if 'remove' in request.GET:
         friend = request.GET['remove']
         friend_obj = Member.objects.get(pk=friend)
         member_obj.following.remove(friend_obj)
         member_obj.save()
      # view user profile
      if 'view' in request.GET:
         return member(request, request.GET['view'])
      else:
         # list of all other members
         members = Member.objects.exclude(pk=username)
         # list of people I'm following
         following = member_obj.following.all()
         # list of people that are following me
         followers = Member.objects.filter(following__username=username)
         # render reponse
         return render(request, 'social/members.html', {
            'appname': appname,
            'username': username,
            'members': members,
            'following': following,
            'followers': followers,
            'loggedin': True}
            )
   else:
      raise Http404("User is not logged it, no access to members page!")

def profile(request):
   if 'username' in request.session:
      u = request.session['username']
      member = Member.objects.get(pk=u)
      if 'text' in request.POST:
         text = request.POST['text']
         if member.profile:
             member.profile.text = text
             member.profile.save()
         else:
             profile = Profile(text=text)
             profile.save()
             member.profile = profile
         member.save()
      else:
         if member.profile:
             text = member.profile.text
         else:
             text = ""
      return render(request, 'social/profile.html', {
         'appname': appname,
         'username': u,
         'text' : text,
         'loggedin': True}
         )
   else:
      raise Http404("User is not logged it, no access to profiles!")
'''
def checkuser(request):
   if 'user' in request.POST:
      u = request.POST['user']
      try:
         member = Member.objects.get(pk=u)
      except Member.DoesNotExist:
         member = None
      if member is not None:
         return HttpResponse("<span class='taken'>&nbsp;&#x2718; This username is taken</span>")
      else:
         return HttpResponse("<span class='available'>&nbsp;&#x2714; This username is available</span>")
