We use Django as the web app framework. The delpoyed website is: http://35.153.186.117:8000/social/

We are sorry to tell you this, but we are running out of the free usage of amazon EC2. So we have to shut down the website for now. If you want to see our website online. Please contact us so that we can deploy it online for you. Or you can download the project to try it offline. Thank you for understanding.

The final work is in the folder 601 Cancer.

To run the app, you need to install the Django and the Django rest framework.

The Django version is 1.11.8, to install you can use the command: pip install Django==1.11.8

The django rest framework is used in the project. To install Django rest framwork, you can use the command: pip install djangorestframework

To run the web app, in the path social, use command: python manage.py runserver However, the model we use in the webapp is not uploaded in the github becase the model is too big for github. So you can use the doc2vec2.py in EC601cancer_detection/Data analysis/ to generate the model before run the app localy. 

Other than the website, the classification function involves the scikit-learn, gensim, stopwords of nltk and pandas.
