from django.db import models

# Create your models here.

class Query(models.Model):
   initial = models.TextField()
   years = models.TextField()
   rate = models.TextField()
   final = models.TextField()