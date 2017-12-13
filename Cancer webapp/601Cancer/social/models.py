from django.db import models

class Profile(models.Model):
	text = models.CharField(max_length=4096)
	member = models.ForeignKey('Member', on_delete=models.CASCADE, null=True, related_name='profile')
	def __str__(self):
		return self.text

class Member(models.Model):
    username = models.CharField(max_length=16,primary_key=True)
    password = models.CharField(max_length=16)
    #profile = models.OneToOneField(Profile, null=True, related_name='member')
    #following = models.ManyToManyField("self", symmetrical=False)

    def __str__(self):
        return self.username