from rest_framework import serializers

from .models import Member

# https://docs.djangoproject.com/en/dev/topics/db/models/#meta-options
'''
class ProfileSerializer(serializers.HyperlinkedModelSerializer):
   member = serializers.HyperlinkedRelatedField(
      many=False,
      read_only=True,
      view_name='member-detail'
   )
   class Meta:
      model = Profile
      fields = ('text', 'member')
'''
class MemberSerializer(serializers.HyperlinkedModelSerializer):
   class Meta:
      model = Member
      fields = ('url', 'username', 'profile', 'following')
      # read_only_fields = ('username', 'profile')
      # depth = 1