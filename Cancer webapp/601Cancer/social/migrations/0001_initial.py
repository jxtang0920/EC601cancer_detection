# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Member',
            fields=[
                ('username', models.CharField(max_length=16, serialize=False, primary_key=True)),
                ('password', models.CharField(max_length=16)),
                #('following', models.ManyToManyField(to='social.Member')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
    ]
    '''
        migrations.CreateModel(
            name='Profile',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('text', models.CharField(max_length=4096)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.AddField(
            model_name='member',
            #name='profile',
            #field=models.OneToOneField(null=True, to='social.Profile'),
            preserve_default=True,
        )'''