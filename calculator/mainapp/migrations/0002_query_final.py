# -*- coding: utf-8 -*-
# Generated by Django 1.10.5 on 2017-01-10 14:05
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mainapp', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='query',
            name='final',
            field=models.TextField(default=0),
            preserve_default=False,
        ),
    ]
