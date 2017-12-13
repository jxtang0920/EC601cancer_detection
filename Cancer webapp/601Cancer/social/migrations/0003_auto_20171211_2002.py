# -*- coding: utf-8 -*-
# Generated by Django 1.11.7 on 2017-12-12 01:02
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('social', '0002_auto_20171210_1615'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='member',
            name='profile',
        ),
        migrations.RemoveField(
            model_name='profile',
            name='resultid',
        ),
        migrations.AddField(
            model_name='profile',
            name='member',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='social.Member'),
        ),
    ]