# Generated by Django 5.0.7 on 2025-02-17 01:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myApp', '0020_event'),
    ]

    operations = [
        migrations.AddField(
            model_name='event',
            name='location_url',
            field=models.URLField(blank=True, max_length=250, null=True),
        ),
    ]
