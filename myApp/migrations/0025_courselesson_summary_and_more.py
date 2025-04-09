# Generated by Django 5.0.7 on 2025-03-24 01:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myApp', '0024_liveclass_notification'),
    ]

    operations = [
        migrations.AddField(
            model_name='courselesson',
            name='summary',
            field=models.TextField(blank=True, editable=False, null=True),
        ),
        migrations.AlterField(
            model_name='lessonvideosubtitle',
            name='language',
            field=models.CharField(choices=[('English', 'English'), ('Simplified English', 'Simplified English'), ('Malayalam', 'Malayalam'), ('Tamil', 'Tamil')], default='English', max_length=20),
        ),
    ]
