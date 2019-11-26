# Generated by Django 2.2.7 on 2019-11-26 14:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('apps', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='app',
            old_name='description',
            new_name='currency',
        ),
        migrations.RemoveField(
            model_name='app',
            name='url',
        ),
        migrations.AddField(
            model_name='app',
            name='prime_genre',
            field=models.TextField(blank=True),
        ),
    ]