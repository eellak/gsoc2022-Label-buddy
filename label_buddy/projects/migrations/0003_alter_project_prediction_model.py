# Generated by Django 3.2.3 on 2022-08-01 08:18

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('projects', '0002_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='project',
            name='prediction_model',
            field=models.ForeignKey(blank=True, default='', help_text='Prediciton Model for the project', null=True, on_delete=django.db.models.deletion.CASCADE, to='projects.predictionmodels', to_field='title'),
        ),
    ]