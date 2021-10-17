# Generated by Django 3.2.8 on 2021-10-16 07:52

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ParkinsonPredict',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('MDVP_Fo', models.FloatField()),
                ('MDVP_Fhi', models.FloatField()),
                ('MDVP_Flo', models.FloatField()),
                ('MDVP_Jitter', models.FloatField()),
                ('MDVP_Jitter_Abs', models.FloatField()),
                ('MDVP_RAP', models.FloatField()),
                ('MDVP_PPQ', models.FloatField()),
                ('Jitter_DDP', models.FloatField()),
                ('MDVP_Shimmer', models.FloatField()),
                ('MDVP_Shimmer_dB', models.FloatField()),
                ('Shimmer_APQ3', models.FloatField()),
                ('Shimmer_APQ5', models.FloatField()),
                ('MDVP_APQ', models.FloatField()),
                ('Shimmer_DDA', models.FloatField()),
                ('NHR', models.FloatField()),
                ('HNR', models.FloatField()),
                ('RPDE', models.FloatField()),
                ('DFA', models.FloatField()),
                ('spread1', models.FloatField()),
                ('spread2', models.FloatField()),
                ('D2', models.FloatField()),
                ('PPE', models.FloatField()),
                ('status', models.CharField(max_length=200)),
            ],
        ),
    ]
