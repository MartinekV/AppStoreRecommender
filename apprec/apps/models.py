from django.db import models

# Create your models here.
class App(models.Model):
    currency = models.TextField(blank=True)
    prime_genre = models.TextField(blank=True)