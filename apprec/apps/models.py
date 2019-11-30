from django.db import models


# Create your models here.
class App(models.Model):
    # id = models.AutoField()
    track_name = models.TextField(blank=True)
    size_bytes = models.TextField(blank=True)
    currency = models.TextField(blank=True)
    price = models.TextField(blank=True)
    rating_count_tot = models.TextField(blank=True)
    rating_count_ver = models.TextField(blank=True)
    user_rating = models.TextField(blank=True)
    user_rating_ver = models.TextField(blank=True)
    ver = models.TextField(blank=True)
    cont_rating = models.TextField(blank=True)
    prime_genre = models.TextField(blank=True)
    sup_devices_num = models.TextField(blank=True)
    ipadSc_urls_num = models.TextField(blank=True)
    lang_num = models.TextField(blank=True)
    vpp_lic = models.TextField(blank=True)
    app_desc = models.TextField(blank=True)
