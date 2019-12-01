import os
from django.apps import AppConfig
from services.utils.tfIdfTransformer import TfIdfTransformer


class AppsConfig(AppConfig):
    name = 'apps'

    def ready(self):
        if os.environ.get("RUN_MAIN", None) == "true" or os.path.isfile("./vectorizer.pickle"):
            return

        from apps.models import App

        corpus = list(App.objects.values_list("app_desc", flat=True))
        TfIdfTransformer.fit(corpus)
