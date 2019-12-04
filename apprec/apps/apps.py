import os
import pickle

from django.apps import AppConfig

from services.comparers.cosineSimilarity import CosineSimilarity
from services.recommenderService import RecommenderService
from services.utils.tfIdfTransformer import TfIdfTransformer


class AppsConfig(AppConfig):
    name = 'apps'

    def ready(self):
        if os.environ.get("RUN_MAIN", None) == "true":
            return
        from apps.models import App

        if not os.path.isfile("./vectorizer.pickle"):
            corpus = list(App.objects.values_list("app_desc", flat=True))
            TfIdfTransformer.fit(corpus)

        if not os.path.isfile("./tfidf_table.pickle"):
            TfIdfTransformer.transform(App.objects.all())
