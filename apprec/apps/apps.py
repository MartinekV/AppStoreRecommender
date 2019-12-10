import os
import pickle
import numpy as np

from django.apps import AppConfig
from sklearn.decomposition import PCA, TruncatedSVD
from services.utils.KMeansHelper import KMeansHelper
from services.utils.ModelHelper import ModelHelper
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

        if not os.path.isfile("./pca.pickle"):
            with open("tfidf_table.pickle", "rb") as f:
                print("computing pca...")
                pca = PCA(n_components=3000)
                ModelHelper.fit_model_and_transform(pca, pickle.load(f), "pca.pickle")

        if not os.path.isfile("./tfidf-kmeans.pickle"):
            with open("tfidf_table.pickle", "rb") as f:
                print("computing tfidf kmeans clusters...")
                table = pickle.load(f)
                tfidf = np.array(list(table.values()))

                #KMeansHelper.compute_cluster_intertia_and_silhouette(tfidf, 2, 100)
                KMeansHelper.compute_clusters(table.keys(), tfidf, 32, "tfidf-kmeans.pickle")

        if not os.path.isfile("./pca-kmeans.pickle"):
            with open("pca.pickle", "rb") as f:
                print("computing pca kmeans clusters...")
                table = pickle.load(f)
                pca = np.array(list(table.values()))

                # KMeansHelper.compute_cluster_intertia_and_silhouette(pca, 2, 100)
                KMeansHelper.compute_clusters(table.keys(), pca, 27, "pca-kmeans.pickle")

        if not os.path.isfile("./lsa.pickle"):
            with open("tfidf_table.pickle", "rb") as f:
                print("computing lsa...")
                lsa = TruncatedSVD(n_components=50)
                ModelHelper.fit_model_and_transform(lsa, pickle.load(f), "lsa.pickle")

        if not os.path.isfile("./lsa-kmeans.pickle"):
            with open("lsa.pickle", "rb") as f:
                print("computing lsa kmeans clusters...")
                table = pickle.load(f)
                lsa = np.array(list(table.values()))

                # KMeansHelper.compute_cluster_intertia_and_silhouette(lsa, 2, 100)
                KMeansHelper.compute_clusters(table.keys(), lsa, 57, "lsa-kmeans.pickle")


