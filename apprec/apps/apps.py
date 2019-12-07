import os
import pickle

import numpy as np
from django.apps import AppConfig
from sklearn.decomposition import PCA, TruncatedSVD

from services.utils.KMeansHelper import KMeansHelper
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
                tfidfs = np.array(list(pickle.load(f).values()))
                pca = PCA(n_components=4000)
                data = pca.fit_transform(tfidfs)

                print(pca.explained_variance_ratio_)

                with open("pca.pickle", "wb") as ft:
                    pickle.dump(data, ft)

                import matplotlib
                matplotlib.use('TkAgg')
                import matplotlib.pyplot as plt

                # Plotting the Cumulative Summation of the Explained Variance
                plt.figure()
                plt.plot(np.cumsum(pca.explained_variance_ratio_))
                plt.xlabel('Number of Components')
                plt.ylabel('Variance (%)')  # for each component
                plt.title('Explained Variance')
                plt.show()

        if not os.path.isfile("./pca-kmeans.pickle"):
            with open("pca.pickle", "rb") as f:
                print("computing pca kmeans clusters...")
                table = pickle.load(f)
                pca = np.array(list(table.values()))

                # KMeansHelper.compute_cluster_intertia_and_silhouette(pca, 2, 100)
                KMeansHelper.compute_clusters(table.keys(), pca, 93, "pca-kmeans.pickle")

        if not os.path.isfile("./lsa.pickle"):
            with open("tfidf_table.pickle", "rb") as f:
                print("computing lsa...")
                table = pickle.load(f)
                tfidfs = np.array(list(table.values()))
                lsa = TruncatedSVD(n_components=50)
                transformed = lsa.fit_transform(tfidfs)

                data = {}
                for id, tfidf in zip(table.keys(), transformed):
                    data[id] = tfidf

                with open("lsa.pickle", "wb") as ft:
                    pickle.dump(data, ft)

                import matplotlib
                matplotlib.use('TkAgg')
                import matplotlib.pyplot as plt

                # Plotting the Cumulative Summation of the Explained Variance
                plt.figure()
                plt.plot(np.cumsum(lsa.explained_variance_ratio_))
                plt.xlabel('Number of Components')
                plt.ylabel('Variance (%)')  # for each component
                plt.title('LSA - Explained Variance')
                plt.show()

        if not os.path.isfile("./lsa-kmeans.pickle"):
            with open("lsa.pickle", "rb") as f:
                print("computing lsa kmeans clusters...")
                table = pickle.load(f)
                lsa = np.array(list(table.values()))

                # KMeansHelper.compute_cluster_intertia_and_silhouette(lsa, 2, 100)
                KMeansHelper.compute_clusters(table.keys(), lsa, 45, "lsa-kmeans.pickle")
