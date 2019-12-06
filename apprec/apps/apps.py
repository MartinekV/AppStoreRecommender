import os
import pickle

import numpy as np
from django.apps import AppConfig
from numpy.random.mtrand import RandomState
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

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

        if not os.path.isfile("./kmeans.pickle"):
            with open("pca.pickle", "rb") as f:
                print("computing kmeans clusters...")
                tfidfs = np.array(list(pickle.load(f).values()))
                inertia = []
                sil = []
                genres = list(App.objects.values_list("prime_genre", flat=True).distinct())
                genre_count = len(genres)
                print("genre count =", genre_count)
                for k in range(2, 100):
                    print("k =", k)
                    km = MiniBatchKMeans(n_clusters=k, random_state=RandomState())
                    pred = km.fit_predict(tfidfs)

                    inertia.append((k, km.inertia_))
                    sil.append((k, silhouette_score(tfidfs, pred)))
                    print("inertia = ", km.inertia_, ", score = ", silhouette_score(tfidfs, pred))

                import matplotlib
                matplotlib.use('TkAgg')
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 2, figsize=(15, 8))

                print("plotting inertia")
                x_iner = [x[0] for x in inertia]
                y_iner = [x[1] for x in inertia]
                ax[0].plot(x_iner, y_iner)
                ax[0].set_xlabel('Number of Clusters')
                ax[0].set_ylabel('Intertia')
                ax[0].set_title('Elbow Curve')

                print("plotting silhoute")
                x_sil = [x[0] for x in sil]
                y_sil = [x[1] for x in sil]
                ax[1].plot(x_sil, y_sil)
                ax[1].set_xlabel('Number of Clusters')
                ax[1].set_ylabel('Silhouetter Score')
                ax[1].set_title('Silhouetter Score Curve')

                plt.show()
