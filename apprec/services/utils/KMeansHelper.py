import pickle
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import cm
from numpy.random.mtrand import RandomState
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


class KMeansHelper(object):
    @staticmethod
    def compute_cluster_intertia_and_silhouette(features, start, end):
        inertia = []
        sil = []
        for k in range(start, end+1):
            print("k =", k)
            km = MiniBatchKMeans(n_clusters=k, random_state=RandomState())
            pred = km.fit_predict(features)

            inertia.append((k, km.inertia_))
            sil.append((k, silhouette_score(features, pred)))
            print("inertia = ", km.inertia_, ", score = ", silhouette_score(features, pred))

        matplotlib.use('TkAgg')

        fig, ax = plt.subplots(1, 2, figsize=(15, 8))

        print("plotting inertia")
        x_iner = [x[0] for x in inertia]
        y_iner = [x[1] for x in inertia]
        ax[0].plot(x_iner, y_iner)
        ax[0].set_xlabel('Number of Clusters')
        ax[0].set_ylabel('Intertia')
        ax[0].set_title('Elbow Curve')

        print("plotting silhouette")
        x_sil = [x[0] for x in sil]
        y_sil = [x[1] for x in sil]
        ax[1].plot(x_sil, y_sil)
        ax[1].set_xlabel('Number of Clusters')
        ax[1].set_ylabel('Silhouette Score')
        ax[1].set_title('Silhouette Score Curve')

        plt.show()

    @staticmethod
    def compute_clusters(ids, features, k, file_name):
        km = MiniBatchKMeans(n_clusters=k, random_state=RandomState())
        pred = km.fit_predict(features)

        data = {}
        for id, cluster in zip(ids, pred):
            data[id] = cluster

        with open(file_name, "wb") as ft:
            pickle.dump(data, ft)

        pca = PCA(n_components=2).fit_transform(features)
        tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(features))

        colors = [cm.hsv(i / max(pred)) for i in pred]

        matplotlib.use('TkAgg')

        fig, ax = plt.subplots(1, 2, figsize=(15, 8))

        ax[0].scatter(pca[:, 0], pca[:, 1], c=colors)
        ax[0].set_title('PCA Cluster Plot')
        ax[1].scatter(tsne[:, 0], tsne[:, 1], c=colors)
        ax[1].set_title('TSNE Cluster Plot')

        plt.show()
