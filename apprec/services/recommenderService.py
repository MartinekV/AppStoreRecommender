import pickle

from services.comparers.cosineSimilarity import CosineSimilarity
from services.comparers.euclidianSimilarity import EuclidianSimilarity
from services.methods.KNN import KNN


comparisonMethod = ["tf-idf_genre", "tf-idf_cluster", "lsc_genre", "lsc_cluster"]


class RecommenderService(object):
    __instance = None

    tfidf_dict = {}
    lsa_dict = {}
    tfidf_labels = {}
    lsa_labels = {}

    def __init__(self):
        if RecommenderService.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            RecommenderService.__instance = self
            with open("tfidf_table.pickle", "rb") as f:
                self.tfidf_dict = pickle.load(f)
            with open("lsa.pickle", "rb") as f:
                self.lsa_dict = pickle.load(f)
            with open("lsa-kmeans.pickle", "rb") as f:
                self.lsa_labels = pickle.load(f)
            with open("tfidf-kmeans.pickle", "rb") as f:
                self.tfidf_labels = pickle.load(f)

    def get_recommended_apps(self, id, method, similar, n):
        recommender = KNN()
        comparer = CosineSimilarity()

        comparisonMethod = ["tf-idf_genre", "tf-idf_cluster", "lsc_genre", "lsc_cluster"]
        if method == comparisonMethod[0]:
            features = self.tfidf_dict
            similarities = self.__compute_genre_similarities(id, features, comparer)
        elif method == comparisonMethod[1]:
            features = self.tfidf_dict
            labels = self.tfidf_labels
            similarities = self.__compute_kmeans_similarities(id, features, labels, comparer)
        elif method == comparisonMethod[2]:
            features = self.lsa_dict
            similarities = self.__compute_genre_similarities(id, features, comparer)
        elif method == comparisonMethod[3]:
            features = self.lsa_dict
            labels = self.lsa_labels
            similarities = self.__compute_kmeans_similarities(id, features, labels, comparer)
        else:
            raise ValueError("Unknown method for recommending: " + method)

        return recommender.get_recommended(similarities, similar, n)

    @staticmethod
    def get_instance():
        if RecommenderService.__instance is None:
            RecommenderService()
        return RecommenderService.__instance

    def __compute_genre_similarities(self, app_id, features, comparer):
        from apps.models import App

        app_tfidf = features[app_id]
        app = App.objects.get(id=app_id)
        apps = App.objects.filter(prime_genre=app.prime_genre).exclude(id=app_id)
        similarities = []

        for compared_app in apps:
            tfidf = features[compared_app.id]
            similarities.append((compared_app.id, comparer.compare(app_tfidf, tfidf)))

        return similarities

    def __compute_kmeans_similarities(self, app_id, features, labels, comparer):
        from apps.models import App

        app_tfidf = features[app_id]
        app_cluster = labels[app_id]
        apps = App.objects.exclude(id=app_id)
        similarities = []

        for compared_app in apps:
            cluster = labels[compared_app.id]

            if cluster != app_cluster:
                continue

            similarities.append((compared_app.id, comparer.compare(app_tfidf, features[compared_app.id])))

        return similarities

