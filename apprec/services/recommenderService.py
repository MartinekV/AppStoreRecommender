import pickle

from services.comparers.cosineSimilarity import CosineSimilarity
from services.comparers.euclidianSimilarity import EuclidianSimilarity
from services.methods.KNN import KNN


class RecommenderService(object):
    __instance = None

    tfidf_dict = {}
    lsa_labels = {}

    def __init__(self):
        if RecommenderService.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            RecommenderService.__instance = self
            with open("lsa.pickle", "rb") as f:
                self.tfidf_dict = pickle.load(f)
            with open("lsa-kmeans.pickle", "rb") as f:
                self.lsa_labels = pickle.load(f)

    def get_recommended_apps(self, id, method, similar, n):
        recommender = KNN()
        comparer = CosineSimilarity()
        # if method == "dfds":
        #     comparer =
        #     recommender =
        # elif method == "fdsds":
        # else:
        #     raise ValueError("Unknown method for recommending: " + method)

        similarities = self.__compute_lsa_kmeans_similarities(id, comparer)
        return recommender.get_recommended(similarities, similar, n)

    @staticmethod
    def get_instance():
        if RecommenderService.__instance is None:
            RecommenderService()
        return RecommenderService.__instance

    def __compute_tfidf_similarities(self, app_id, comparer):
        from apps.models import App

        app_tfidf = self.tfidf_dict[app_id]
        app = App.objects.get(id=app_id)
        apps = App.objects.filter(prime_genre=app.prime_genre).exclude(id=app_id)
        similarities = []

        for compared_app in apps:
            tfidf = self.tfidf_dict[compared_app.id]
            similarities.append((compared_app.id, comparer.compare(app_tfidf, tfidf)))

        return similarities

    def __compute_lsa_kmeans_similarities(self, app_id, comparer):
        from apps.models import App

        app_tfidf = self.tfidf_dict[app_id]
        app_cluster = self.lsa_labels[app_id]
        apps = App.objects.exclude(id=app_id)
        similarities = []

        for compared_app in apps:
            cluster = self.lsa_labels[compared_app.id]

            if cluster != app_cluster:
                continue

            similarities.append((compared_app.id, comparer.compare(app_tfidf, self.tfidf_dict[compared_app.id])))

        return similarities

