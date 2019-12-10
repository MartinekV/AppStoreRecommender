import pickle
import graphene
from graphene_django import DjangoObjectType
from django.db.models import Q, Case, When

from services.recommenderService import RecommenderService
from services.recommenderService import comparisonMethod
from .models import App


def recommendedAppsHelper(appId, method, similar):
    recommended = RecommenderService.get_instance().get_recommended_apps(appId, method, similar, 10)
    preserved = Case(*[When(pk=pk, then=pos) for pos, pk in enumerate(recommended)])
    return App.objects.filter(id__in=recommended).order_by(preserved)


class AppType(DjangoObjectType):
    recommendedApps_tfidf_genre_similar = graphene.List(lambda: AppType)
    recommendedApps_tfidf_cluster_similar = graphene.List(lambda: AppType)
    recommendedApps_lsc_genre_similar = graphene.List(lambda: AppType)
    recommendedApps_lsc_cluster_similar = graphene.List(lambda: AppType)
    recommendedApps_tfidf_genre_opposite = graphene.List(lambda: AppType)
    recommendedApps_tfidf_cluster_opposite = graphene.List(lambda: AppType)
    recommendedApps_lsc_genre_opposite = graphene.List(lambda: AppType)
    recommendedApps_lsc_cluster_opposite = graphene.List(lambda: AppType)

    class Meta:
        model = App

    # comparisonMethod = ["tf-idf_genre", "tf-idf_cluster", "lsc_genre", "lsc_cluster"]


    def resolve_recommendedApps_tfidf_genre_similar(self, info):
        return recommendedAppsHelper(self.id, comparisonMethod[0], True)

    def resolve_recommendedApps_tfidf_cluster_similar(self, info):
        return recommendedAppsHelper(self.id, comparisonMethod[1], True)

    def resolve_recommendedApps_lsc_genre_similar(self, info):
        return recommendedAppsHelper(self.id, comparisonMethod[2], True)

    def resolve_recommendedApps_lsc_cluster_similar(self, info):
        return recommendedAppsHelper(self.id, comparisonMethod[3], True)

    def resolve_recommendedApps_tfidf_genre_opposite(self, info):
        return recommendedAppsHelper(self.id, comparisonMethod[0], False)

    def resolve_recommendedApps_tfidf_cluster_opposite(self, info):
        return recommendedAppsHelper(self.id, comparisonMethod[1], False)

    def resolve_recommendedApps_lsc_genre_opposite(self, info):
        return recommendedAppsHelper(self.id, comparisonMethod[2], False)

    def resolve_recommendedApps_lsc_cluster_opposite(self, info):
        return recommendedAppsHelper(self.id, comparisonMethod[3], False)

class Query(graphene.ObjectType):
    apps = graphene.List(AppType, id=graphene.String(), searchDesc=graphene.String(), searchTrackName=graphene.String())

    def resolve_apps(self, info, id=None, searchDesc = None, searchTrackName = None, **kwargs):
        # ToDo: extend this if it is necessary to filter using more/other attributes

        vectorizer = pickle.load(open("vectorizer.pickle", "rb"))  # load vectorizer
        desc = list(App.objects.values_list("app_desc", flat=True))[:1]  # load first app's description
        tfidf = vectorizer.transform(desc).toarray()  # transform that description

        print(tfidf)  # print it
        print(tfidf[:, vectorizer.vocabulary_.get("save")])  # print tfidfs of a word "save"

        if id:
            filterQuery = (
                Q(id=id)
            )
            return App.objects.using('default').filter(filterQuery)

        if searchDesc:
            filterQuery = (
                Q(app_desc__contains=searchDesc)
            )
            return App.objects.using('default').filter(filterQuery)

        if searchTrackName:
            filterQuery = (
                Q(track_name__contains=searchTrackName)
            )
            return App.objects.using('default').filter(filterQuery)

        return App.objects.using('default')

