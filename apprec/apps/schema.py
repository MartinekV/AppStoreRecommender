import pickle
import graphene
from graphene_django import DjangoObjectType
from django.db.models import Q, Case, When

from services.recommenderService import RecommenderService
from .models import App


class AppType(DjangoObjectType):
    recommendedApps = graphene.List(lambda: AppType)

    class Meta:
        model = App

    def resolve_recommendedApps(self, info):
        # ToDo call some recommenderSys algo to return recommended apps
        # object 'self' is current App
        recommended = RecommenderService.get_instance().get_recommended_apps(self.id, "", True, 20)
        preserved = Case(*[When(pk=pk, then=pos) for pos, pk in enumerate(recommended)])
        return App.objects.filter(id__in=recommended).order_by(preserved)


class Query(graphene.ObjectType):
    apps = graphene.List(AppType, id=graphene.String(), searchDesc=graphene.String())

    def resolve_apps(self, info, id=None, searchDesc = None, **kwargs):
        # ToDo: extend this if it is necessary to filter using more/other attributes

        vectorizer = pickle.load(open("vectorizer.pickle", "rb"))  # load vectorizer
        print(vectorizer.get_feature_names())  # print all known words
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

        return App.objects.using('default')

