import pickle
import graphene
from graphene_django import DjangoObjectType
from django.db.models import Q
from .models import App


class AppType(DjangoObjectType):
    recommendedApps = graphene.List(lambda: AppType)

    class Meta:
        model = App

    def resolve_recommendedApps(self, info):
        # ToDo call some recommenderSys algo to return recommended apps
        # object 'self' is current App
        return [self, self, self, self]


class Query(graphene.ObjectType):
    apps = graphene.List(AppType, id=graphene.String())

    def resolve_apps(self, info, id=None, **kwargs):
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
        return App.objects.using('default')

