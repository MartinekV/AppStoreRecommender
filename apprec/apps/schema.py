import graphene
from graphene_django import DjangoObjectType

from .models import App


class AppType(DjangoObjectType):
    class Meta:
        model = App


class Query(graphene.ObjectType):
    apps = graphene.App(AppType)

    def resolve_apps(self, info, **kwargs):
        return App.objects.all()