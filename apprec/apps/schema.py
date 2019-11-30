import graphene
from graphene_django import DjangoObjectType
from django.db.models.query import QuerySet

from .models import App


class AppType(DjangoObjectType):
    class Meta:
        model = App


class Query(graphene.ObjectType):
    apps = graphene.List(AppType)

    def resolve_apps(self, info, **kwargs):
        return App.objects.using('default')