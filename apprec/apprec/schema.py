import graphene

import apps.schema


class Query(apps.schema.Query, graphene.ObjectType):
    pass


schema = graphene.Schema(query=Query)