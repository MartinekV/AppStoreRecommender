from services.methods.recommenderMethod import RecommenderMethod


class KNN(RecommenderMethod):
    def get_recommended(self, similarities, similar, n):
        similarities.sort(reverse=similar, key=lambda sim: sim[1])

        return [app[0] for app in similarities[:n]]
