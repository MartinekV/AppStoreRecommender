from services.methods.recommenderMethod import RecommenderMethod


class KNN(RecommenderMethod):
    def get_recommended(self, similarities, similar, n):
        similarities.sort(reverse=similar, key=lambda sim: sim[1])

        start = 0
        if not similar:
            start = len(similarities) // 3 * 2
            if len(similarities)-start < 4:
                start = len(similarities)-4

        return [app[0] for app in similarities[start:start+n]]
