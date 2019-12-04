import numpy as np
from services.comparers.comparer import Comparer


class EuclidianSimilarity(Comparer):
    def compare(self, object1, object2):
        return 1 / (1 + np.linalg.norm(object1 - object2))
