import numpy as np
from services.comparers.comparer import Comparer


class CosineSimilarity(Comparer):
    def compare(self, object1, object2):
        return np.dot(object1, object2) / (np.linalg.norm(object1) * np.linalg.norm(object2))
