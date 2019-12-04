from abc import ABC, abstractmethod


class RecommenderMethod(ABC):
    @abstractmethod
    def get_recommended(self, similarities, similar, n):
        pass
