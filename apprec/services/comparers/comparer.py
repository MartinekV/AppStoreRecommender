from abc import ABC, abstractmethod


class Comparer(ABC):
    @abstractmethod
    def compare(self, object1, object2):
        pass
