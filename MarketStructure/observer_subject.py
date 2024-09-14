from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum


class Subject(ABC):
    def __init__(self):
        self._observers = defaultdict(list)
        self.handled = True

    def add_observer(self, observer_class, observer):
        self._observers[observer_class].append(observer)

    def remove_observer(self, observer_class, observer):
        self._observers[observer_class].remove(observer)

    def notify_observers(self, observer_class):
        if self.handled:
            for observer in self._observers[observer_class]:
                observer.update(self)
            self.handled = False

    def reset_observer_state(self):
        self.handled = True

class Observer(ABC):
    @abstractmethod
    def update(self, subject: Subject):
        raise NotImplementedError("Not Implemented")


class NodeObserverType(int, Enum):
    """
    Type of observer.
    """
    MARKET_STRUCTURE = 1
    CMS = 2
