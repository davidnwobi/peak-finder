from __future__ import annotations
import datetime as dt
import heapq
import typing as tp

import numpy as np
import pandas as pd
from .observer_subject import Subject, Observer, NodeObserverType
from dataclasses import dataclass
from enum import Enum


@dataclass
class PriorityQueueNode():
    date_index: tp.Optional[tp.Union[np.datetime64, dt.datetime]] = None
    position: tp.Optional[int] = 0
    node: tp.Optional[Node] = None

    def __lt__(self, other):
        return (self.date_index < other.date_index or
                self.date_index == other.date_index and self.position < other.position)

    def __eq__(self, other):
        return self.date_index == other.date_index and self.position == other.position and self.node == other.node


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def push(self, item):
        heapq.heappush(self.elements, item)

    def pop(self):
        if not self.is_empty():
            return heapq.heappop(self.elements)
        else:
            raise IndexError("Priority queue is empty")

    def is_empty(self):
        return not bool(self.elements)

    def __getitem__(self, index):
        return self.elements[index]


class MarketStructureState(int, Enum):
    """
    State of the market structure.
    """
    CERTAIN = 1
    POSSIBLE_TREND_CHANGE = 2
    INVALIDATED_FIXING = 3


class Node(Subject):
    """
    Represents a node in the market structure.
    """

    def __init__(self) -> None:
        """
        Initializes a new Node object.
        """
        super().__init__()
        self._date_index: tp.Optional[tp.Union[np.datetime64, dt.datetime]] = None
        self._value: tp.Optional[float] = None
        self._prev: tp.Optional[Node] = None
        self._next: tp.Optional[Node] = None
        self._is_permanent: bool = True
        self._is_cms: bool = False
        self._broke: bool = False
        self._was_cms: bool = False
        self._index: tp.Optional[int] = None
        self._trend: tp.Optional[str] = None
        self.COMMENT: str = ""
        self.parent = None

        # notify if any of the above attributes change

    @property
    def date_index(self):
        return self._date_index

    @date_index.setter
    def date_index(self, value):
        self._date_index = value
        self.notify_observers(NodeObserverType.MARKET_STRUCTURE)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self.notify_observers(NodeObserverType.MARKET_STRUCTURE)

    @property
    def prev(self):
        return self._prev

    @prev.setter
    def prev(self, value):
        self._prev = value
        self.notify_observers(NodeObserverType.MARKET_STRUCTURE)

    @property
    def next(self):
        return self._next

    @next.setter
    def next(self, value):
        self._next = value
        self.notify_observers(NodeObserverType.MARKET_STRUCTURE)

    @property
    def is_permanent(self):
        return self._is_permanent

    @is_permanent.setter
    def is_permanent(self, value):
        self._is_permanent = value
        self.notify_observers(NodeObserverType.MARKET_STRUCTURE)

    @property
    def is_cms(self):
        return self._is_cms

    @is_cms.setter
    def is_cms(self, value):
        self._is_cms = value
        self.notify_observers(NodeObserverType.MARKET_STRUCTURE)

    @property
    def broke(self):
        return self._broke

    @broke.setter
    def broke(self, value):
        self._broke = value
        self.notify_observers(NodeObserverType.MARKET_STRUCTURE)

    @property
    def was_cms(self):
        return self._was_cms

    @was_cms.setter
    def was_cms(self, value):
        self._was_cms = value
        self.notify_observers(NodeObserverType.MARKET_STRUCTURE)
        self.notify_observers(NodeObserverType.CMS)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value
        self.notify_observers(NodeObserverType.MARKET_STRUCTURE)

    @property
    def trend(self):
        return self._trend

    @trend.setter
    def trend(self, value):
        self._trend = value
        self.notify_observers(NodeObserverType.MARKET_STRUCTURE)

    def __copy__(self):
        new_node = self.__new__(self.__class__)
        new_node.__init__()
        new_node._date_index = self.date_index
        new_node._value = self.value
        new_node._is_permanent = self.is_permanent
        new_node._is_cms = self.is_cms
        new_node._broke = self.broke
        new_node._was_cms = self.was_cms
        new_node._index = self.index
        new_node._trend = self.trend
        new_node.COMMENT = self.COMMENT
        return new_node


class Peak(Node):
    """
    Represents a peak node in the market structure.
    """

    def __init__(self, date_index: tp.Optional[tp.Union[np.datetime64, dt.datetime]] = None,
                 value: tp.Optional[float] = None, next: tp.Optional[Trough] = None, prev: tp.Optional[Trough] = None,
                 is_permanent: bool = True, is_cms: bool = False, index: tp.Optional[int] = None) -> None:
        """
        Initializes a new Peak object.

        :param date_index: Date index associated with the peak.
        :param value: Value associated with the peak.
        :param next: Reference to the next Trough node.
        :param prev: Reference to the previous Trough node.
        :param is_permanent: Flag indicating if the peak is a permanent node.
        :param is_cms: Flag indicating if the peak is a change in market structure (CMS) point.
        :param index: Index associated with the peak.
        """

        super().__init__()
        self._date_index: tp.Optional[tp.Union[np.datetime64, dt.datetime]] = date_index
        self._value: tp.Optional[float] = value
        self._next: tp.Optional[Trough] = next
        self._prev: tp.Optional[Trough] = prev
        self._is_permanent: bool = is_permanent
        self._is_cms: bool = is_cms
        self._index: tp.Optional[int] = index

    def __repr__(self):

        output = f"Peak({self.value}"
        if self.date_index is not None:
            output += f", Date = {self.date_index}"
        if self.is_permanent is not None:
            output += f", is_permanent={self.is_permanent}"
        if self.is_cms is not None:
            output += f", is_cms={self.is_cms}"
        output += ")"
        return output

    def __eq__(self, other):
        if isinstance(other, Peak):
            return (
                    self.value == other.value and
                    self.date_index == other.date_index and
                    self.is_permanent == other.is_permanent and
                    self.is_cms == other.is_cms and
                    self.index == other.index
            )
        return False


class Trough(Node):
    """
    Represents a trough node in the market structure.
    """

    def __init__(self, date_index: tp.Optional[tp.Union[np.datetime64, dt.datetime]] = None,
                 value: tp.Optional[float] = None, next: tp.Optional[Peak] = None, prev: tp.Optional[Peak] = None,
                 is_permanent: bool = True, is_cms: bool = False, index: tp.Optional[int] = None) -> None:
        """
        Initializes a new Trough object.

        :param date_index: Date index associated with the trough.
        :param value: Value associated with the trough.
        :param next: Reference to the next Peak node.
        :param prev: Reference to the previous Peak node.
        :param is_permanent: Flag indicating if the trough is a permanent node.
        :param is_cms: Flag indicating if the trough is a change in market structure (CMS) point.
        :param index: Index associated with the trough.
        """
        super().__init__()
        self._date_index: tp.Optional[tp.Union[np.datetime64, dt.datetime]] = date_index
        self._value: tp.Optional[float] = value
        self._next: tp.Optional[Peak] = next
        self._prev: tp.Optional[Peak] = prev
        self._is_permanent: bool = is_permanent
        self._is_cms: bool = is_cms
        self._index: tp.Optional[int] = index

    def __repr__(self):

        output = f"Trough({self.value}"
        if self.date_index is not None:
            output += f", Date = {self.date_index}"
        if self.is_permanent is not None:
            output += f", is_permanent={self.is_permanent}"
        if self.is_cms is not None:
            output += f", is_cms={self.is_cms}"
        output += ")"
        return output

    def __eq__(self, other):
        if isinstance(other, Trough):
            return (
                    self.value == other.value and
                    self.date_index == other.date_index and
                    self.is_permanent == other.is_permanent and
                    self.is_cms == other.is_cms and
                    self.index == other.index
            )
        return False
