from __future__ import annotations

from dataclasses import dataclass
import functools
import typing as tp
import datetime as dt
import pandas as pd
import copy
from enum import Enum
from sys import setrecursionlimit
from .observer_subject import Subject, Observer
from .node import Node, NodeObserverType, PriorityQueueNode, Peak, Trough
from collections import namedtuple
from .logger import log_function_call, flow_logger

setrecursionlimit(100000)

NodeTuple = namedtuple("NodeTuple", ["date", "is_peak", "value", "is_permanent", "was_cms"])


# class syntax

class Direction(str, Enum):
    """
    Direction of traversal.
    """
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass
class MarketStructure(Observer):
    """
    A doubly-linked list of peaks and troughs.`
    """

    def __init__(self, start_node: tp.Optional[Node] = None):
        """
        Initializes a new MarketStructure object.

        :param start_node: Optional starting node for the market structure.
        """
        self._head_: tp.Optional[Node] = start_node
        self._tail_: tp.Optional[Node] = start_node
        self._length_: int = 1 if start_node is not None else 0
        self.changed_nodes = []

    def update(self, subject: Subject):
        if isinstance(subject, Node) and subject.date_index is not None:
            if subject.prev and subject.prev.date_index == subject.date_index:
                self.changed_nodes.append(PriorityQueueNode(subject.date_index, 1, subject))
            else:
                self.changed_nodes.append(PriorityQueueNode(subject.date_index, 0, subject))

        # print(f"Subject {subject} has changed")

    @property
    def head(self) -> tp.Optional[Node]:
        """
        The first node in the market structure.
        """
        return self._head_

    @property
    def tail(self) -> tp.Optional[Node]:
        """
        The last node in the market structure.
        """
        return self._tail_

    def __len__(self):
        """
        Returns the length of the market structure.
        """
        return self._length_

    def __iter__(self):
        """
        Iterates over the market structure nodes.
        """
        current_node = self._head_
        while current_node:
            yield current_node
            current_node = current_node.next

    def __repr__(self):
        return f"PeakDoublyLinked({[node.value for node in self]})"

    def slice_list(self, start_node: tp.Optional[Node] = None,
                   end_node: tp.Optional[Node] = None, inplace: bool = False) -> MarketStructure:
        """

        Slice the market structure from start_node to end_node

        :param start_node: The starting node for the slice.
        :param end_node: The ending node for the slice.
        :param inplace: If True, the nodes will be removed from the market structure.
        :return:
        """

        if (start_node and start_node.parent is not self) or (end_node and end_node.parent is not self):
            raise ValueError("Node is not in this market structure")
        start_node = self._head_ if start_node is None else start_node
        end_node = self._tail_ if end_node is None else end_node
        new_list = MarketStructure()
        current_node = start_node

        while current_node and current_node is not end_node:
            next_node = current_node.next
            if inplace:
                self._remove_node(current_node)
                new_list.append(current_node)
            else:
                new_list.append(copy.copy(current_node))

            current_node = next_node

        if inplace:
            self._remove_node(current_node)
            new_list.append(current_node)
        else:
            new_list.append(copy.copy(current_node))

        return new_list

    def remove(self, direction: str = Direction.FORWARD, start_node: tp.Optional[Node] = None,
               end_node: tp.Optional[Node] = None):
        """
        Removes a range of nodes from the market structure.
        This function is unavoidably different. It will remove the nodes in the range including the start and end node.

        :param direction: The direction of removal, either "forward" or "backward".
        :param start_node: The starting node for the removal.
        :param end_node: The ending node for the removal.

        """
        if (start_node and start_node.parent is not self) or (end_node and end_node.parent is not self):
            raise ValueError("Node is not in this market structure")
        if direction == Direction.FORWARD:
            start_node = self._head_ if start_node is None else start_node
            end_node = self._tail_ if end_node is None else end_node
            current_node = start_node
            while current_node and current_node is not end_node:
                next_node = current_node.next
                self._remove_node(current_node)
                current_node = next_node
            self._remove_node(current_node)
        else:
            start_node = self._tail_ if start_node is None else start_node
            end_node = self._head_ if end_node is None else end_node

            current_node = start_node
            while current_node and current_node is not end_node:
                prev_node = current_node.prev
                self._remove_node(current_node)
                current_node = prev_node
            self._remove_node(current_node)

    def __eq__(self, other):
        if isinstance(other, MarketStructure):
            if len(self) != len(other):
                return False
            for node1, node2 in zip(self, other):
                if node1 is not node2:
                    return False
            return True
        return False

    def _remove_node(self, node: Node):
        node.remove_observer(NodeObserverType.MARKET_STRUCTURE, self)
        node.parent = None
        if node is self._head_:
            self._head_ = node.next
            if self._head_ is not None:
                self._head_.prev = None
            if len(self) == 1:
                self._tail_ = None
        if node is self._tail_:
            self._tail_ = node.prev
            if self._tail_ is not None:
                self._tail_.next = None
            if len(self) == 1:
                self._head_ = None
        if node.prev is not None:
            node.prev.next = node.next
        if node.next is not None:
            node.next.prev = node.prev
        node.prev = None
        node.next = None
        self._length_ -= 1

    def append(self, new_node: Node):
        """
        Appends a new node to the market structure.

        :param new_node: The node to be appended.
        """
        new_node.add_observer(NodeObserverType.MARKET_STRUCTURE, self)
        new_node.parent = self
        if self._head_ is None:
            self._head_ = new_node
            self._tail_ = new_node
        else:
            new_node.prev = self._tail_  # pyright: ignore[reportOptionalMemberAccess]
            self._tail_.next = new_node  # pyright: ignore[reportOptionalMemberAccess]
            self._tail_ = new_node

        self._length_ += 1

    def popback(self):
        """
        Removes and returns the last node from the market structure.

        :return: The removed node.
        """
        if self._tail_ is None:
            raise IndexError("pop from empty list")
        node = self._tail_
        self._remove_node(self.tail)
        return node

    def popfront(self):

        """
        Removes and returns the first node from the market structure.
        :return: The removed node.
        """

        if self._head_ is None:
            raise IndexError("pop from empty list")
        node = self._head_
        self._remove_node(self.head)
        return node

    def insert(self, index: int, node: Node):
        """
        Inserts a new node at a specified index in the market structure.

        :param index: Index at which to insert the new node.
        :param node: The node to be inserted.
        """
        # implement negative indexing

        index = self._length_ + index if index < 0 else index
        if index < 0 or index > self._length_:
            raise IndexError("Index out of range")
        node.add_observer(NodeObserverType.MARKET_STRUCTURE, self)
        node.parent = self
        if index == 0:

            node.next = self._head_
            self._head_.prev = node  # pyright: ignore[reportOptionalMemberAccess]
            self._head_ = node
        elif index >= self._length_:
            node.prev = self._tail_
            self._tail_.next = node  # pyright: ignore[reportOptionalMemberAccess]
            self._tail_ = node
        else:
            current_node = self._head_
            counter = 0
            while counter != index:
                current_node = current_node.next  # pyright: ignore[reportOptionalMemberAccess]
                counter += 1
            node.prev = current_node.prev  # pyright: ignore[reportOptionalMemberAccess]
            node.next = current_node
            current_node.prev.next = node  # pyright: ignore[reportOptionalMemberAccess]
            current_node.prev = node  # pyright: ignore[reportOptionalMemberAccess]

        self._length_ += 1

    def find(self, what: tp.Callable, direction: str = Direction.FORWARD, start_node: Node = None,
             end_node: Node = None) -> \
            tp.Optional[Node]:
        """
        Finds a specific node in the market structure by searching forward or backward from a specified node.
        It is start and end node inclusive

        :param end_node: The end node for the search.
        :param what: Search criteria.
        :param start_node: The starting node for the search.
        :param direction: The direction of search, either "forward" or "backward".
        :return: The found node or None if not found.
        """

        temp: tp.Optional[Node] = None
        if direction == Direction.FORWARD:
            start_node = self._head_ if start_node is None else start_node
            end_node = self._tail_ if end_node is None else end_node
            temp = start_node
            while temp and not what(temp):
                temp = temp.next
                if temp is end_node.next:
                    temp = None
        elif direction == Direction.BACKWARD:
            start_node = self._tail_ if start_node is None else start_node
            end_node = self._head_ if end_node is None else end_node
            temp = start_node
            while temp and not what(temp):
                temp = temp.prev
                if temp is end_node.prev:
                    temp = None
        else:
            raise ValueError("direction must either be 'forward' or 'backward'")
        return temp

    def index(self, node: Node, start: int = 0, stop: int = None):
        """
        Returns the index of a node within a specified range.

        :param node: The node to find the index of.
        :param start: The starting index of the search range.
        :param stop: The stopping index of the search range.
        :return: The index of the node.
        """
        if node.parent is not self:
            raise ValueError("Node is not in this market structure")
        if stop is None:
            stop = len(self)

        if start < 0:
            start += len(self)
        if stop < 0:
            stop += len(self)

        current_node = self._head_
        index = 0
        while current_node is not None and index < start:
            current_node = current_node.next
            index += 1

        while current_node is not None and index < stop:
            if current_node is node:
                return index
            current_node = current_node.next
            index += 1

        raise ValueError(f"{node} is not in the list")

    def _from_num_list(self, data: tp.List[tp.Union[int, float]]):
        current_date = dt.datetime.now()
        i = 0
        while i < len(data):
            current_date = current_date + dt.timedelta(seconds=1)
            self.append(Peak(dt.datetime.now(), data[i]))
            i += 1
            if i >= len(data):
                break
            current_date = current_date + dt.timedelta(seconds=1)
            self.append(Trough(dt.datetime.now(), data[i]))
            i += 1

    def _from_num_list_with_bool(self, data: tp.List[tp.Tuple[tp.Union[int, float], bool]]):
        current_date = dt.datetime.now()
        i = 0
        while i < len(data):
            current_date = current_date + dt.timedelta(seconds=1)
            self.append(Peak(current_date, data[i][0], is_permanent=data[i][1]))
            i += 1
            if i >= len(data):
                break
            current_date = current_date + dt.timedelta(seconds=1)
            self.append(Trough(current_date, data[i][0], is_permanent=data[i][1]))
            i += 1

    def _from_num_list_with_bool_and_date_time(self, data: tp.List[
        tp.Tuple[tp.Union[dt.datetime, pd.Timestamp], tp.Union[int, float], bool]]):
        i = 0
        while i < len(data):
            self.append(Peak(data[i][0], data[i][1], is_permanent=data[i][2]))
            i += 1
            if i >= len(data):
                break
            self.append(Trough(data[i][0], data[i][1], is_permanent=data[i][2]))
            i += 1

    def _from_num_list_with_date_time(self, data: tp.List[
        tp.Tuple[tp.Union[dt.datetime, pd.Timestamp], tp.Union[int, float]]]):

        i = 0
        while i < len(data):
            self.append(Peak(data[i][0], data[i][1]))
            i += 1
            if i >= len(data):
                break
            self.append(Trough(data[i][0], data[i][1]))
            i += 1

    def fill_from_list(self,
                       data: tp.List[
                           tp.Union[
                               int, float,
                               tp.Tuple[tp.Union[int, float], bool],
                               tp.Tuple[tp.Union[dt.datetime, pd.Timestamp], tp.Union[int, float], bool],
                               tp.Tuple[tp.Union[dt.datetime, pd.Timestamp], tp.Union[int, float]]]]) -> None:
        """
        Warning: This will delete all data
        :param data:
        :return:
        """
        if len(self) != 0:
            self.remove(direction=Direction.FORWARD, start_node=self._head_, end_node=self._tail_)
        if len(data) == 0:
            return None
        if isinstance(data[0], (int, float)):
            self._from_num_list(data)
        elif isinstance(data[0], tuple) and isinstance(data[0][0], (int, float)) and isinstance(data[0][1], bool):
            self._from_num_list_with_bool(data)
        elif isinstance(data[0], tuple) and isinstance(data[0][0], (dt.datetime, pd.Timestamp)) and \
                isinstance(data[0][1], (int, float)):
            self._from_num_list_with_date_time(data)
        elif isinstance(data[0], tuple) and isinstance(data[0][0], (dt.datetime, pd.Timestamp)) and \
                isinstance(data[0][2], bool):
            self._from_num_list_with_bool_and_date_time(data)
        else:
            raise Exception("Data format not recognized")

    def min_element(self, direction: str = Direction.FORWARD, start_node: tp.Optional[Node] = None,
                    end_node: tp.Optional[Node] = None):

        """
        Finds the minimum node in the market structure.
        It is start and end node inclusive

        :param direction:
        :param start_node:
        :param end_node:
        :return:
        """

        if direction not in [Direction.FORWARD, Direction.BACKWARD]:
            raise ValueError("direction must either be 'forward' or 'backward'")
        if (start_node and start_node.parent is not self) or (end_node and end_node.parent is not self):
            raise ValueError("Node is not in this market structure")
        if direction == Direction.FORWARD:
            start_node = self._head_ if start_node is None else start_node
            end_node = self._tail_ if end_node is None else end_node

            min_node = start_node
            current_node = start_node
            while current_node and current_node is not end_node.next:
                if current_node.value < min_node.value:
                    min_node = current_node
                current_node = current_node.next
            return min_node
        else:
            start_node = self._tail_ if start_node is None else start_node
            end_node = self._head_ if end_node is None else end_node

            min_node = start_node
            current_node = start_node
            while current_node and current_node is not end_node.prev:
                if current_node.value < min_node.value:
                    min_node = current_node
                current_node = current_node.prev
            return min_node

    def max_element(self, direction: str = Direction.FORWARD, start_node: tp.Optional[Node] = None,
                    end_node: tp.Optional[Node] = None):

        """
        Finds the maximum node in the market structure.
        It is start and end node inclusive
        :param direction:
        :param start_node:
        :param end_node:
        :return:
        """
        if direction not in [Direction.FORWARD, Direction.BACKWARD]:
            raise ValueError("direction must either be 'forward' or 'backward'")
        if (start_node and start_node.parent is not self) or (end_node and end_node.parent is not self):
            raise ValueError("Node is not in this market structure")
        if direction == Direction.FORWARD:
            start_node = self._head_ if start_node is None else start_node
            end_node = self._tail_ if end_node is None else end_node

            max_node = start_node
            current_node = start_node
            while current_node and current_node is not end_node.next:
                if current_node.value > max_node.value:
                    max_node = current_node
                current_node = current_node.next
            return max_node
        else:
            start_node = self._tail_ if start_node is None else start_node
            end_node = self._head_ if end_node is None else end_node

            max_node = start_node
            current_node = start_node
            while current_node and current_node is not end_node.prev:
                if current_node.value > max_node.value:
                    max_node = current_node
                current_node = current_node.prev
            return max_node

    def transform(self, transform_function: tp.Callable, direction: str = Direction.FORWARD,
                  start_node: tp.Optional[Node] = None,
                  end_node: tp.Optional[Node] = None):

        """
        Applies a transformation function to each node in the market structure.
        It is start and end node inclusive

        :param transform_function:
        :param direction:
        :param start_node:
        :param end_node:
        :return:
        """

        if direction not in [Direction.FORWARD, Direction.BACKWARD]:
            raise ValueError("direction must either be 'forward' or 'backward'")
        if (start_node and start_node.parent is not self) or (end_node and end_node.parent is not self):
            raise ValueError("Node is not in this market structure")
        if direction == Direction.FORWARD:
            start_node = self._head_ if start_node is None else start_node
            end_node = self._tail_ if end_node is None else end_node

            current_node = start_node
            while current_node and current_node is not end_node.next:
                transform_function(current_node)
                current_node = current_node.next
        else:
            start_node = self._tail_ if start_node is None else start_node
            end_node = self._head_ if end_node is None else end_node

            current_node = start_node
            while current_node and current_node is not end_node.prev:
                transform_function(current_node)
                current_node = current_node.prev

    def extend(self, other: MarketStructure, consume=False):
        """
        Extends the market structure with another market structure.

        :param other: The market structure to be appended.
        :param consume: If True, the nodes will be removed from the other market structure.
        """
        if len(other) == 0:
            return None
        for node in other:
            self.append(copy.copy(node))
        if consume:
            other.erase(other.head, other.tail)

    def erase(self, begin_node, end_node):
        """
        Erases a range of nodes from the market structure.
        Be careful when using this function as it does not check if the nodes are in the market structure.
        It is [begin_node, end_node] inclusive
        :param begin_node:
        :param end_node:
        :return:
        """

        node = begin_node
        while node and node is not end_node:
            next_node = node.next
            self._remove_node(node)
            node = next_node
        self._remove_node(node)
