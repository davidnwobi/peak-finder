import typing as tp
from collections.abc import Iterable
from .MKS import *
import copy
from abc import ABC, abstractmethod
import datetime as dt
from . import market_data_initializer as mdi
from .market_data_initializer import BadStartPoint, CMSNodeInfo
from collections import namedtuple

AnimationListTuple = namedtuple("AnimationListTuple", ["loc", "market_structure"])


def find_index_of_first_different_node_in_market_structure(market_structure1: MarketStructure,
                                                           market_structure2: MarketStructure) -> int:
    """
    Find the index of the first different node in two market structures.
    :param market_structure1:
    :param market_structure2:
    :return: index of the first different node in two market structures.
    """

    ms1 = market_structure1.head
    ms2 = market_structure2.head
    index = 0
    while ms1 and ms2:
        if ms1 is not ms2:
            return index
        ms1 = ms1.next
        ms2 = ms2.next
        index += 1
    return index


def find_node_and_reset_observer_state(mdi: mdi.MarketDataInitializer) -> tp.Optional[Node]:
    """
    Find the node that was changed and reset the observer state of all nodes in the market structure.
    :param mdi:
    :return: The node that was changed or None if no node was changed.
    """
    i = 0
    node = mdi.market_structure.changed_nodes[i]

    count = 0
    while i < len(mdi.market_structure.changed_nodes) and node.node.parent is not mdi.market_structure:
        node = mdi.market_structure.changed_nodes[i]
        node.node.reset_observer_state()
        i += 1

    while i < len(mdi.market_structure.changed_nodes):
        cur_node = mdi.market_structure.changed_nodes[i]
        if cur_node < node and cur_node.node.parent is mdi.market_structure:
            node = cur_node
        cur_node.node.reset_observer_state()
        count += 1

        i += 1

    if node and node.node.parent is mdi.market_structure:
        return node.node
    else:
        return None


def better_store_difference(mdi: mdi.MarketDataInitializer):
    """
    Store the difference between the current market structure and the previous market structure.

    :param mdi:
    :return:
    """

    if len(mdi.market_structure.changed_nodes) == 0:
        return
    curr = find_node_and_reset_observer_state(mdi)
    slicing_node = curr
    if curr is None:
        return

    i = 0
    while curr is not mdi.market_structure.tail:
        curr = curr.next
        i += 1
    index_curr = len(mdi.market_structure) - 1 - i

    mdi.animation_list.append(AnimationListTuple(loc=index_curr, market_structure=mdi.market_structure.slice_list(
        start_node=slicing_node, inplace=False)))

    mdi.market_structure.changed_nodes = []


def save_state_in_animation_list(func):
    """
    Decorator to save the state of the market structure in the animation list before and after a function is called.

    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            self_mdi = getattr(self, "mdi")
            better_store_difference(self_mdi)
            func_out = func(self, *args, **kwargs)
            better_store_difference(self_mdi)
            return func_out
        except Exception as e:
            raise

    return wrapper


class MRP(ABC):
    """
    Initialize the MRP (Market Reversal Pattern) object.

    :param market_data_initializer: Instance of MarketDataInitializer.
    """

    def __init__(self, market_data_initializer: mdi.MarketDataInitializer):
        self.mdi: mdi.MarketDataInitializer = market_data_initializer

    def join_disconnect_left(self, left, mid):
        """
        Join nodes and disconnect the range on the left side of the midpoint.

        :param left: The left node of the range to be joined and disconnected.
        :param mid: The midpoint node.
        """

        self.mdi.market_structure.remove(direction="forward", start_node=left.next, end_node=mid.prev)

    def join_disconnect_right(self, mid, right):
        """
        Join nodes and disconnect the range on the right side of the midpoint.

        :param mid: The midpoint node.
        :param right: The right node of the range to be joined and disconnected.
        """

        self.mdi.market_structure.remove(direction="forward", start_node=mid.next, end_node=right.prev)

    def join_disconnect_both_sides(self, left, mid, right):
        """
        Join nodes and disconnect the range on both sides of the midpoint.

        :param left: The left node of the range to be joined and disconnected.
        :param mid: The midpoint node.
        :param right: The right node of the range to be joined and disconnected.
        """
        self.join_disconnect_left(left, mid)
        self.join_disconnect_right(mid, right)

    def clean_between_high(self, high1: Peak, high2: Peak):
        """
        Find the lowest trough between two peaks and clean the nodes between them.

        :param high1: The first peak.
        :param high2: The second peak.
        :return: True if a CMS node was removed and not replaced by the lowest trough, False otherwise.
        """
        # finds the lowest trough between two peaks
        # deletes all nodes between the two peaks except the lowest trough

        # find the lowest trough between the two peaks
        cms = self.mdi.market_structure.find(lambda x: x.was_cms, start_node=high1.next, end_node=high2.prev)
        cms_was_removed = True if cms is not None else False
        lowest_trough: tp.Optional[Node] = self.mdi.market_structure.min_element("backward", high2.prev, high1.next)

        if lowest_trough is None:
            raise Exception("No trough found between the two peaks. This should not happen.")
        else:
            # delete all nodes between the two peaks except the lowest trough
            if high1.next is lowest_trough:
                if high2 is lowest_trough.next:
                    return False
                else:
                    self.join_disconnect_right(lowest_trough, high2)
            else:
                if high2 is lowest_trough.next:
                    self.join_disconnect_right(high1, lowest_trough)
                else:
                    self.join_disconnect_both_sides(high1, lowest_trough, high2)
        return cms_was_removed and (cms is not lowest_trough if cms is not None else False)

    def clean_between_low(self, low1: Trough, low2: Trough):
        """
        Find the highest peak between two troughs and clean the nodes between them.

        :param low1: The first trough.
        :param low2: The second trough.
        :param market_structure: The market structure containing nodes.
        :return: True if a CMS node was removed and not replaced by the highest peak, False otherwise.
        """
        # finds the highest peak between two troughs
        # deletes all nodes between the two troughs except the highest peak

        # find the highest peak between the two troughs
        cms = self.mdi.market_structure.find(lambda x: x.was_cms, start_node=low1.next, end_node=low2.prev)
        cms_was_removed = True if cms is not None else False
        highest_peak: tp.Optional[Node] = self.mdi.market_structure.max_element("backward", low2.prev,
                                                                                low1.next)  # pick the latest highest peak
        if highest_peak is None:
            raise RuntimeError("No peak found between the two troughs. This should not happen.")
        else:
            # delete all nodes between the two troughs except the highest peak
            if low1.next is highest_peak:
                if low2 is highest_peak.next:
                    return False
                else:
                    self.join_disconnect_right(highest_peak, low2)
            else:
                if low2 is highest_peak.next:
                    self.join_disconnect_left(low1, highest_peak)
                else:
                    self.join_disconnect_both_sides(low1, highest_peak, low2)
        return cms_was_removed and (cms is not highest_peak if cms is not None else False)

    # test functions

    def clean_between_points(self, point1: tp.Optional[Node], point2: tp.Optional[Node]):
        """
        Clean the nodes between two points in the market structure.
        Cleaning means deleting all nodes between the two points except the lowest trough or highest peak.
        :param point1: The first point.
        :param point2: The second point.
        :param market_structure: The market structure containing nodes.
        :return: True if a CMS node was removed and not replaced by the cleaned nodes, False otherwise.
        """
        if point1 is None or point2 is None:
            return False
        if point1.next is not point1:
            if isinstance(point1, Peak) and isinstance(point2, Peak):
                return self.clean_between_high(point1, point2)
            elif isinstance(point1, Trough) and isinstance(point2, Trough):
                return self.clean_between_low(point1, point2)
            else:
                # if the two points are not of the same type, join them
                cms = self.mdi.market_structure.find(lambda x: x.was_cms, start_node=point1.next, end_node=point2.prev)
                cms_was_removed = True if cms is not None else False

                self.mdi.market_structure.remove(direction="forward", start_node=point1.next, end_node=point2.prev)

                return cms_was_removed

    def find_last_broken_low(self, low: Trough) -> tp.Optional[Node]:
        """
        Find the last low that was broken by the given trough.

        :param low: The trough node.
        :param market_structure: The market structure containing nodes.
        :return: The last low that was broken by the given trough, or None if no such low exists.
        """

        last_checked_low: tp.Optional[Node] = low.prev.prev if low.prev else None

        while last_checked_low and last_checked_low.prev and last_checked_low.value > low.value:
            last_checked_low = last_checked_low.prev.prev
            low.broke = True  # This lets us know that this low broke another low even if it returns None
            # TODO: This is never used. An error is just thrown if this is None
            child = None
            child_high = None
            grandchild = None
            grandchild_high = None

            if last_checked_low:
                child = last_checked_low.next.next
                child_high = last_checked_low.next.next.next
                grandchild = last_checked_low.next.next.next.next
                grandchild_high = last_checked_low.next.next.next.next.next

            required_nodes_exist = last_checked_low and grandchild_high
            if required_nodes_exist:
                current_low_broke_the_child = child.value > low.value
                child_is_near_cms = child_high.was_cms or grandchild_high.was_cms
                market_reversed_on_child = last_checked_low.value >= child.value and child.value < grandchild.value
                if current_low_broke_the_child and child_is_near_cms and market_reversed_on_child:
                    break

        if last_checked_low is None:
            return None
        elif last_checked_low.next.next is low:
            # Didn't enter the while loop
            if last_checked_low.value >= low.value:
                low.broke = True
                return last_checked_low
            else:
                return None
        else:
            return last_checked_low.next.next

    def find_last_broken_high(self, high: Peak) -> tp.Optional[Node]:
        """
        Find the last (earliest) high that was broken by the given peak.

        :param high: The peak node.
        :return: The last high that was broken by the given peak or None if no such high exists.
        """
        last_checked_high: tp.Optional[Node] = high.prev.prev if high.prev else None
        while last_checked_high and last_checked_high.prev and last_checked_high.value < high.value:
            last_checked_high = last_checked_high.prev.prev
            high.broke = True

            child = None
            child_low = None
            grandchild = None
            grandchild_low = None
            if last_checked_high:
                child = last_checked_high.next.next
                child_low = last_checked_high.next.next.next
                grandchild = last_checked_high.next.next.next.next
                grandchild_low = last_checked_high.next.next.next.next.next

            required_nodes_exist = last_checked_high and grandchild_low
            if required_nodes_exist:
                current_peak_broke_the_child = child.value < high.value
                child_is_near_cms = child_low.was_cms or grandchild_low.was_cms
                market_reversed_on_child = last_checked_high.value <= child.value and child.value > grandchild.value
                if current_peak_broke_the_child and child_is_near_cms and market_reversed_on_child:
                    break

        if last_checked_high is None:
            return None
        elif last_checked_high.next.next is high:
            # Didn't enter the while loop
            if last_checked_high.value <= high.value:
                high.broke = True
                return last_checked_high
            else:
                return None
        else:
            return last_checked_high.next.next

    def find_last_permanent_high(self, high: Peak) -> tp.Optional[Peak]:
        """
        Find the last permanent high before the given peak.

        :param high: The peak node.
        :return: The last permanent high before the given peak, or None if no such high exists.
        """
        return self.mdi.market_structure.find(lambda x: x.is_permanent and (isinstance(x, Peak)), direction="backward",
                                              start_node=high.prev)

    def find_last_permanent_low(self, low: Trough) -> tp.Optional[Trough]:
        """
        Find the last permanent low before the given trough.

        :param low: The trough node.
        :return: The last permanent low before the given trough, or None if no such low exists.
        """
        return self.mdi.market_structure.find(lambda x: x.is_permanent and (isinstance(x, Trough)),
                                              direction="backward",
                                              start_node=low.prev)

    @staticmethod
    def broke_a_low(low: Trough) -> bool:
        """
        Determine if at least one low was broken.

        :param low: The trough node.
        :return: True if at least one low was broken, False otherwise.
        """
        # Determines if at least one low was broken
        if low is None:
            return False

        last_checked_low: tp.Optional[Node] = low.prev.prev if low.prev else None
        if last_checked_low is None:
            return False
        if last_checked_low.value >= low.value:
            low.broke = True
            return True

    @staticmethod
    def broke_a_high(high: Peak) -> bool:
        """
        Determine if at least one high was broken.

        :param high: The peak node.
        :return: True if at least one high was broken, False otherwise.
        """
        if high is None:
            return False
        last_checked_high: tp.Optional[Node] = high.prev.prev if high.prev else None
        if last_checked_high is None:
            return False
        if last_checked_high.value <= high.value:
            high.broke = True
            return True

    def broke_permanent_low(self, low: Trough):
        """
        Check if a permanent low has been broken.
        This function should only be called if a low has been broken. Undefined behavior otherwise.
        :param low: The trough node.
        :return: True if a permanent low has been broken, False otherwise.
        """
        # we know a low has been broken so no need to check if low is none
        last_permanent_low = self.mdi.market_structure.find(lambda x: isinstance(x, Trough) and x.is_permanent,
                                                            direction="backward",
                                                            start_node=low.prev)
        if last_permanent_low is None:
            return False
        if last_permanent_low.value < low.value:
            return False
        return True

    def broke_permanent_high(self, high: Peak):
        """
        Check if a permanent high has been broken.
        This function should only be called if a high has been broken. Undefined behavior otherwise.
        :param high: The peak node.
        :return: True if a permanent high has been broken, False otherwise.
        """
        last_permanent_high = self.mdi.market_structure.find(lambda x: isinstance(x, Peak) and x.is_permanent,
                                                             direction="backward",
                                                             start_node=high.prev)
        if last_permanent_high is None:
            return False
        if last_permanent_high.value > high.value:
            return False
        return True

    def verify_cms_in_market(self, cms: Node):
        """
        Verify if a CMS (Change in Market Structure) node exists within the market structure.

        :param cms: The CMS node to verify.
        :return: True if the CMS node is found within the market structure, False otherwise.
        """
        is_found = self.mdi.market_structure.find(lambda x: x is cms, direction="backward")
        return is_found is not None

    def find_removed_cms(self) -> \
            tp.Tuple[
                Node, str]:
        """
        Find the removed CMS (Change in Market Structure) node from a list of CMS nodes.
        :return: Tuple containing the removed CMS node and its associated direction.
        """
        last_removed_cms = None
        for node in reversed(self.mdi.cms_list):
            if not self.verify_cms_in_market(node.concrete_node):
                return node

    @abstractmethod
    def validate_high(self, current_high: Peak):
        raise NotImplementedError("Not Implemented")

    @abstractmethod
    def validate_low(self, current_low: Trough):
        raise NotImplementedError("Not Implemented")

    @abstractmethod
    def calculate_market_structure(self):
        raise NotImplementedError("Not Implemented")


class CmsMrp(MRP):
    """
    Initialize the CMS (Change in Market Structure) MRP (Market Reversal Pattern) object.
    This pattern reverse the market structure when the market breaks two lows in the reverse direction.

    """

    def __init__(self, market_data_initializer: mdi.MarketDataInitializer):
        """
        Initialize the CMS (Change in Market Structure) MRP (Market Reversal Pattern) object.
        :param market_data_initializer: Instance of MarketDataInitializer.
        """
        super().__init__(market_data_initializer)

    def complete_cms_search_high_downtrend(self, current_high: Peak, last_permanent_high: Peak) -> bool:
        """
        Complete the CMS (Change in Market Structure) search for a downtrend.
        :param current_high: The current peak node.
        :param last_permanent_high: The last permanent peak node.
        :return: True if a CMS node was removed while removing temporary nodes, False otherwise.
        """
        self.mdi.trend = "uptrend"
        current_high.is_permanent = True
        cms_was_removed = self.clean_between_points(last_permanent_high,
                                                    current_high)

        last_permanent_high.is_cms = False
        self.reset_to_normal_high(current_high)

        last_permanent_high.prev.is_permanent = True  # set trough before peak1(the peak that needed to be broken
        # to confirms cms) to True

        last_permanent_high.was_cms = True

        return cms_was_removed

    def start_cms_search_high_downtrend(self, current_high: Peak, last_permanent_high: Peak) -> bool:
        """
        Start the CMS (Change in Market Structure) search for a downtrend.
        :param current_high: The current peak node.
        :param last_permanent_high: The last permanent peak node.
        :return: True if a CMS node was removed while removing temporary nodes, False otherwise.
        """
        last_broken_high = self.find_last_broken_high(current_high)
        if last_broken_high is None:
            raise BadStartPoint
        cms_was_removed = self.clean_between_points(last_permanent_high,
                                                    current_high)
        last_broken_high = self.find_last_broken_high(current_high)
        if last_broken_high is None:
            raise BadStartPoint
        if last_broken_high.prev and last_broken_high.prev and last_broken_high.prev.prev.value <= \
                last_broken_high.value:
            last_broken_high = last_broken_high.next.next

        def process_node(node):
            node.is_permanent = False

        self.mdi.market_structure.transform(lambda x: process_node(x) if isinstance(x, Peak) else None,
                                            start_node=last_broken_high,
                                            end_node=current_high.prev)
        current_high.COMMENT = "POSSIBLE TREND CHANGE; " + current_high.COMMENT
        current_high.is_cms = True
        last_broken_high.is_permanent = False

        return cms_was_removed

    def handle_broken_permanent_high_downtrend(self, current_high: Peak) -> bool:
        """
        Handle a broken permanent high in a downtrend.
        :param current_high: The current peak node.
        :return: True if a CMS node was removed while removing temporary nodes, False otherwise.
        """

        last_permanent_high = self.find_last_permanent_high(current_high)
        if last_permanent_high.is_cms:
            return self.complete_cms_search_high_downtrend(current_high, last_permanent_high)
        else:
            return self.start_cms_search_high_downtrend(current_high, last_permanent_high)

    def handle_broken_temp_high_downtrend(self, current_high: Peak) -> bool:
        """
        Handle a broken temporary high in a downtrend.
        :param current_high: The current peak node.
        :return: True if a CMS node was removed while removing temporary nodes, False otherwise.
        """
        last_permanent_high = self.find_last_permanent_high(
            current_high) if current_high.prev is not None else None
        current_high.is_permanent = False
        return self.clean_between_points(last_permanent_high, current_high)

    def handle_broken_high_downtrend(self, current_high: Peak) -> bool:
        if self.broke_permanent_high(current_high):
            return self.handle_broken_permanent_high_downtrend(current_high)
        else:
            return self.handle_broken_temp_high_downtrend(current_high)

    def handle_broken_permanent_high_uptrend(self, current_high: Peak) -> bool:
        last_permanent_high = self.find_last_permanent_high(current_high)
        current_high.is_permanent = True
        cms_was_removed = self.clean_between_points(last_permanent_high, current_high)
        prev_last_permanent_low = self.find_last_permanent_low(current_high.prev)
        if prev_last_permanent_low is not None:
            cms_was_removed = cms_was_removed or self.clean_between_points(prev_last_permanent_low,
                                                                           current_high.prev)
        # Clear this up so this it is not triggerd in the future
        current_high.prev.is_cms = False
        return cms_was_removed

    def handle_invalidated_cms_search_high_uptrend(self, current_high: Peak) -> bool:
        cms_was_removed = False
        last_permanent_high = self.find_last_permanent_high(current_high)

        if last_permanent_high is not None:
            cms_was_removed = cms_was_removed or self.clean_between_points(last_permanent_high,
                                                                           current_high)

        current_high.prev.is_cms = False
        nodes_to_recheck = self.mdi.market_structure.slice_list(current_high.prev, inplace=True)
        # This is dangerous. It can cause an infinite loop. I need to find a way to fix this. 🤞🤞
        temp_cms_removed, _ = self.abnormal_trend_for_cms_validation(nodes_to_recheck)
        cms_was_removed = cms_was_removed or temp_cms_removed
        while len(nodes_to_recheck) > 0:
            # flow_logger.info("Looping through nodes to recheck in validate_high")
            node = nodes_to_recheck.popfront()
            self.mdi.market_structure.append(node)
        return cms_was_removed

    def handle_potential_invalidated_cms_search_high_uptrend(self, current_high: Peak) -> bool:
        """
        Handle a potential invalidated CMS (Change in Market Structure) search for an uptrend.
        :param current_high: The current peak node.
        :return: True if a CMS node was removed while removing temporary nodes, False otherwise.

        """
        last_permanent_low = self.find_last_permanent_low(
            current_high.prev) if current_high.prev is not None else None
        cms_was_removed = False

        if last_permanent_low is None:
            return cms_was_removed
        if last_permanent_low.prev.value <= current_high.value:  # which high got broken? Before or after the cms?
            return self.handle_invalidated_cms_search_high_uptrend(current_high)  # before

        # We maximize separation by keeping the highest high
        max_element = self.mdi.market_structure.max_element("backward", current_high, last_permanent_low)
        if max_element is current_high:
            return self.clean_between_points(last_permanent_low, current_high)
        else:
            self.mdi.market_structure.erase(begin_node=max_element.next,
                                            end_node=self.mdi.market_structure.tail)
            return False

    def handle_broken_temp_high_uptrend(self, current_high: Peak) -> bool:
        cms_was_removed = False
        if not current_high.prev.is_cms:
            cms_was_removed = cms_was_removed or self.handle_potential_invalidated_cms_search_high_uptrend(current_high)

        else:  # then we are assured that the temporary high is between cms and last permanent high since we
            # know a temp high was broken
            last_permanent_high = self.find_last_permanent_high(current_high)
            if last_permanent_high is not None:
                cms_was_removed = cms_was_removed or self.clean_between_points(last_permanent_high, current_high)

            current_high.prev.is_cms = False
        current_high.is_permanent = False
        return cms_was_removed

    def handle_broken_high_uptrend(self, current_high: Peak) -> bool:
        if self.broke_permanent_high(current_high):
            return self.handle_broken_permanent_high_uptrend(current_high)
        else:
            return self.handle_broken_temp_high_uptrend(current_high)

    def handle_validate_high_uptrend(self, current_high: Peak) -> bool:
        cms_was_removed = False
        if self.broke_a_high(current_high):
            return self.handle_broken_high_uptrend(current_high)
        else:
            current_high.is_permanent = False
        return cms_was_removed

    def handle_validate_high_downtrend(self, current_high: Peak) -> bool:
        cms_was_removed = False
        if self.broke_a_high(current_high):
            return self.handle_broken_high_downtrend(current_high)
        else:
            if current_high.prev is not None and current_high.prev.is_permanent is False:
                current_high.is_permanent = False
        return cms_was_removed

    @log_function_call
    @save_state_in_animation_list
    def validate_high(self, current_high: Peak):
        """
            Validate a high point in the market structure based on the given trend direction.

            This function ensures the validity and consistency of the market structure in accordance with the specified trend.
            A valid trend is one in which the highest local high is joined to the lowest local low, creating a visually
            recognizable pattern. This function programmatically enforces this principle.

            :param market_structure: The market structure containing peaks and troughs.
            :param current_high: The high point to be validated.

            * Uptrend Validation Logic:
            In an uptrend, the goal is to maintain a sequence of higher highs and higher lows. To validate a high point:
                1. Check if the current high has broken any previous highs.
                2. If the current high broke a previous high, check if the previous high was a permanent high.
                    i. Permanent high:
                       a. To maintain a consistent market structure, remove the nodes between the previous permanent high and
                          the current high, except for the permanent trough between them.
                       b. If the last permanent low fell below previous permanent lows, remove them to ensure a higher-high,
                          higher-low structure.
                       c. Clear the CMS flag on the previous high if it's no longer valid.
                    ii. Not a permanent high:
                        Reduce complexity by retaining only the highest high in the structure as a temporary high.

                3. If the current high didn't break a previous high, mark it as a temporary high for potential future removal.

            * DownTrend Validation Logic:
            In a downtrend, high points are critical for determining trend changes. The process is more complex:
                1. Check if the current high has broken any previous highs.
                2. If the current high broke a previous high, check if the previous high was a permanent high.
                    i. Permanent high:
                        a. If the previous high was not a CMS (Confirmation of Market Structure change), there are two scenarios we
                        may encounter in the future:
                           I. CMS confirmation: Mark the current high as a potential CMS.
                           II.  CMS Invalidated: Mark highs that this high broke as temporary highs.
                        b. If the previous high was a CMS, the trend changes from downtrend to uptrend. Remove temporary highs,
                           leaving only the lowest low between the previous CMS and the current high. Set relevant flags.
                    ii. Not a permanent high:
                        Retain only the highest high as a temporary high, as in the uptrend.

                3. If the current high didn't break a previous high, additional logic is present that requires further evaluation.

            Note: This function modifies the market_structure and trend_direction in place. It does not return any value.
            The removal of a CMS invalidates the trend and is handled elsewhere.
            """

        if self.mdi.trend == "uptrend":

            return self.handle_validate_high_uptrend(current_high)
        else:
            return self.handle_validate_high_downtrend(current_high)

    def complete_cms_search_low_uptrend(self, current_low: Trough, last_permanent_low: Trough) -> bool:
        self.mdi.trend = "downtrend"
        current_low.is_permanent = True
        cms_was_removed = self.clean_between_points(last_permanent_low, current_low)
        last_permanent_low.is_cms = False

        self.reset_to_normal_low(current_low)

        last_permanent_low.prev.is_permanent = True  # set peak before trough2(the trough that needed to
        # be broken
        # to confirms cms) to True. It will only be one
        last_permanent_low.was_cms = True

        return cms_was_removed

    def start_cms_search_low_uptrend(self, current_low: Trough, last_permanent_low: Trough) -> bool:
        last_broken_low = self.find_last_broken_low(current_low)
        if last_broken_low is None:
            raise BadStartPoint

        # same assurance we give that when important events like these occur. We get rid of temp nodes
        # to keep bipartite-ness
        cms_was_removed = self.clean_between_points(last_permanent_low, current_low)

        last_broken_low = self.find_last_broken_low(current_low)
        if last_broken_low is None:
            raise BadStartPoint
        if last_broken_low.prev and last_broken_low.prev and last_broken_low.prev.prev.value >= \
                last_broken_low.value:
            last_broken_low = last_broken_low.next.next

        def process_node(node):
            node.is_permanent = False

        try:
            self.mdi.market_structure.transform(
                lambda x: process_node(x) if isinstance(x, Trough) else None,
                start_node=last_broken_low,
                end_node=current_low.prev)  # this lets us have special markers because peaks will be
        except:
            self.mdi.market_structure.transform(
                lambda x: process_node(x) if isinstance(x, Trough) else None,
                start_node=last_broken_low,
                end_node=current_low.prev)  # this lets us have special markers because peaks will be
        # permanent while troughs  will be temporary

        current_low.COMMENT = "POSSIBLE TREND CHANGE; " + current_low.COMMENT

        current_low.is_cms = True
        last_broken_low.is_permanent = False
        return cms_was_removed

    def handle_broken_permanent_low_uptrend(self, current_low: Trough) -> bool:
        last_permanent_low = self.find_last_permanent_low(current_low)
        if last_permanent_low.is_cms:
            return self.complete_cms_search_low_uptrend(current_low, last_permanent_low)
        else:
            return self.start_cms_search_low_uptrend(current_low, last_permanent_low)

    def handle_broken_temp_low_uptrend(self, current_low: Trough) -> bool:
        last_permanent_low = self.find_last_permanent_low(
            current_low) if current_low.prev is not None else None
        current_low.is_permanent = False
        return self.clean_between_points(last_permanent_low, current_low)

    def handle_broken_low_uptrend(self, current_low: Trough) -> bool:
        if self.broke_permanent_low(current_low):
            return self.handle_broken_permanent_low_uptrend(current_low)
        else:
            return self.handle_broken_temp_low_uptrend(current_low)

    def handle_broken_permanent_low_downtrend(self, current_low: Trough) -> bool:
        cms_was_removed = False
        last_permanent_low = self.find_last_permanent_low(current_low)
        current_low.is_permanent = True
        cms_was_removed = cms_was_removed or self.clean_between_points(last_permanent_low, current_low)
        prev_last_permanent_high = self.find_last_permanent_high(current_low.prev)
        if prev_last_permanent_high is not None:
            cms_was_removed = cms_was_removed or self.clean_between_points(prev_last_permanent_high,
                                                                           current_low.prev)

        # Clear this up so this it is not triggerd in the future
        current_low.prev.is_cms = False
        return cms_was_removed

    def handle_invalidated_cms_search_low_downtrend(self, current_low: Trough) -> bool:
        cms_was_removed = False
        last_permanent_low = self.find_last_permanent_low(current_low)
        if last_permanent_low is not None:
            cms_was_removed = cms_was_removed or self.clean_between_points(last_permanent_low,
                                                                           current_low)

            self.reset_to_normal_high(current_low)
            current_low.prev.is_cms = False
            nodes_to_recheck = self.mdi.market_structure.slice_list(current_low.prev, inplace=True)
            # This is dangerous. It can cause an infinite loop. I need to find a way to fix
            # this. 🤞🤞

            temp_cms_removed, _ = self.abnormal_trend_for_cms_validation(nodes_to_recheck)

            cms_was_removed = cms_was_removed or temp_cms_removed  # Send message to the resolve_cms
            if len(nodes_to_recheck) > 0:
                # flow_logger.info("Looping through nodes to recheck in validate_low")
                self.mdi.market_structure.extend(other=nodes_to_recheck)

        return cms_was_removed

    def handle_potential_invalidated_cms_search_low_downtrend(self, current_low: Trough) -> bool:
        last_permanent_high = self.find_last_permanent_high(
            current_low.prev) if current_low.prev is not None else None

        if last_permanent_high is not None:
            if last_permanent_high.prev.value < current_low.value:  # which low got broken?
                min_element = self.mdi.market_structure.min_element("backward", current_low, last_permanent_high)

                if min_element is not current_low:  # We can optimize this keeping track of the max element

                    self.mdi.market_structure.erase(begin_node=min_element.next,
                                                    end_node=self.mdi.market_structure.tail)

                    return False
                else:
                    return self.clean_between_points(last_permanent_high, current_low)

            else:
                return self.handle_invalidated_cms_search_low_downtrend(current_low)

    def handle_broken_temp_low_downtrend(self, current_low: Trough) -> bool:
        cms_was_removed = False
        if not current_low.prev.is_cms:
            cms_was_removed = cms_was_removed or self.handle_potential_invalidated_cms_search_low_downtrend(
                current_low)

        else:  # then we are assured that the temporary low is between cms and last permanent low since we
            # know a temp low was broken
            last_permanent_low = self.find_last_permanent_low(current_low)
            if last_permanent_low is not None:
                cms_was_removed = cms_was_removed or self.clean_between_points(last_permanent_low,
                                                                               current_low)

            current_low.prev.is_cms = False
        current_low.is_permanent = False
        return cms_was_removed

    def handle_broken_low_downtrend(self, current_low: Trough) -> bool:
        if self.broke_permanent_low(current_low):
            return self.handle_broken_permanent_low_downtrend(current_low)
        else:
            return self.handle_broken_temp_low_downtrend(current_low)

    def handle_validate_low_downtrend(self, current_low: Trough) -> bool:
        cms_was_removed = False
        if self.broke_a_low(current_low):
            return self.handle_broken_low_downtrend(current_low)
        else:
            current_low.is_permanent = False
        return cms_was_removed

    def handle_validate_low_uptrend(self, current_low: Trough) -> bool:
        cms_was_removed = False
        if self.broke_a_low(current_low):
            return self.handle_broken_low_uptrend(current_low)
        else:
            if current_low.prev is not None and current_low.prev.is_permanent is False:
                current_low.is_permanent = False
        return cms_was_removed

    @log_function_call
    @save_state_in_animation_list
    def validate_low(self, current_low: Trough):
        """
            Validate a low point in the market structure based on the given trend direction.

            This function is used to validate a low point in the market structure based on the specified trend direction.
            The goal is to ensure that the market structure is consistent with the specified trend. A valid trend is one
            in which the lowest local low is joined to the highest local high. This function performs the necessary checks
            and modifications to maintain a valid market structure.

            :param market_structure: The market structure containing peaks and troughs.
            :param current_low: The low point to be validated.
            :param self.mdi.trend: The current trend direction ("uptrend" or "downtrend").

            * Uptrend Validation Logic:
                The logic for uptrend validation is the inverse of downtrend validation. Since lows and highs are inverse in
                their roles, the checks and operations performed mirror those in the `validate_high` function.

            *DownTrend Validation Logic:
                The downtrend validation logic is very similar to the `validate_high` function, with roles reversed for lows
                and highs. This function ensures that the market structure is consistent with a downtrend.

            Note: The function does not return anything, as it modifies the market_structure and self.mdi.trend in place.
            Note that whether a cms was removed is being tracked. If this happens, it invalidates the trend from where the cms
            was removed, and this is handled elsewhere.
        """
        if self.mdi.trend == "uptrend":
            return self.handle_validate_low_uptrend(current_low)

        else:
            return self.handle_validate_low_downtrend(current_low)

    def reset_to_normal_low(self, start_node):
        def process_node(node):
            node.is_permanent = True
            node.next.is_permanent = True

        if start_node.prev:
            start_point = self.mdi.market_structure.find(
                lambda x: x.is_permanent and isinstance(x, Trough),
                direction="backward",
                start_node=start_node.prev)
            if start_point is not None:
                end_permanent_low = self.mdi.market_structure.find(
                    lambda x: x.is_permanent and isinstance(x, Trough),
                    direction="backward",
                    start_node=start_point.prev)
                if end_permanent_low is not None:
                    self.mdi.market_structure.transform(
                        lambda x: process_node(x) if isinstance(x, Trough) else None,
                        direction="backward",
                        start_node=start_point,
                        end_node=end_permanent_low)

            start_node.prev.is_permanent = True  # set peak before trough2(the trough that confirms cms) to True

    def reset_to_normal_high(self, start_node):
        def process_node(node):
            node.is_permanent = True
            node.next.is_permanent = True

        if start_node.prev:
            start_point = self.mdi.market_structure.find(
                lambda x: x.is_permanent and isinstance(x, Peak),
                direction="backward",
                start_node=start_node.prev)
            if start_point is not None:
                end_permanent_high = self.mdi.market_structure.find(
                    lambda x: x.is_permanent and isinstance(x, Peak),
                    direction="backward",
                    start_node=start_point.prev)
                if end_permanent_high is not None:
                    self.mdi.market_structure.transform(lambda x: process_node(x) if isinstance(x, Peak) else None,
                                                        direction="backward",
                                                        start_node=start_point,
                                                        end_node=end_permanent_high)

            start_node.prev.is_permanent = True  # set trough before peak2(the peak that confirms cms) to True

    @log_function_call
    def resolve_cms(self) -> tp.Tuple[bool, tp.Optional[Node]]:
        try:
            removed_cms = self.find_removed_cms()
            if len(self.mdi.cms_list) <= 1:
                raise BadStartPoint

            last_valid_cms_index = self.mdi.cms_list.index(removed_cms)
            while last_valid_cms_index >= 0 and self.mdi.cms_list[
                last_valid_cms_index].concrete_node.parent is not self.mdi.market_structure:
                last_valid_cms_index -= 1

            if last_valid_cms_index < 0:  # if less than zero, ensure you change the direction of the trend
                raise BadStartPoint
            del self.mdi.cms_list[last_valid_cms_index + 1:]
            start_point = self.mdi.cms_list[last_valid_cms_index].concrete_node
            self.mdi.trend = self.mdi.cms_list[last_valid_cms_index].direction
            last_valid_node = start_point.next
            last_valid_node.is_permanent = True
            last_valid_node = start_point.next
            last_valid_node.is_cms = False
            last_valid_node.was_cms = False
            del self.mdi.cms_list[last_valid_cms_index + 1:]
            points_for_validation = self.mdi.market_structure.slice_list(last_valid_node.next, inplace=True)

            cms_removed, next_node = self.abnormal_trend_for_cms_validation(points_for_validation)
            return cms_removed, next_node
        except Exception as e:
            raise

    @log_function_call
    def normal_trend(self) -> tp.Tuple[bool, tp.Optional[Node]]:
        cms_removed = False
        try:
            if isinstance(self.mdi.market_structure.tail, Trough):
                next_node = self.mdi.get_next_element_in_gen(self.mdi.high_generator)
            else:
                next_node = self.mdi.get_next_element_in_gen(self.mdi.low_generator)
            while next_node is not None and not cms_removed:
                if isinstance(self.mdi.market_structure.tail, Trough):
                    high = next_node
                    self.mdi.market_structure.append(high)
                    cms_removed = self.validate_high(high)
                    if cms_removed:
                        high.COMMENT = high.COMMENT + ";ASSUMED TREND INVALIDATED"
                    else:
                        high.COMMENT = high.COMMENT + ";ASSUMED TREND OK"
                    high.trend = self.mdi.trend
                    # flow_logger.info(f"Market Structure: {self.mdi.market_structure}, Cms Removed: {cms_removed}")
                    if cms_removed:
                        break
                    else:
                        if self.mdi.trend == "uptrend" and high.prev and high.prev.prev and high.prev.prev.was_cms:
                            self.mdi.cms_list.append(CMSNodeInfo(high.prev.prev, "uptrend"))
                    next_node = self.mdi.get_next_element_in_gen(self.mdi.low_generator)
                if isinstance(self.mdi.market_structure.tail, Peak):
                    if next_node is None:
                        break
                    low = next_node
                    self.mdi.market_structure.append(low)
                    cms_removed = self.validate_low(low)
                    if cms_removed:
                        low.COMMENT = low.COMMENT + " ;ASSUMED TREND INVALIDATED"
                    else:
                        low.COMMENT = low.COMMENT + " ;ASSUMED TREND OK"
                    low.trend = self.mdi.trend
                    # flow_logger.info(f"Market Structure: {self.mdi.market_structure}, Cms Removed: {cms_removed}")
                    if cms_removed:

                        break
                    else:
                        if self.mdi.trend == "downtrend" and low.prev and low.prev.prev and low.prev.prev.was_cms:
                            self.mdi.cms_list.append(CMSNodeInfo(low.prev.prev, "downtrend"))
                next_node = self.mdi.get_next_element_in_gen(self.mdi.high_generator)
            return cms_removed, next_node
        except Exception as e:
            raise

    @log_function_call
    @save_state_in_animation_list
    def abnormal_trend_for_cms_validation(self, data_points: MarketStructure) -> tp.Tuple[bool, tp.Optional[Node]]:
        try:
            cms_removed = False
            while len(data_points) > 0:
                if isinstance(self.mdi.market_structure.tail, Trough):
                    high = data_points.popfront()
                    # reset the node
                    high.is_permanent = True
                    high.is_cms = False
                    high.was_cms = False
                    high.COMMENT = ""
                    high.reset_observer_state()  # reset the observer state
                    self.mdi.market_structure.append(high)
                    high.COMMENT = "RECALCULATING; "
                    cms_removed = self.validate_high(high)

                    if cms_removed:
                        high.COMMENT = high.COMMENT + ";ASSUMED TREND INVALIDATED"
                    else:
                        high.COMMENT = high.COMMENT + ";ASSUMED TREND OK"
                    high.trend = self.mdi.trend
                    if self.mdi.trend == "uptrend" and high.prev and high.prev.prev and high.prev.prev.was_cms:
                        self.mdi.cms_list.append(CMSNodeInfo(high.prev.prev, "uptrend"))
                    if len(data_points) == 0:
                        break
                    if cms_removed:
                        if len(data_points) > 0:
                            self.mdi.market_structure.extend(data_points, consume=True)
                        return cms_removed, self.mdi.market_structure.tail
                if isinstance(self.mdi.market_structure.tail, Peak):
                    low = data_points.popfront()
                    # reset the node
                    low.is_permanent = True
                    low.is_cms = False
                    low.was_cms = False
                    low.COMMENT = ""
                    low.reset_observer_state()
                    self.mdi.market_structure.append(low)
                    low.COMMENT = "RECALCULATING; "
                    cms_removed = self.validate_low(low)

                    if cms_removed:
                        low.COMMENT = low.COMMENT + " ;ASSUMED TREND INVALIDATED"
                    else:
                        low.COMMENT = low.COMMENT + " ;ASSUMED TREND OK"

                    low.trend = self.mdi.trend
                    if self.mdi.trend == "downtrend" and low.prev and low.prev.prev and low.prev.prev.was_cms:
                        self.mdi.cms_list.append(CMSNodeInfo(low.prev.prev, "downtrend"))

                    if len(data_points) == 0:
                        break
                    if cms_removed:
                        if len(data_points) > 0:
                            self.mdi.market_structure.extend(data_points, consume=True)
                        return cms_removed, self.mdi.market_structure.tail
            return cms_removed, self.mdi.market_structure.tail
        except Exception as e:
            raise

    def calculate_market_structure(self):

        self.mdi.market_structure.changed_nodes.clear()
        for node in self.mdi.market_structure:
            node.reset_observer_state()
            node.value = node.value
        end_node = self.mdi.market_structure.tail
        try:
            while end_node is not None:
                trend_invalid, end_node = self.normal_trend()

                while trend_invalid:
                    trend_invalid, end_node = self.resolve_cms()
        except BadStartPoint as e:
            self.mdi.update_data()
            raise
        except Exception as e:
            raise
