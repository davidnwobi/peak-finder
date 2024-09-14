import typing as tp
from collections.abc import Iterable
from .MKS import *
import copy


class BadStartPoint(Exception):
    def __init__(self, message="Bad start point"):
        self.message = message
        super().__init__(self.message)


def lowest_trough_between_two_peaks(high1: Peak, high2: Peak, market_structure: MarketStructure) -> tp.Optional[Trough]:
    """
    Find the lowest trough between two peaks in the market structure.

    :param high1: The first peak.
    :param high2: The second peak.
    :param market_structure: The market structure containing nodes.
    :return: The lowest trough between the two peaks, or None if not found.
    """
    current_trough: tp.Optional[Node] = high1.next
    lowest_trough: tp.Optional[Node] = high1.next
    if high1.next is None:
        return None
    while current_trough.next.next and current_trough.next != high2:
        current_trough = current_trough.next.next
        if current_trough.value <= lowest_trough.value:
            lowest_trough = current_trough

    return lowest_trough


def highest_peak_between_two_trough(low1: Trough, low2: Trough, market_structure: MarketStructure) -> tp.Optional[Peak]:
    """
    Find the highest peak between two troughs in the market structure.

    :param low1: The first trough.
    :param low2: The second trough.
    :param market_structure: The market structure containing nodes.
    :return: The highest peak between the two troughs, or None if not found.
    """
    current_peak: tp.Optional[Node] = low1.next
    highest_peak: tp.Optional[Node] = low1.next
    if low1.next is None:
        return None
    while current_peak.next.next and current_peak.next != low2:
        current_peak = current_peak.next.next
        if current_peak.value >= highest_peak.value:
            highest_peak = current_peak
    return highest_peak


def find_cms(start: Node, end: Node) -> tp.Optional[Node]:
    """
    Find a Change in Market Structure (CMS) node within two nodes (exclusive).

    :param start: The starting node for the search.
    :param end: The ending node for the search (exclusive).
    :return: The CMS node found, or None if not found.
    """
    temp = start.next
    while temp and temp != end:
        if temp.was_cms:
            return temp
        temp = temp.next


def join_disconnect_left(left, mid, market_structure: MarketStructure):
    """
    Join nodes and disconnect the range on the left side of the midpoint.

    :param left: The left node of the range to be joined and disconnected.
    :param mid: The midpoint node.
    :param market_structure: The market structure containing nodes.
    """
    left_index = market_structure.index(left)
    mid_index = market_structure.index(mid)
    del market_structure[left_index + 1:mid_index]


def join_disconnect_right(mid, right, market_structure: MarketStructure):
    """
    Join nodes and disconnect the range on the right side of the midpoint.

    :param mid: The midpoint node.
    :param right: The right node of the range to be joined and disconnected.
    :param market_structure: The market structure containing nodes.
    """
    mid_index = market_structure.index(mid)
    right_index = market_structure.index(right)
    del market_structure[mid_index + 1:right_index]


def join_disconnect_both_sides(left, mid, right, market_structure: MarketStructure):
    """
    Join nodes and disconnect the range on both sides of the midpoint.

    :param left: The left node of the range to be joined and disconnected.
    :param mid: The midpoint node.
    :param right: The right node of the range to be joined and disconnected.
    :param market_structure: The market structure containing nodes.
    """
    join_disconnect_left(left, mid, market_structure)
    join_disconnect_right(mid, right, market_structure)


def clean_between_high(high1: Peak, high2: Peak, market_structure: MarketStructure):
    """
    Find the lowest trough between two peaks and clean the nodes between them.

    :param high1: The first peak.
    :param high2: The second peak.
    :param market_structure: The market structure containing nodes.
    :return: True if a CMS node was removed and not replaced by the lowest trough, False otherwise.
    """
    # finds the lowest trough between two peaks
    # deletes all nodes between the two peaks except the lowest trough

    # find the lowest trough between the two peaks
    cms = find_cms(high1, high2)
    cms_was_removed = True if cms is not None else False
    lowest_trough: tp.Optional[Node] = lowest_trough_between_two_peaks(high1, high2, market_structure)

    if lowest_trough is None:
        raise Exception("No trough found between the two peaks. This should not happen.")
    else:
        # delete all nodes between the two peaks except the lowest trough
        if high1.next == lowest_trough:
            if high2 == lowest_trough.next:
                return False
            else:
                join_disconnect_right(lowest_trough, high2, market_structure)
        else:
            if high2 == lowest_trough.next:
                join_disconnect_right(high1, lowest_trough, market_structure)
            else:
                join_disconnect_both_sides(high1, lowest_trough, high2, market_structure)
    return cms_was_removed and (cms != lowest_trough if cms is not None else False)


def clean_between_low(low1: Trough, low2: Trough, market_structure: MarketStructure):
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
    cms = find_cms(low1, low2)
    cms_was_removed = True if cms is not None else False
    highest_peak: tp.Optional[Node] = highest_peak_between_two_trough(low1, low2, market_structure)
    if highest_peak is None:
        raise RuntimeError("No peak found between the two troughs. This should not happen.")
    else:
        # delete all nodes between the two troughs except the highest peak
        if low1.next == highest_peak:
            if low2 == highest_peak.next:
                return False
            else:
                join_disconnect_right(highest_peak, low2, market_structure)
        else:
            if low2 == highest_peak.next:
                join_disconnect_left(low1, highest_peak, market_structure)
            else:
                join_disconnect_both_sides(low1, highest_peak, low2, market_structure)
    return cms_was_removed and (cms != highest_peak if cms is not None else False)


# test functions

def clean_between_points(point1: tp.Optional[Node], point2: tp.Optional[Node], market_structure: MarketStructure):
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
    if point1.next != point1:
        if isinstance(point1, Peak) and isinstance(point2, Peak):
            return clean_between_high(point1, point2, market_structure)
        elif isinstance(point1, Trough) and isinstance(point2, Trough):
            return clean_between_low(point1, point2, market_structure)
        else:
            # if the two points are not of the same type, join them
            cms_was_removed = True if find_cms(point1, point2) is not None else False
            point1_index = market_structure.index(point1)
            point2_index = market_structure.index(point2)
            del market_structure[point1_index + 1:point2_index]
            return cms_was_removed


def find_last_broken_low(low: Trough, market_structure: MarketStructure) -> tp.Optional[Node]:
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

        child = None  # the next low
        child_high = None  # the next low's high
        grandchild = None  # then next low's child's
        grandchild_high = None  # next low's child's high
        if last_checked_low:
            child = last_checked_low.next.next
            child_high = last_checked_low.next.next.next
            grandchild = last_checked_low.next.next.next.next
            grandchild_high = last_checked_low.next.next.next.next.next

        # verify that there are enough nodes up to the next low's high
        # verify that the trough is lower than the child
        # The last low we should check should be its next high, or its next low's high be a cms
        if last_checked_low and grandchild_high and \
                child.value > low.value and \
                (child_high.was_cms or grandchild_high.was_cms) and \
                last_checked_low.value >= child.value and child.value < grandchild.value:
            break
    if last_checked_low is None:
        return None
    elif last_checked_low.next.next == low:
        # In case it didn't enter the while loop, we check if the last low is lower than the current low. it is
        # redundant. Will try to remove this later
        if last_checked_low.value >= low.value:
            low.broke = True
            return last_checked_low
        else:
            return None
    else:
        return last_checked_low.next.next


def find_last_broken_high(high: Peak, market_structure: MarketStructure) -> tp.Optional[Node]:
    """
    Find the last (earliest) high that was broken by the given peak.

    :param high: The peak node.
    :param market_structure: The market structure containing nodes.
    :return: The last high that was broken by the given peak or None if no such high exists.
    """
    last_checked_high: tp.Optional[Node] = high.prev.prev if high.prev else None
    while last_checked_high and last_checked_high.prev and last_checked_high.value < high.value:
        last_checked_high = last_checked_high.prev.prev
        high.broke = True
        # the high is not the first node in the market structure
        # Believe me I thought about this for a while. this is the easiest way to do it
        # this serves to limit for far back we have to go to find the last high that was broken, so if the market makes
        # a completely new high, we don't have to go back to the beginning of time
        child = None  # the next high
        child_low = None  # the next high's low
        grandchild = None
        grandchild_low = None  # next high's child's low
        if last_checked_high:
            child = last_checked_high.next.next  # the next high
            child_low = last_checked_high.next.next.next  # the next high's low
            grandchild = last_checked_high.next.next.next.next
            grandchild_low = last_checked_high.next.next.next.next.next  # next high's child's low

        # verify that there are enough nodes up to the next high's low
        # verify that the peak is high than the child
        # The last high we should check have its child's or grandchild's low be a cms
        if last_checked_high and grandchild_low and \
                child.value < high.value and \
                (child_low.was_cms or grandchild_low.was_cms) and \
                last_checked_high.value <= child.value and child.value > grandchild.value:
            break  # we have found the last high that was broken
    if last_checked_high is None:
        return None
    elif last_checked_high.next.next == high:
        # In case it didn't enter the while loop, we check if the last high is higher than the current high. it is
        # redundant. Will try to remove this later
        if last_checked_high.value <= high.value:
            high.broke = True
            return last_checked_high
        else:
            return None
    else:
        return last_checked_high.next.next


def construct_market_structure(data_list, constructor) -> MarketStructure:
    """
    Construct a market structure from a list of data using the provided constructor function.

    :param data_list: The list of data to construct the market structure from.
    :param constructor: A constructor function that takes index and data arguments to create nodes.
    :return: The constructed market structure.
    """
    market_structure = MarketStructure()
    for i in range(0, len(data_list)):
        if isinstance(data_list[i], Iterable):
            item = constructor(i, *data_list[i])
        else:
            item = constructor(i, data_list[i])
        market_structure.append(item)
    return market_structure


def fill_market_structure_with_data(data: tp.List[
    tp.Union[int, float, tp.Tuple[float, bool], tp.Tuple[tp.Union[datetime, pd.Timestamp], float, bool]]]):
    """
    Fill a market structure with data based on the provided list.
    Supports three data formats:
    1. A list of numbers (int or float)
    2. A list of tuples of length 2, where the first item is a number (int or float) and the second item is a boolean
    3. A list of tuples of length 3, where the first item is a datetime, the second item is a number (int or float),

    :param data: The list of data to fill the market structure with.
    :return: The filled market structure.
    """
    if len(data) % 2 == 0:
        raise ValueError("Data must have an odd length")
    if len(data) < 3:
        raise ValueError("Data must have a length of 3 or greater")

    first_item = data[0]
    if isinstance(first_item, (int, float)):
        return construct_market_structure(data, lambda i, args: Peak(date_index=datetime.now(),
                                                                     value=args) if i % 2 == 0 else Trough(
            date_index=datetime.now(),
            value=args))

    if isinstance(first_item, tuple) and len(first_item) == 2 and isinstance(first_item[0],
                                                                             (int, float)) and isinstance(first_item[1],
                                                                                                          bool):
        return construct_market_structure(data, lambda i, *args: Peak(date_index=datetime.now(),
                                                                      value=args[0],
                                                                      is_permanent=args[1]) if i % 2 == 0 else
        Trough(date_index=datetime.now(),
               value=args[0],
               is_permanent=args[1]))

    if isinstance(first_item, tuple) and len(first_item) == 3 and isinstance(first_item[0],
                                                                             (datetime, pd.Timestamp)) and isinstance(
        first_item[2], bool):
        return construct_market_structure(data, lambda i, *args: Peak(date_index=args[0], value=args[1],
                                                                      is_permanent=args[
                                                                          2]) if i % 2 == 0 else Trough(
            date_index=args[0], value=args[1], is_permanent=args[2]))

    raise ValueError("Unsupported data format")


def find_last_permanent_high(high: Peak, market_structure: MarketStructure) -> tp.Optional[Peak]:
    """
    Find the last permanent high before the given peak.

    :param high: The peak node.
    :param market_structure: The market structure containing nodes.
    :return: The last permanent high before the given peak, or None if no such high exists.
    """
    if high.prev is None:
        return None
    if high.prev.prev is None:
        return None
    last_high = high.prev.prev
    while last_high and not last_high.is_permanent:
        last_high = last_high.prev.prev
    return last_high


def find_last_permanent_low(low: Trough, market_structure: MarketStructure) -> tp.Optional[Trough]:
    """
    Find the last permanent low before the given trough.

    :param low: The trough node.
    :param market_structure: The market structure containing nodes.
    :return: The last permanent low before the given trough, or None if no such low exists.
    """
    if low.prev is None:
        return None
    if low.prev.prev is None:
        return None
    last_low = low.prev.prev
    while last_low and not last_low.is_permanent:
        last_low = last_low.prev.prev
    return last_low


def do_between(point1, process_func, next_func, point2=None, break_cond_func=None):
    """
    Perform a process function on nodes between two points or until a break condition is met.
    the function is work on all points ∈ [point1, point2)

    :param point1: The starting node.
    :param process_func: The function to process nodes.
    :param next_func: The function to get the next node.
    :param point2: The ending node.
    :param break_cond_func: The break condition function.
    """
    temp = point1
    if point2 is None and break_cond_func is None:
        raise Exception("An End Point or Break Condition must be provided")
    elif point2 is None:
        while temp and not break_cond_func(temp):
            process_func(temp)
            temp = next_func(temp)
    else:
        while temp and temp is not point2:
            process_func(temp)
            temp = next_func(temp)


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


def broke_permanent_low(low: Trough):
    """
    Check if a permanent low has been broken.
    This function should only be called if a low has been broken. Undefined behavior otherwise.
    :param low: The trough node.
    :return: True if a permanent low has been broken, False otherwise.
    """
    # we know a low has been broken so no need to check if low is none
    last_checked_low: tp.Optional[Trough] = low.prev.prev
    if last_checked_low.is_permanent:
        return True
    while last_checked_low and last_checked_low.prev and last_checked_low.value >= low.value:
        last_checked_low = last_checked_low.prev.prev
        if last_checked_low and last_checked_low.is_permanent and last_checked_low.value >= low.value:
            return True
    return False


def broke_permanent_high(high: Peak):
    """
    Check if a permanent high has been broken.
    This function should only be called if a high has been broken. Undefined behavior otherwise.
    :param high: The peak node.
    :return: True if a permanent high has been broken, False otherwise.
    """
    # we know a high has been broken so no need to check if high is none
    last_checked_high: tp.Optional[Peak] = high.prev.prev
    if last_checked_high.is_permanent:
        return True
    while last_checked_high and last_checked_high.prev and last_checked_high.value <= high.value:
        last_checked_high = last_checked_high.prev.prev
        if last_checked_high and last_checked_high.is_permanent and last_checked_high.value <= high.value:
            return True
    return False


def verify_cms_in_market(cms: Node, final_node_in_market_structure: Node):
    """
    Verify if a CMS (Change in Market Structure) node exists within the market structure.

    :param cms: The CMS node to verify.
    :param final_node_in_market_structure: The final node in the market structure.
    :return: True if the CMS node is found within the market structure, False otherwise.
    """
    temp = cms
    while temp != final_node_in_market_structure and temp.next:
        temp = temp.next
    return temp == final_node_in_market_structure


def find_removed_cms(final_node_in_market_structure: Node, cms_list: tp.List[tp.Tuple[Node, str]]) -> tp.Tuple[
    Node, str]:
    """
    Find the removed CMS (Change in Market Structure) node from a list of CMS nodes.

    :param final_node_in_market_structure: The final node in the market structure.
    :param cms_list: List of CMS nodes along with their associated directions.
    :return: Tuple containing the removed CMS node and its associated direction.
    """
    for node in reversed(cms_list):
        if not verify_cms_in_market(node[0], final_node_in_market_structure):
            return node


def end_node_post_check(node1: Node, node2: Node,
                        comparison_func: tp.Callable[[Node, Node], bool]) -> Node:
    """
    Determine the end node after performing a post-checking based on the comparison function.

    This function is used to decide the final end node of a pattern after checking conditions with two different nodes
    in order to ensure accuracy. It checks if the pattern is valid based on the provided comparison function.

    :param node1: The first node to consider for checking.
    :param node2: The second node to consider for checking.
    :param comparison_func: A callable function that takes two nodes and returns True if the pattern is valid, else False.
    :return: The selected end node based on the comparison function.
    """
    if node1.next is None:
        return node1
    elif node1.next.next is None:  # i.e it ended on a type(node2)
        if comparison_func(node2, node2.next.next):  # check if node2 is okay
            return node2.next.next
        else:
            return node1
    else:
        return node1.next


def validate_high(market_structure: MarketStructure, current_high: Peak, trend_direction: str):
    """
    Validate a high point in the market structure based on the given trend direction.

    This function ensures the validity and consistency of the market structure in accordance with the specified trend.
    A valid trend is one in which the highest local high is joined to the lowest local low, creating a visually
    recognizable pattern. This function programmatically enforces this principle.

    :param market_structure: The market structure containing peaks and troughs.
    :param current_high: The high point to be validated.
    :param trend_direction: The current trend direction ("uptrend" or "downtrend").

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

    cms_was_removed = False
    if trend_direction == "uptrend":

        if broke_a_high(current_high):

            if broke_permanent_high(current_high):
                last_permanent_high = find_last_permanent_high(current_high, market_structure)
                current_high.is_permanent = True
                cms_was_removed = cms_was_removed or clean_between_points(last_permanent_high, current_high,
                                                                          market_structure)
                prev_last_permanent_low = find_last_permanent_low(current_high.prev, market_structure)
                if prev_last_permanent_low is not None:
                    cms_was_removed = cms_was_removed or clean_between_points(prev_last_permanent_low,
                                                                              current_high.prev, market_structure)
                # Clear this up so this it is not triggerd in the future
                current_high.prev.is_cms = False

            else:
                if not current_high.prev.is_cms:
                    last_permanent_low = find_last_permanent_low(current_high.prev,
                                                                 market_structure) if current_high.prev is not None else None

                    if last_permanent_low is not None:
                        if last_permanent_low.prev.value > current_high.value:  # which high got broken?
                            cms_was_removed = cms_was_removed or clean_between_points(last_permanent_low, current_high,
                                                                                      market_structure)
                        else:
                            last_permanent_high = find_last_permanent_high(current_high, market_structure)
                            if last_permanent_high is not None:
                                cms_was_removed = cms_was_removed or clean_between_points(last_permanent_high,
                                                                                          current_high,
                                                                                          market_structure)

                            current_high.prev.is_cms = False

                else:  # then we are assured that the temporary high is between cms and last permanent high since we
                    # know a temp high was broken
                    last_permanent_high = find_last_permanent_high(current_high, market_structure)
                    if last_permanent_high is not None:
                        cms_was_removed = cms_was_removed or clean_between_points(last_permanent_high, current_high,
                                                                                  market_structure)

                    current_high.prev.is_cms = False
                current_high.is_permanent = False

        else:
            current_high.is_permanent = False
    else:
        if broke_a_high(current_high):
            if broke_permanent_high(current_high):
                last_permanent_high = find_last_permanent_high(current_high, market_structure)
                if last_permanent_high.is_cms:
                    trend_direction = "uptrend"
                    current_high.is_permanent = True
                    cms_was_removed = cms_was_removed or clean_between_points(last_permanent_high, current_high,
                                                                              market_structure)

                    last_permanent_high.is_cms = False

                    def process_node(node):
                        node.is_permanent = True
                        node.next.is_permanent = True

                    def get_prev_prev(node):
                        if node.prev is not None and node.prev.prev is not None:
                            return node.prev.prev
                        else:
                            return None

                    def break_condition(node):
                        return node.is_permanent is True

                    do_between(last_permanent_high.prev.prev, process_node, get_prev_prev,
                               break_cond_func=break_condition)
                    current_high.prev.is_permanent = True  # set trough before peak2(the peak that confirms cms) to True
                    last_permanent_high.prev.is_permanent = True  # set trough before peak1(the peak that needed to be broken
                    # to confirms cms) to True

                    last_permanent_high.was_cms = True
                else:
                    last_broken_high = find_last_broken_high(current_high, market_structure)
                    if last_broken_high is None:
                        raise BadStartPoint
                    cms_was_removed = cms_was_removed or clean_between_high(last_permanent_high, current_high,
                                                                            market_structure)
                    last_high = current_high.prev.prev if (
                            current_high.prev is not None and current_high.prev.prev is not None) else None

                    def process_node(node):
                        node.is_permanent = False

                    def get_prev_prev(node):
                        if node.prev is not None and node.prev.prev is not None:
                            return node.prev.prev
                        else:
                            return None

                    if last_broken_high.prev and last_broken_high.prev and last_broken_high.prev.prev.value <= \
                            last_broken_high.value:
                        last_broken_high = last_broken_high.next.next
                    do_between(last_high, process_node, get_prev_prev, point2=last_broken_high)
                    current_high.is_cms = True
                    last_broken_high.is_permanent = False

            else:

                last_permanent_high = find_last_permanent_high(current_high,
                                                               market_structure) if current_high.prev is not None else None
                current_high.is_permanent = False
                cms_was_removed = cms_was_removed or clean_between_points(last_permanent_high, current_high,
                                                                          market_structure)

        else:
            if current_high.prev is not None and current_high.prev.is_permanent is False:
                current_high.is_permanent = False
    return trend_direction, cms_was_removed


def validate_low(market_structure: MarketStructure, current_low: Trough, trend_direction: str):
    """
        Validate a low point in the market structure based on the given trend direction.

        This function is used to validate a low point in the market structure based on the specified trend direction.
        The goal is to ensure that the market structure is consistent with the specified trend. A valid trend is one
        in which the lowest local low is joined to the highest local high. This function performs the necessary checks
        and modifications to maintain a valid market structure.

        :param market_structure: The market structure containing peaks and troughs.
        :param current_low: The low point to be validated.
        :param trend_direction: The current trend direction ("uptrend" or "downtrend").

        * Uptrend Validation Logic:
            The logic for uptrend validation is the inverse of downtrend validation. Since lows and highs are inverse in
            their roles, the checks and operations performed mirror those in the `validate_high` function.

        *DownTrend Validation Logic:
            The downtrend validation logic is very similar to the `validate_high` function, with roles reversed for lows
            and highs. This function ensures that the market structure is consistent with a downtrend.

        Note: The function does not return anything, as it modifies the market_structure and trend_direction in place.
        Note that whether a cms was removed is being tracked. If this happens, it invalidates the trend from where the cms
        was removed, and this is handled elsewhere.
    """
    cms_was_removed = False
    if trend_direction == "uptrend":
        if broke_a_low(current_low):
            if broke_permanent_low(current_low):
                last_permanent_low = find_last_permanent_low(current_low, market_structure)
                if last_permanent_low.is_cms:
                    trend_direction = "downtrend"
                    current_low.is_permanent = True
                    cms_was_removed = cms_was_removed or clean_between_points(last_permanent_low, current_low,
                                                                              market_structure)
                    last_permanent_low.is_cms = False

                    def process_node(node):
                        node.is_permanent = True
                        node.next.is_permanent = True

                    def get_prev_prev(node):
                        if node.prev is not None and node.prev.prev is not None:
                            return node.prev.prev
                        else:
                            return None

                    def break_condition(node):
                        return node.is_permanent is True

                    do_between(last_permanent_low.prev.prev, process_node, get_prev_prev,
                               break_cond_func=break_condition)
                    current_low.prev.is_permanent = True  # set peak before trough2(the trough that confirms cms) to
                    # True
                    last_permanent_low.prev.is_permanent = True  # set peak before trough2(the trough that needed to
                    # be broken
                    # to confirms cms) to True. It will only be one
                    last_permanent_low.was_cms = True
                else:
                    last_broken_low = find_last_broken_low(current_low, market_structure)
                    if last_broken_low is None:
                        raise BadStartPoint

                    # same assurance we give that when important events like these occur. We get rid of temp nodes
                    # to keep bipartite-ness
                    cms_was_removed = cms_was_removed or clean_between_points(last_permanent_low, current_low,
                                                                              market_structure)

                    last_low = current_low.prev.prev if (
                            current_low.prev is not None and current_low.prev.prev is not None) else None

                    def process_node(node):
                        node.is_permanent = False

                    def get_prev_prev(node):
                        if node.prev is not None and node.prev.prev is not None:
                            return node.prev.prev
                        else:
                            return None

                    if last_broken_low.prev and last_broken_low.prev and last_broken_low.prev.prev.value >= \
                            last_broken_low.value:
                        last_broken_low = last_broken_low.next.next

                    do_between(last_low, process_node, get_prev_prev, point2=last_broken_low)
                    current_low.is_cms = True
                    last_broken_low.is_permanent = False

            else:

                last_permanent_low = find_last_permanent_low(current_low,
                                                             market_structure) if current_low.prev is not None else None
                current_low.is_permanent = False
                cms_was_removed = cms_was_removed or clean_between_points(last_permanent_low, current_low,
                                                                          market_structure)

        else:
            if current_low.prev is not None and current_low.prev.is_permanent is False:
                current_low.is_permanent = False

    else:
        if broke_a_low(current_low):

            if broke_permanent_low(current_low):
                last_permanent_low = find_last_permanent_low(current_low, market_structure)
                current_low.is_permanent = True
                cms_was_removed = cms_was_removed or clean_between_points(last_permanent_low, current_low,
                                                                          market_structure)
                prev_last_permanent_high = find_last_permanent_high(current_low.prev, market_structure)
                if prev_last_permanent_high is not None:
                    cms_was_removed = cms_was_removed or clean_between_points(prev_last_permanent_high,
                                                                              current_low.prev, market_structure)

                # Clear this up so this it is not triggerd in the future
                current_low.prev.is_cms = False

            else:
                if not current_low.prev.is_cms:
                    last_permanent_high = find_last_permanent_high(current_low.prev,
                                                                   market_structure) if current_low.prev is not None else None

                    if last_permanent_high is not None:
                        if last_permanent_high.prev.value < current_low.value:  # which low got broken?
                            cms_was_removed = cms_was_removed or clean_between_points(last_permanent_high, current_low,
                                                                                      market_structure)
                        else:
                            last_permanent_low = find_last_permanent_low(current_low, market_structure)
                            if last_permanent_low is not None:
                                cms_was_removed = cms_was_removed or clean_between_points(last_permanent_low,
                                                                                          current_low,
                                                                                          market_structure)

                            current_low.prev.is_cms = False

                else:  # then we are assured that the temporary low is between cms and last permanent low since we
                    # know a temp low was broken
                    last_permanent_low = find_last_permanent_low(current_low, market_structure)
                    if last_permanent_low is not None:
                        cms_was_removed = cms_was_removed or clean_between_points(last_permanent_low, current_low,
                                                                                  market_structure)

                    current_low.prev.is_cms = False
                current_low.is_permanent = False
        else:
            current_low.is_permanent = False

    return trend_direction, cms_was_removed


def is_high(market_price_slice):
    expected_peak = market_price_slice[1]
    return expected_peak == max(market_price_slice)


def is_low(market_price_slice):
    expected_trough = market_price_slice[1]
    return expected_trough == min(market_price_slice)


def find_a_high(current_idx, market_price, last_low):
    market_price_slice = copy.deepcopy(market_price[current_idx - 1:current_idx + 2])
    market_price_slice[0] = last_low
    while current_idx < len(market_price) - 1 and not is_high(market_price_slice):
        current_idx += 1
        market_price_slice = market_price[current_idx - 1:current_idx + 2]
    return current_idx


def find_a_low(current_idx, market_price, last_high):
    market_price_slice = copy.deepcopy(market_price[current_idx - 1:current_idx + 2])
    market_price_slice[0] = last_high
    while current_idx < len(market_price) - 1 and not is_low(market_price_slice):
        current_idx += 1
        market_price_slice = market_price[current_idx - 1:current_idx + 2]
    return current_idx
