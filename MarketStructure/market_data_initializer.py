# import tpqoa
import datetime as dt
from . import MKS
import pandas as pd
import typing as tp
import copy
import re
from . import data_retrieval as dr
import numba
import numpy as np
from numba.typed import List
from collections import namedtuple

CMSNodeInfo = namedtuple("CMSNodeInfo", ["concrete_node", "direction"])
NodeInfo = namedtuple("NodeInfo", ["index", "is_peak"])


@numba.njit(cache=True)
def is_high_numpy(market_price_slice: np.ndarray):
    expected_peak = market_price_slice[1]
    return expected_peak == market_price_slice.max()


@numba.njit(cache=True)
def is_low_numpy(market_price_slice: np.ndarray):
    expected_trough = market_price_slice[1]
    return expected_trough == market_price_slice.min()


@numba.njit(cache=True)
def find_highs_numpy(current_idx: int, market_price: np.ndarray) -> list:
    # TODO : Rework this Logic
    # 0 = time, 1 = open, 2 = high, 3 = low, 4 = close
    # New index 0 = open, 1 = high, 2 = low, 3 = close
    market_price_slice = market_price[current_idx - 1:current_idx + 2].copy()
    highs = []
    while current_idx < len(market_price) - 1:
        if is_high_numpy(market_price_slice[:, 1]):
            highs.append(current_idx)
        elif market_price[current_idx - 1, 0] > market_price[current_idx - 1, 3] and market_price[current_idx, 0] < \
                market_price[current_idx, 3]:
            market_price_slice[0] = market_price[current_idx]
            if is_high_numpy(market_price_slice[:, 1]):
                highs.append(current_idx)
        current_idx += 1
        market_price_slice = market_price[current_idx - 1:current_idx + 2].copy()
    return highs


@numba.njit(cache=True)
def find_lows_numpy(current_idx: int, market_price: np.ndarray) -> list:
    market_price_slice = market_price[current_idx - 1:current_idx + 2].copy()
    lows = []
    while current_idx < len(market_price) - 1:
        if is_low_numpy(market_price_slice[:, 2]):
            lows.append(current_idx)
        elif market_price[current_idx - 1, 0] < market_price[current_idx - 1, 3] and market_price[current_idx, 0] > \
                market_price[current_idx, 3]:
            market_price_slice[0] = market_price[current_idx]
            if is_low_numpy(market_price_slice[:, 2]):
                lows.append(current_idx)
        current_idx += 1
        market_price_slice = market_price[current_idx - 1:current_idx + 2].copy()
    return lows


@numba.njit(cache=True)
def find_highs_find_lows(market_price: np.ndarray) -> tp.Tuple[list, list]:
    highs = find_highs_numpy(current_idx=1, market_price=market_price)
    lows = find_lows_numpy(current_idx=1, market_price=market_price)

    return highs, lows


@numba.njit(cache=True)
def efficient_high_part_find(curr_high_index: int, curr_low_index: int, node_stack: numba.typed.List,
                             high_timestamp: np.ndarray, low_timestamp: np.ndarray, highs: np.ndarray,
                             lows: np.ndarray) -> tp.Tuple[int, int, bool]:
    if curr_high_index < 0 or curr_low_index < 0:
        return curr_high_index, curr_low_index, False

    peak_index = curr_high_index
    curr_high_index -= 1
    while curr_high_index >= 0 and high_timestamp[curr_high_index] >= low_timestamp[curr_low_index]:
        if highs[peak_index] < highs[curr_high_index]:
            peak_index = curr_high_index
            curr_high_index -= 1
        # We don't want to delete this low because it could become valid after the next high
        elif high_timestamp[curr_high_index] != low_timestamp[curr_low_index]:
            curr_high_index -= 1
        else:
            break

        if curr_high_index < 0:
            node_stack.append(NodeInfo(peak_index, True))
            return curr_high_index, curr_low_index, False
    node_stack.append(NodeInfo(peak_index, True))
    if curr_high_index < 0:
        return curr_high_index, curr_low_index, False
    return curr_high_index, curr_low_index, True


@numba.njit(cache=True)
def efficient_low_part_find(curr_high_index: int, curr_low_index: int, node_stack: numba.typed.List,
                            high_timestamp: np.ndarray, low_timestamp: np.ndarray, highs: np.ndarray,
                            lows: np.ndarray) -> tp.Tuple[int, int, bool]:
    if curr_high_index < 0 or curr_low_index < 0:
        return curr_high_index, curr_low_index, False
    low_index = curr_low_index
    curr_low_index -= 1
    while curr_low_index >= 0 and low_timestamp[curr_low_index] >= high_timestamp[curr_high_index]:
        if lows[low_index] > lows[curr_low_index]:
            # this is a lower low
            low_index = curr_low_index
            curr_low_index -= 1
        elif low_timestamp[curr_low_index] != high_timestamp[curr_high_index]:
            # remove it
            curr_low_index -= 1
        else:
            break

        if curr_low_index < 0:
            node_stack.append(NodeInfo(low_index, False))
            return curr_high_index, curr_low_index, False
    node_stack.append(NodeInfo(low_index, False))
    if curr_low_index < 0:
        return curr_high_index, curr_low_index, False
    return curr_high_index, curr_low_index, True


@numba.njit(cache=True)
def efficient_post_process(current_high_index: int, current_low_index: int, node_stack: numba.typed.List) -> bool:
    if current_high_index < 0 and current_low_index < 0:
        return node_stack[-1].is_peak
    if current_high_index >= 0:
        node_stack.append(NodeInfo(current_high_index, True))
        return True  # First node is a high
    if current_low_index >= 0:
        node_stack.append(NodeInfo(current_low_index, False))
        return False  # First node is a low


@numba.njit(cache=True)
def efficient_split_convert_nodes(node_stack: numba.typed.List, high_first, highs: np.ndarray, lows: np.ndarray,
                                  high_timestamp: np.ndarray, low_timestamp: np.ndarray) -> \
        tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_stack.reverse()
    node_stack.pop()

    index_highs = np.full(len(highs), False, dtype=np.bool_)
    index_lows = np.full(len(lows), False, dtype=np.bool_)
    for record in node_stack:
        if record.is_peak:
            index_highs[record.index] = True
        else:
            index_lows[record.index] = True

    if high_first:
        return highs[index_highs], lows[index_lows], high_timestamp[index_highs], low_timestamp[index_lows]
    else:
        return highs[index_highs], lows[index_lows], high_timestamp[index_highs], low_timestamp[index_lows]


@numba.njit(cache=True)
def efficient_clean_points(high_last: bool, high_timestamp: np.ndarray, low_timestamp: np.ndarray, highs: np.ndarray,
                           lows: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    high_first = True
    successful = True
    node_stack = List()
    node_stack.append(NodeInfo(1, True))  # This is for numba to get a signature
    current_high_index = len(highs) - 1
    current_low_index = len(lows) - 1
    if high_last:
        while successful:
            current_high_index, current_low_index, successful = (
                efficient_high_part_find(curr_high_index=current_high_index, curr_low_index=current_low_index,
                                         node_stack=node_stack, high_timestamp=high_timestamp,
                                         low_timestamp=low_timestamp, highs=highs, lows=lows))
            if not successful:
                break
            current_high_index, current_low_index, successful = (
                efficient_low_part_find(curr_high_index=current_high_index, curr_low_index=current_low_index,
                                        node_stack=node_stack, high_timestamp=high_timestamp,
                                        low_timestamp=low_timestamp, highs=highs, lows=lows))
        high_first = efficient_post_process(current_high_index, current_low_index, node_stack)
    else:
        while successful:
            current_high_index, current_low_index, successful = (
                efficient_low_part_find(curr_high_index=current_high_index, curr_low_index=current_low_index,
                                        node_stack=node_stack, high_timestamp=high_timestamp,
                                        low_timestamp=low_timestamp, highs=highs, lows=lows))
            if not successful:
                break
            current_high_index, current_low_index, successful = (
                efficient_high_part_find(curr_high_index=current_high_index, curr_low_index=current_low_index,
                                         node_stack=node_stack, high_timestamp=high_timestamp,
                                         low_timestamp=low_timestamp, highs=highs, lows=lows))
        high_first = efficient_post_process(current_high_index=current_high_index, current_low_index=current_low_index,
                                            node_stack=node_stack)

    return efficient_split_convert_nodes(node_stack, high_first, highs, lows, high_timestamp, low_timestamp)


class BadStartPoint(Exception):
    def __init__(self, message="Bad start point"):
        self.message = message
        super().__init__(self.message)


class MarketDataInitializer:
    def __init__(self, data_retriever: dr.DataRetriever):
        self.is_init = False
        self.market_structure = None
        self.instrument = ""
        self.granularity = ""
        self.data = pd.DataFrame()
        self.data_retriever: dr.DataRetriever = data_retriever
        self.start_date: dt.datetime = dt.datetime.now()
        self.end_date: dt.datetime = dt.datetime.now()
        self.high_date_index: tp.Union[None, np.ndarray] = None
        self.low_date_index: tp.Union[None, np.ndarray] = None
        self.highs: tp.Union[None, np.ndarray] = None
        self.lows: tp.Union[None, np.ndarray] = None
        self.high_generator: tp.Callable = lambda: self.raise_exception("Data not initialized")
        self.low_generator: tp.Callable = lambda: self.raise_exception("Data not initialized")
        self.trend: tp.Optional[str] = None
        self.no_of_tries = 100
        self.no_of_seconds_per_period = 0
        self.default_periods = 28  # seemed good enough
        self.animation_list = []
        self.prev_market_structure = MKS.MarketStructure()
        self.cms_list: tp.List[tp.Optional[CMSNodeInfo]] = []
        self.animate = False

    @staticmethod
    def raise_exception(message):
        def inner_function():
            raise Exception(message)

        return inner_function

    def get_data(self):
        return self.data_retriever.get_data(self.instrument, self.start_date, self.end_date, self.granularity)

    def update_data(self):
        max_retries = 10
        retry_count = 0
        new_start_date = self.start_date
        while retry_count < max_retries:
            try:
                new_start_date = new_start_date - dt.timedelta(
                    seconds=self.default_periods * self.no_of_seconds_per_period)

                self.data = self.data_retriever.update_data(self.instrument, new_start_date, self.start_date,
                                                            self.granularity,
                                                            self.data)
                self.start_date = new_start_date
                break

            except dr.DataRetrieverException as e:
                retry_count += 1
                if retry_count == max_retries:
                    self.start_date = new_start_date
                    raise
            except Exception as e:
                self.start_date = new_start_date
                raise

    def validate_granularity(self):
        try:
            possible_oanda_granularities = dict(S={"S5", "S10", "S15", "S30"},
                                                M={'M1', 'M2', 'M4', 'M5', 'M10', 'M15', 'M30'},
                                                H={"H1", 'H2', 'H3', 'H4', 'H6', 'H8', 'H12'},
                                                D={"D"},
                                                W={"W"})
            mult = {"S": 1, "M": 60, "H": 60 * 60, "D": 60 * 60 * 24, "W": 60 * 60 * 24 * 7}
            pattern = r'^([SMHD])'
            match = re.match(pattern, self.granularity)
            if not match:
                raise ValueError(f"""Invalid Granularity. Granularity for Oanda must be one of these:
            S5, S10, S15,  30, M1, M2, M4, M5, M10, M15, M30, H1, H2, H3, H4, H6, H8, H12, D, W, M """)
            general_time_frame = match.group(1)
            if self.granularity == "M":
                self.no_of_seconds_per_period = 60 * 60 * 24 * 30
            elif self.granularity == "D":
                self.no_of_seconds_per_period = 60 * 60 * 24
            elif self.granularity == "W":
                self.no_of_seconds_per_period = 60 * 60 * 24 * 7
            else:
                if general_time_frame in possible_oanda_granularities and self.granularity in \
                        possible_oanda_granularities[general_time_frame]:
                    self.no_of_seconds_per_period = mult[general_time_frame] * int(self.granularity[1:])
            if self.no_of_seconds_per_period == 0:
                raise ValueError(f"""Invalid Granularity. Granularity for Oanda must be one of these:
            S5, S10, S15,  30, M1, M2, M4, M5, M10, M15, M30, H1, H2, H3, H4, H6, H8, H12, D, W, M """)
        except ValueError as err:
            raise

    def start_initialization(self, instrument, start_date: dt.datetime, end_date: dt.datetime, granularity,
                             animate=False):
        try:
            self.instrument = instrument
            self.start_date = start_date
            self.end_date = end_date
            self.granularity = granularity
            self.validate_granularity()
            self.initialization()
            self.is_init = True
            self.animate = animate
            if self.animate:
                self.animation_list = []
        except Exception as err:
            raise

    def reinitialize(self, instrument, start_date: dt.datetime, end_date: dt.datetime, granularity, animate=False):
        self.is_init = False
        self.market_structure = None
        self.instrument = ""
        self.granularity = ""
        self.data = pd.DataFrame()
        self.start_date: dt.datetime = dt.datetime.now()
        self.end_date: dt.datetime = dt.datetime.now()
        self.high_date_index: tp.Union[None, np.ndarray] = None
        self.low_date_index: tp.Union[None, np.ndarray] = None
        self.highs: tp.Union[None, np.ndarray] = None
        self.lows: tp.Union[None, np.ndarray] = None
        self.high_generator: tp.Callable = lambda: self.raise_exception("Data not initialized")
        self.low_generator: tp.Callable = lambda: self.raise_exception("Data not initialized")
        self.trend: tp.Optional[str] = None
        self.no_of_tries = 100
        self.no_of_seconds_per_period = 0
        self.default_periods = 28  # seemed good enough
        self.animation_list = []
        self.prev_market_structure = MKS.MarketStructure()
        self.cms_list = []
        self.animate = False
        self.start_initialization(instrument, start_date, end_date, granularity, animate)

    def look_for_uptrend(self, structure: MKS.MarketStructure, no_of_pairs=3):

        curr_node_type1 = structure.head
        curr_node_type2 = structure.head.next

        for idx, node in enumerate(structure):
            if 1 < idx < 2 * no_of_pairs:
                if isinstance(node, type(curr_node_type1)):
                    if node.value <= curr_node_type1.value:
                        return False
                    curr_node_type1 = node
                else:
                    if node.value <= curr_node_type2.value:
                        return False
                    curr_node_type2 = node
        return True

    def look_for_downtrend(self, structure: MKS.MarketStructure, no_of_pairs=3):

        curr_node_type1 = structure.head
        curr_node_type2 = structure.head.next
        for idx, node in enumerate(structure):
            if 1 < idx < 2 * no_of_pairs:
                if isinstance(node, type(curr_node_type1)):
                    if node.value >= curr_node_type1.value:
                        return False
                    curr_node_type1 = node
                else:
                    if node.value >= curr_node_type2.value:
                        return False
                    curr_node_type2 = node
        return True

    def final_uptrend_validation(self):

        return self.look_for_uptrend(self.market_structure, no_of_pairs=2)

    def final_downtrend_validation(self):
        return self.look_for_downtrend(self.market_structure, no_of_pairs=2)

    def fill_for_uptrend(self, structure: MKS.MarketStructure, current_idx: int):
        current_idx = self.find_a_high(current_idx, self.data.h, self.data.iloc[current_idx].l)
        high = MKS.Peak(self.data.iloc[current_idx].name, self.data.iloc[current_idx].h, index=current_idx)
        structure.append(high)
        high.trend = "uptrend"
        for i in range(3):
            current_idx = self.find_a_low(current_idx + 1, self.data.l, structure.tail.value)
            low = MKS.Trough(self.data.iloc[current_idx].name, self.data.iloc[current_idx].l, index=current_idx)
            structure.append(low)
            low.trend = "uptrend"
            current_idx = self.find_a_high(current_idx + 1, self.data.h, structure.tail.value)
            high = MKS.Peak(self.data.iloc[current_idx].name, self.data.iloc[current_idx].h, index=current_idx)
            structure.append(high)
            high.trend = "uptrend"
        return current_idx

    def fill_for_downtrend(self, structure: MKS.MarketStructure, current_idx: int):
        current_idx = self.find_a_low(current_idx, self.data.l, self.data.iloc[current_idx].h)
        low = MKS.Trough(self.data.iloc[current_idx].name, self.data.iloc[current_idx].l, index=current_idx)
        structure.append(low)
        low.trend = "downtrend"
        for i in range(3):
            current_idx = self.find_a_high(current_idx + 1, self.data.h, structure.tail.value)
            high = MKS.Peak(self.data.iloc[current_idx].name, self.data.iloc[current_idx].h, index=current_idx)
            structure.append(high)
            high.trend = "downtrend"
            current_idx = self.find_a_low(current_idx + 1, self.data.l, structure.tail.value)
            low = MKS.Trough(self.data.iloc[current_idx].name, self.data.iloc[current_idx].l, index=current_idx)
            structure.append(low)
            low.trend = "downtrend"
        return current_idx

    def search_for_start_point(self, try_no=0) -> tp.Tuple[
        MKS.MarketStructure, int, str, dt.datetime, dt.datetime]:
        try:
            if self.data.empty:
                self.data = self.get_data()
            current_idx = 1
            structure = MKS.MarketStructure()
            current_idx = self.fill_for_uptrend(structure, current_idx)
            if self.look_for_uptrend(structure):
                self.trend = "uptrend"
                return structure, current_idx, "uptrend", self.start_date, self.end_date

            current_idx = 1
            structure = MKS.MarketStructure()
            current_idx = self.fill_for_downtrend(structure, current_idx)
            if self.look_for_downtrend(structure):
                self.trend = "downtrend"
                return structure, current_idx, "downtrend", self.start_date, self.end_date

            if try_no < self.no_of_tries:
                self.update_data()
                return self.search_for_start_point(try_no=try_no + 1)

            else:
                raise Exception("Could not find a start point")
        except dr.DataRetrieverException as e:
            raise
        except Exception as err:
            raise err

    def is_high(self, market_price_slice: pd.Series):
        expected_peak = market_price_slice.iloc[1]
        return expected_peak == market_price_slice.max()

    def is_low(self, market_price_slice: pd.Series):
        expected_trough = market_price_slice.iloc[1]
        return expected_trough == market_price_slice.min()

    def find_a_high(self, current_idx, market_price, last_low):
        market_price_slice = copy.deepcopy(market_price.iloc[current_idx - 1:current_idx + 2])
        market_price_slice.iloc[0] = last_low
        while current_idx < len(market_price) - 1 and not self.is_high(market_price_slice):
            current_idx += 1
            market_price_slice = market_price.iloc[current_idx - 1:current_idx + 2]
        return current_idx

    def find_a_low(self, current_idx, market_price, last_high):
        market_price_slice = copy.deepcopy(market_price.iloc[current_idx - 1:current_idx + 2])
        market_price_slice.iloc[0] = last_high
        while current_idx < len(market_price) - 1 and not self.is_low(market_price_slice):
            current_idx += 1
            market_price_slice = market_price.iloc[current_idx - 1:current_idx + 2]
        return current_idx

    @staticmethod
    def get_next_element_in_gen(iterable):
        try:
            first = next(iterable)
        except StopIteration:
            return None
        return first

    @property
    def snapshot_generator(self):
        market_structure = MKS.MarketStructure()
        for loc, mks in self.animation_list:
            if len(market_structure) > 0:
                try:
                    i = 0
                    curr_node = market_structure.tail
                    while i != len(market_structure) - 1 - loc:
                        curr_node = curr_node.prev
                        i += 1
                    market_structure.remove(direction="forward", start_node=curr_node)
                except:
                    i = 0
                    curr_node = market_structure.tail
                    while i != len(market_structure) - 1 - loc:
                        curr_node = curr_node.prev
                        i += 1
                    market_structure.remove(direction="forward", start_node=curr_node)

            market_structure.extend(mks)
            yield market_structure


    def initialization(self):
        retry_count = 0

        while retry_count < self.no_of_tries:
            try:
                self.search_for_start_point()

                numpy_data = self.data.reset_index()[['time', 'o', 'h', 'l', 'c']].to_numpy()
                list_of_highs, list_of_lows = find_highs_find_lows(numpy_data[:, 1:].astype(np.float64))
                is_high_last = numpy_data[list_of_highs[-1], 0] >= numpy_data[list_of_lows[-1], 0]
                numpy_highs, numpy_lows, high_time, low_time = efficient_clean_points(
                    highs=numpy_data[list_of_highs, 2].astype(np.float64),
                    lows=numpy_data[list_of_lows, 3].astype(np.float64),
                    high_timestamp=numpy_data[list_of_highs, 0].astype(np.datetime64),
                    low_timestamp=numpy_data[list_of_lows, 0].astype(np.datetime64),
                    high_last=is_high_last)  # make the points bipartite

                self.high_date_index = high_time
                self.low_date_index = low_time
                self.highs = numpy_highs
                self.lows = numpy_lows

                del list_of_highs

                del list_of_lows

                def high_generator():
                    for index, value in zip(self.high_date_index, self.highs):
                        yield MKS.Peak(index, value)

                def low_generator():
                    for index, value in zip(self.low_date_index, self.lows):
                        yield MKS.Trough(index, value)

                self.high_generator = high_generator()
                self.low_generator = low_generator()
                self.market_structure = MKS.MarketStructure()
                if self.high_date_index[0] <= self.low_date_index[0]:
                    for i in range(2):
                        self.market_structure.append(next(self.high_generator))
                        self.market_structure.tail.trend = self.trend
                        self.market_structure.append(next(self.low_generator))
                        self.market_structure.tail.trend = self.trend
                else:
                    for i in range(2):
                        self.market_structure.append(next(self.low_generator))
                        self.market_structure.tail.trend = self.trend
                        self.market_structure.append(next(self.high_generator))
                        self.market_structure.tail.trend = self.trend
                if self.trend == "uptrend":
                    if not self.final_uptrend_validation():
                        raise BadStartPoint
                else:
                    if not self.final_downtrend_validation():
                        raise BadStartPoint

                return self.market_structure
            except BadStartPoint as e:
                retry_count += 1
                if retry_count < self.no_of_tries:
                    self.update_data()
                else:
                    raise
            except Exception as err:
                raise err
