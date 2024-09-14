import datetime as dt
from MarketStructure import data_retrieval as dr
from MarketStructure.market_data_initializer import MarketDataInitializer, BadStartPoint
from MarketStructure import MRP
from MarketStructure.MRP import save_state_in_animation_list
import MarketStructure.visualization_tools as vt
from MarketStructure.node import Trough
from collections import namedtuple


def animate(mkr_rev_ptrn: MRP.MRP):
    x = mkr_rev_ptrn.mdi
    vt.animate_market_structure(vt.prepare_oanda_data(x.data), x.granularity, x.no_of_seconds_per_period,
                                x.animation_list)


AnimationListTuple = namedtuple("AnimationListTuple", ["loc", "market_structure"])


def create_market_structure(mkr_rev_ptrn, currency, start, end, granularity,
                            animate_mks=False) -> MarketDataInitializer:
    max_retries = 10
    retry_count = 0
    while retry_count < max_retries:
        try:
            mkr_rev_ptrn.mdi.reinitialize(currency, start, end, granularity, animate_mks)
            mkr_rev_ptrn.calculate_market_structure()
            if animate_mks:
                animate(mkr_rev_ptrn)
            return mkr_rev_ptrn.mdi
        except BadStartPoint as e:
            retry_count += 1
            if retry_count < max_retries:
                start = mkr_rev_ptrn.mdi.start_date
                start -= dt.timedelta(
                    seconds=mkr_rev_ptrn.mdi.default_periods * mkr_rev_ptrn.mdi.no_of_seconds_per_period)
            else:
                if animate_mks:
                    animate(mkr_rev_ptrn)
                raise
        except Exception as e:
            if animate_mks:
                animate(mkr_rev_ptrn)
            raise


@save_state_in_animation_list
def append_to_mks(mkr_rev_ptrn, node):
    mkr_rev_ptrn.mdi.market_structure.append(node)


# def original_highs_and_lows(mkr_rev_ptrn, currency, start, end, granularity,
#                             animate_mks=False) -> MarketDataInitializer:
#     mkr_rev_ptrn.mdi.reinitialize(currency, start, end, granularity, animate_mks)
#     mkr_rev_ptrn.mdi.market_structure.changed_nodes.clear()
#     for node in mkr_rev_ptrn.mdi.market_structure:
#         node.reset_observer_state()
#         node.value = node.value
#     while True:
#         if isinstance(mkr_rev_ptrn.mdi.market_structure.tail, Trough):
#             next_node = mkr_rev_ptrn.mdi.get_next_element_in_gen(mkr_rev_ptrn.mdi.high_generator)
#         else:
#             next_node = mkr_rev_ptrn.mdi.get_next_element_in_gen(mkr_rev_ptrn.mdi.low_generator)
#         if next_node is None:
#             break
#         append_to_mks(mkr_rev_ptrn, next_node)
#     animate(mkr_rev_ptrn)
#     return mkr_rev_ptrn.mdi


start = dt.datetime(2021, 4, 26, 13, 0, 0)
end = dt.datetime(2022, 6, 15, 15, 0, 0)
currency = "EUR_USD"
granularity = "D"

data_retriever = dr.LocalDataRetriever()
mdi = MarketDataInitializer(data_retriever)

mrp = MRP.CmsMrp(mdi)
create_market_structure(mrp, currency, start, end, granularity, animate_mks=True)

# mdi2 = MarketDataInitializer(data_retriever)
# mrp2 = MRP.CmsMrp(mdi2)
# original_highs_and_lows(mrp2, currency, mrp.mdi.start_date, mrp.mdi.end_date, granularity, animate_mks=False)
