from abc import ABC, abstractmethod
# import tpqoa
import pandas as pd
from oanda_disk_data import disk_data_retriver
import typing as tp
import datetime as dt


class DataRetrieverException(Exception):
    def __init__(self, message="Data could not be retrieved"):
        self.message = message

    def __str__(self):
        return self.message


class DataRetriever(ABC):
    """
    Abstract class for data retrieval.

    Due to the way the algorithm works, there is no guarantee that the start date requested will suitable for its anaylsis.
    Therefore, the algorithm amy need to look for a better start date. This is why the update_data method is needed.

    """
    @abstractmethod
    def get_data(self, instrument: str, start_date: dt.datetime, end_date: dt.datetime, granularity: str) -> pd.DataFrame:
        raise NotImplementedError("Not Implemented")

    @abstractmethod
    def update_data(self, instrument: str, new_start_date: dt.datetime, start_date: dt.datetime, granularity: str, old_data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Not Implemented")

# Doesn't work because of the import of tpqoa
# class OandaDataRetriever(DataRetriever):
#
#     def __init__(self, oanda_config_file="oanda.cfg"):
#         self.oanda: tpqoa.tpqoa = tpqoa.tpqoa(oanda_config_file)
#
#     def get_data(self, instrument, start_date, end_date, granularity):
#         try:
#             data = self.oanda.get_history(instrument=instrument,
#                                           start=start_date,
#                                           end=end_date,
#                                           granularity=granularity,
#                                           price="M",
#                                           localize=False)
#             data.drop_duplicates(inplace=True)
#             return data
#         except Exception as e:
#             raise DataRetrieverException(f"Data could not be retrieved: {e}")
#
#     def update_data(self, instrument, new_start_date, start_date, granularity, old_data):
#         try:
#             new_data = self.get_data(instrument, new_start_date, start_date, granularity)
#             new_data = pd.concat([new_data.iloc[:-1], old_data])
#             # assert no duplicates indices
#             return new_data
#         except DataRetrieverException as e:
#             raise


class LocalDataRetriever(DataRetriever):
    """
    This is a class that retrieves data from the local disk.
    Obviously, this is
    """

    def __init__(self):
        pass

    def get_data(self, instrument, start_date, end_date, granularity):
        try:
            data = disk_data_retriver.retrieve_data_from_disk(instrument=instrument,
                                                              granularity=granularity,
                                                              start_date=start_date,
                                                              end_date=end_date)
            data.drop_duplicates(inplace=True)
            return data
        except Exception as e:
            raise DataRetrieverException(f"Data could not be retrieved: {e}")

    def update_data(self, instrument, new_start_date, start_date, granularity, old_data):
        try:
            new_data = self.get_data(instrument, new_start_date, start_date, granularity)
            new_data = pd.concat([new_data.iloc[:-1], old_data])
            # assert no duplicates indices
            return new_data
        except DataRetrieverException as e:
            raise
