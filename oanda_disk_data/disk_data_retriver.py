import pathlib
import pprint
from collections import namedtuple
from typing import List
import datetime as dt
import pandas as pd
import timeit

data_path = pathlib.Path("F:/PSQL_CSV_DATA")

TimeRange = namedtuple("TimeRange", ["start", "end"])


def _retrieve_list_of_all_files_in_dir(instrument: str, granularity: str):
    file_path = data_path / instrument / granularity
    return list(file_path.glob("*"))


def _format_for_binary_search(file_name: str):
    file_name = file_name.replace(".csv", "")
    return TimeRange(*[int(num) for num in file_name.split("_")[-2:]])


def _binary_search_for_file(formatted_files: List[TimeRange], target: int):
    start = 0
    end = len(formatted_files) - 1
    while start <= end:
        mid = (start + end) // 2
        if formatted_files[mid].start <= target <= formatted_files[mid].end:
            return mid
        if formatted_files[mid].end < target:
            start = mid + 1
        else:
            end = mid

    return -1


def retrieve_data_from_disk(instrument: str, granularity: str, start_date: dt.datetime, end_date: dt.datetime):
    files = _retrieve_list_of_all_files_in_dir(instrument=instrument, granularity=granularity)
    files_formatted_for_search = [_format_for_binary_search(file_name=file.name) for file in files]
    files_formatted_for_search, files = zip(*sorted(zip(files_formatted_for_search, files), key=lambda x: x[0].start))

    target_test_start = int(start_date.strftime("%Y%m%d%H%M%S"))
    target_test_end = int(end_date.strftime("%Y%m%d%H%M%S"))
    start_index = _binary_search_for_file(formatted_files=files_formatted_for_search, target=target_test_start)
    end_index = _binary_search_for_file(formatted_files=files_formatted_for_search, target=target_test_end)

    if start_index == -1 or end_index == -1:
        raise FileNotFoundError("Data not found")

    result = pd.concat(
        [pd.read_csv(files[i], parse_dates=True, index_col=0) for i in range(start_index, end_index + 1)])
    result = result.tz_localize(None)
    result = result[(result.index >= pd.Timestamp(start_date)) & (result.index <= pd.Timestamp(end_date))]
    if len(result) == 0:
        raise FileNotFoundError("Data not found")
    return result


if __name__ == "__main__":
    currency = "EUR_GBP"
    gran = "H1"
    start = dt.datetime(2021, 10, 27, 9, 0, 0)
    end = dt.datetime(2021, 10, 28, 13, 0, 0)

    print(retrieve_data_from_disk(currency, gran, start, end))
