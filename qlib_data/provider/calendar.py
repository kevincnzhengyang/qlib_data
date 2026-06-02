# Copyright (c) 2026
# Licensed under the MIT License

"""Calendar provider for trading day management."""

import bisect
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from ..config import C
from ..logging import get_logger
from ..storage.bin_storage import CalendarStorage


logger = get_logger(__name__)


class CalendarProvider:
    """Calendar provider for accessing trading days."""

    def __init__(self, provider_uri: str = None):
        self._provider_uri = provider_uri
        self._calendar_cache = {}

    @property
    def provider_uri(self) -> str:
        return self._provider_uri or C.provider_uri

    def _get_storage(self, freq: str, future: bool = False) -> CalendarStorage:
        """Get calendar storage instance."""
        key = (freq, future)
        if key not in self._calendar_cache:
            logger.debug("calendar.storage_create", freq=freq, future=future)
            self._calendar_cache[key] = CalendarStorage(freq=freq, future=future, provider_uri=self.provider_uri)
        return self._calendar_cache[key]

    def calendar(self, start_time: str = None, end_time: str = None, freq: str = "day", future: bool = False) -> List[pd.Timestamp]:
        """Get calendar of certain market in given time range.

        Parameters
        ----------
        start_time : str, optional
            Start of the time range
        end_time : str, optional
            End of the time range
        freq : str
            Time frequency (e.g., "day", "1min")
        future : bool
            Whether including future trading day

        Returns
        -------
        list
            Calendar list
        """
        storage = self._get_storage(freq, future)
        calendar_list = storage.read()

        if not calendar_list:
            logger.warning("calendar.empty", freq=freq, future=future)
            return []

        if start_time is None:
            start_time = calendar_list[0]
        else:
            start_time = pd.Timestamp(start_time)
            if start_time > calendar_list[-1]:
                logger.info("calendar.range_out_of_bounds", start=str(start_time), end=str(calendar_list[-1]))
                return []

        if end_time is None:
            end_time = calendar_list[-1]
        else:
            end_time = pd.Timestamp(end_time)
            if end_time < calendar_list[0]:
                logger.info("calendar.range_out_of_bounds", start=str(calendar_list[0]), end=str(end_time))
                return []

        _, _, si, ei = self.locate_index(start_time, end_time, freq, future)
        result = calendar_list[si:ei + 1]
        logger.debug("calendar.range_returned", start=str(result[0]) if result else None, end=str(result[-1]) if result else None, count=len(result))
        return result

    def locate_index(
        self,
        start_time: Union[pd.Timestamp, str],
        end_time: Union[pd.Timestamp, str],
        freq: str,
        future: bool = False,
    ) -> tuple:
        """Locate the start time index and end time index in calendar.

        Parameters
        ----------
        start_time : pd.Timestamp or str
            Start time
        end_time : pd.Timestamp or str
            End time
        freq : str
            Frequency
        future : bool
            Include future

        Returns
        -------
        tuple
            (real_start_time, real_end_time, start_index, end_index)
        """
        storage = self._get_storage(freq, future)
        calendar_list = storage.read()

        if not calendar_list:
            logger.error("calendar.locate_empty", freq=freq, future=future)
            raise ValueError("Calendar is empty")

        start_time = pd.Timestamp(start_time)
        end_time = pd.Timestamp(end_time)

        # Find start index
        if start_time not in calendar_list:
            try:
                start_time = calendar_list[bisect.bisect_left(calendar_list, start_time)]
            except IndexError:
                logger.error("calendar.start_in_future", start=str(start_time))
                raise IndexError(f"`start_time` uses a future date: {start_time}")

        start_index = calendar_list.index(start_time)

        # Find end index
        if end_time not in calendar_list:
            end_time = calendar_list[bisect.bisect_right(calendar_list, end_time) - 1]
        end_index = calendar_list.index(end_time)

        logger.debug(
            "calendar.locate_index",
            freq=freq,
            start=str(start_time),
            end=str(end_time),
            start_index=start_index,
            end_index=end_index,
        )
        return start_time, end_time, start_index, end_index

    def time_to_index(self, time_point: Union[pd.Timestamp, str], freq: str = "day") -> int:
        """Convert time point to calendar index."""
        storage = self._get_storage(freq, False)
        calendar_list = storage.read()
        time_point = pd.Timestamp(time_point)

        if time_point not in calendar_list:
            idx = bisect.bisect_left(calendar_list, time_point)
            if idx >= len(calendar_list):
                logger.error("calendar.time_beyond", time=str(time_point))
                raise ValueError(f"Time {time_point} is beyond calendar range")
            time_point = calendar_list[idx]
        index = calendar_list.index(time_point)
        logger.debug("calendar.time_to_index", time=str(time_point), index=index)
        return index

    def index_to_time(self, index: int, freq: str = "day") -> pd.Timestamp:
        """Convert calendar index to time point."""
        storage = self._get_storage(freq, False)
        calendar_list = storage.read()
        ts = calendar_list[index]
        logger.debug("calendar.index_to_time", index=index, time=str(ts))
        return ts


# Global instance
_calendar_provider = None


def get_calendar_provider() -> CalendarProvider:
    """Get the global calendar provider instance."""
    global _calendar_provider
    if _calendar_provider is None:
        logger.debug("calendar.provider_singleton_create")
        _calendar_provider = CalendarProvider()
    return _calendar_provider
