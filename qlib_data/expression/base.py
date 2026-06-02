# Copyright (c) 2026
# Licensed under the MIT License

"""Base classes for the expression system."""

from __future__ import division
from __future__ import print_function

import abc
from typing import Tuple

import pandas as pd


class Expression(abc.ABC):
    """
    Expression base class.

    Expression is designed to handle the calculation of data with the format below:
    data with two dimensions for each instrument:
    - feature
    - time: it could be observation time or period time
    """

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return str(self)

    def __gt__(self, other):
        from .ops import Gt
        return Gt(self, other)

    def __ge__(self, other):
        from .ops import Ge
        return Ge(self, other)

    def __lt__(self, other):
        from .ops import Lt
        return Lt(self, other)

    def __le__(self, other):
        from .ops import Le
        return Le(self, other)

    def __eq__(self, other):
        from .ops import Eq
        return Eq(self, other)

    def __ne__(self, other):
        from .ops import Ne
        return Ne(self, other)

    def __add__(self, other):
        from .ops import Add
        return Add(self, other)

    def __radd__(self, other):
        from .ops import Add
        return Add(other, self)

    def __sub__(self, other):
        from .ops import Sub
        return Sub(self, other)

    def __rsub__(self, other):
        from .ops import Sub
        return Sub(other, self)

    def __mul__(self, other):
        from .ops import Mul
        return Mul(self, other)

    def __rmul__(self, other):
        from .ops import Mul
        return Mul(self, other)

    def __div__(self, other):
        from .ops import Div
        return Div(self, other)

    def __rdiv__(self, other):
        from .ops import Div
        return Div(other, self)

    def __truediv__(self, other):
        from .ops import Div
        return Div(self, other)

    def __rtruediv__(self, other):
        from .ops import Div
        return Div(other, self)

    def __pow__(self, other):
        from .ops import Power
        return Power(self, other)

    def __rpow__(self, other):
        from .ops import Power
        return Power(other, self)

    def load(self, instrument: str, start_index: int, end_index: int, *args) -> pd.Series:
        """
        Load feature data.

        This function is responsible for loading feature/expression based on the expression engine.

        Parameters
        ----------
        instrument : str
            instrument code
        start_index : int
            feature start index [in calendar]
        end_index : int
            feature end index [in calendar]
        *args : tuple
            Additional arguments (e.g., freq)

        Returns
        -------
        pd.Series
            feature series with calendar index
        """
        if start_index is not None and end_index is not None and start_index > end_index:
            raise ValueError(f"Invalid index range: {start_index} {end_index}")
        return self._load_internal(instrument, start_index, end_index, *args)

    @abc.abstractmethod
    def _load_internal(self, instrument: str, start_index: int, end_index: int, *args) -> pd.Series:
        """Internal load implementation - must be implemented by subclasses."""
        raise NotImplementedError("This function must be implemented in your newly defined feature")

    @abc.abstractmethod
    def get_longest_back_rolling(self) -> int:
        """Get the longest length of historical data the feature has accessed."""
        raise NotImplementedError("This function must be implemented in your newly defined feature")

    @abc.abstractmethod
    def get_extended_window_size(self) -> Tuple[int, int]:
        """
        Get extended window size for calculation range.

        For to calculate this Operator in range[start_index, end_index]
        We have to get the *leaf feature* in range[start_index - lft_etd, end_index + rght_etd].

        Returns
        -------
        Tuple[int, int]
            (lft_etd, rght_etd)
        """
        raise NotImplementedError("This function must be implemented in your newly defined feature")


class ExpressionOps(Expression):
    """Operator Expression - This kind of feature will use operator for feature construction on the fly."""


class Feature(Expression):
    """Static Expression - This kind of feature will load data from provider."""

    def __init__(self, name: str = None):
        if name:
            self._name = name
        else:
            self._name = type(self).__name__

    def __str__(self):
        return "$" + self._name

    def _load_internal(self, instrument: str, start_index: int, end_index: int, freq: str) -> pd.Series:
        """Load data from FeatureProvider."""
        from ..provider.feature import get_feature_provider
        provider = get_feature_provider()
        return provider.feature(instrument, str(self), start_index, end_index, freq)

    def get_longest_back_rolling(self) -> int:
        return 0

    def get_extended_window_size(self) -> Tuple[int, int]:
        return 0, 0


class PFeature(Feature):
    """Period Feature for Point-in-Time data."""

    def __str__(self):
        return "$$" + self._name

    def _load_internal(self, instrument: str, start_index: int, end_index: int, cur_time: pd.Timestamp, period: int = None) -> pd.Series:
        """Load period feature data."""
        from ..provider.feature import PITFeatureProvider
        return PITFeatureProvider.period_feature(instrument, str(self), start_index, end_index, cur_time, period)
