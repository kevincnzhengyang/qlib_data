# Copyright (c) 2026
# Licensed under the MIT License

"""Operators for the expression system."""

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from typing import Tuple

from .base import Expression, ExpressionOps


np.seterr(invalid="ignore")


#################### Element-Wise Operator ####################
class ElemOperator(ExpressionOps):
    """Element-wise Operator.

    Parameters
    ----------
    feature : Expression
        feature instance
    func : str
        numpy element-wise function name
    """

    def __init__(self, feature, func: str = None):
        self.feature = feature
        self.func = func

    def __str__(self):
        return f"{type(self).__name__}({self.feature})"

    def _load_internal(self, instrument: str, start_index: int, end_index: int, *args) -> pd.Series:
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.func:
            return getattr(np, self.func)(series)
        return series

    def get_longest_back_rolling(self) -> int:
        return self.feature.get_longest_back_rolling()

    def get_extended_window_size(self) -> Tuple[int, int]:
        return self.feature.get_extended_window_size()


class NpElemOperator(ElemOperator):
    """Numpy Element-wise Operator."""

    def _load_internal(self, instrument: str, start_index: int, end_index: int, *args) -> pd.Series:
        series = self.feature.load(instrument, start_index, end_index, *args)
        return getattr(np, self.func)(series)


class Abs(NpElemOperator):
    """Feature Absolute Value."""

    def __init__(self, feature):
        super().__init__(feature, "abs")


class Sign(NpElemOperator):
    """Feature Sign."""

    def __init__(self, feature):
        super().__init__(feature, "sign")

    def _load_internal(self, instrument: str, start_index: int, end_index: int, *args) -> pd.Series:
        series = self.feature.load(instrument, start_index, end_index, *args)
        series = series.astype(np.float32)
        return getattr(np, self.func)(series)


class Log(NpElemOperator):
    """Feature Log."""

    def __init__(self, feature):
        super().__init__(feature, "log")


class Exp(NpElemOperator):
    """Feature Exponential."""

    def __init__(self, feature):
        super().__init__(feature, "exp")


class Sqrt(NpElemOperator):
    """Feature Square Root."""

    def __init__(self, feature):
        super().__init__(feature, "sqrt")


class Not(NpElemOperator):
    """Not Operator."""

    def __init__(self, feature):
        super().__init__(feature, "bitwise_not")


class ChangeInstrument(ElemOperator):
    """Change Instrument Operator.

    In some cases, one may want to change to another instrument when calculating,
    for example, to calculate beta of a stock with respect to a market index.
    """

    def __init__(self, instrument: str, feature):
        self.instrument = instrument
        super().__init__(feature)

    def __str__(self):
        return f"{type(self).__name__}('{self.instrument}', {self.feature})"

    def _load_internal(self, instrument: str, start_index: int, end_index: int, *args) -> pd.Series:
        return self.feature.load(self.instrument, start_index, end_index, *args)


#################### Pair-Wise Operator ####################
class PairOperator(ExpressionOps):
    """Pair-wise operator.

    Parameters
    ----------
    feature_left : Expression
        feature instance or numeric value
    feature_right : Expression
        feature instance or numeric value
    """

    def __init__(self, feature_left, feature_right):
        self.feature_left = feature_left
        self.feature_right = feature_right

    def __str__(self):
        return f"{type(self).__name__}({self.feature_left}, {self.feature_right})"

    def get_longest_back_rolling(self) -> int:
        left_br = self.feature_left.get_longest_back_rolling() if isinstance(self.feature_left, Expression) else 0
        right_br = self.feature_right.get_longest_back_rolling() if isinstance(self.feature_right, Expression) else 0
        return max(left_br, right_br)

    def get_extended_window_size(self) -> Tuple[int, int]:
        if isinstance(self.feature_left, Expression):
            ll, lr = self.feature_left.get_extended_window_size()
        else:
            ll, lr = 0, 0
        if isinstance(self.feature_right, Expression):
            rl, rr = self.feature_right.get_extended_window_size()
        else:
            rl, rr = 0, 0
        return max(ll, rl), max(lr, rr)


class NpPairOperator(PairOperator):
    """Numpy Pair-wise operator."""

    def __init__(self, feature_left, feature_right, func: str):
        self.func = func
        super().__init__(feature_left, feature_right)

    def _load_internal(self, instrument: str, start_index: int, end_index: int, *args) -> pd.Series:
        if isinstance(self.feature_left, Expression):
            series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        else:
            series_left = self.feature_left
        if isinstance(self.feature_right, Expression):
            series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        else:
            series_right = self.feature_right
        return getattr(np, self.func)(series_left, series_right)


class Power(NpPairOperator):
    """Power Operator."""

    def __init__(self, feature_left, feature_right):
        super().__init__(feature_left, feature_right, "power")


class Add(NpPairOperator):
    """Add Operator."""

    def __init__(self, feature_left, feature_right):
        super().__init__(feature_left, feature_right, "add")


class Sub(NpPairOperator):
    """Subtract Operator."""

    def __init__(self, feature_left, feature_right):
        super().__init__(feature_left, feature_right, "subtract")


class Mul(NpPairOperator):
    """Multiply Operator."""

    def __init__(self, feature_left, feature_right):
        super().__init__(feature_left, feature_right, "multiply")


class Div(NpPairOperator):
    """Division Operator."""

    def __init__(self, feature_left, feature_right):
        super().__init__(feature_left, feature_right, "divide")


class Greater(NpPairOperator):
    """Greater Operator - element-wise maximum."""

    def __init__(self, feature_left, feature_right):
        super().__init__(feature_left, feature_right, "maximum")


class Less(NpPairOperator):
    """Less Operator - element-wise minimum."""

    def __init__(self, feature_left, feature_right):
        super().__init__(feature_left, feature_right, "minimum")


class Gt(NpPairOperator):
    """Greater Than Operator."""

    def __init__(self, feature_left, feature_right):
        super().__init__(feature_left, feature_right, "greater")


class Ge(NpPairOperator):
    """Greater Equal Than Operator."""

    def __init__(self, feature_left, feature_right):
        super().__init__(feature_left, feature_right, "greater_equal")


class Lt(NpPairOperator):
    """Less Than Operator."""

    def __init__(self, feature_left, feature_right):
        super().__init__(feature_left, feature_right, "less")


class Le(NpPairOperator):
    """Less Equal Than Operator."""

    def __init__(self, feature_left, feature_right):
        super().__init__(feature_left, feature_right, "less_equal")


class Eq(NpPairOperator):
    """Equal Operator."""

    def __init__(self, feature_left, feature_right):
        super().__init__(feature_left, feature_right, "equal")


class Ne(NpPairOperator):
    """Not Equal Operator."""

    def __init__(self, feature_left, feature_right):
        super().__init__(feature_left, feature_right, "not_equal")


class And(NpPairOperator):
    """And Operator."""

    def __init__(self, feature_left, feature_right):
        super().__init__(feature_left, feature_right, "bitwise_and")


class Or(NpPairOperator):
    """Or Operator."""

    def __init__(self, feature_left, feature_right):
        super().__init__(feature_left, feature_right, "bitwise_or")


#################### Triple-wise Operator ####################
class If(ExpressionOps):
    """If Operator.

    Parameters
    ----------
    condition : Expression
        feature instance with bool values as condition
    feature_left : Expression
        feature instance - value when condition is True
    feature_right : Expression
        feature instance - value when condition is False
    """

    def __init__(self, condition, feature_left, feature_right):
        self.condition = condition
        self.feature_left = feature_left
        self.feature_right = feature_right

    def __str__(self):
        return f"If({self.condition}, {self.feature_left}, {self.feature_right})"

    def _load_internal(self, instrument: str, start_index: int, end_index: int, *args) -> pd.Series:
        series_cond = self.condition.load(instrument, start_index, end_index, *args)
        series_left = self.feature_left.load(instrument, start_index, end_index, *args) if isinstance(self.feature_left, Expression) else self.feature_left
        series_right = self.feature_right.load(instrument, start_index, end_index, *args) if isinstance(self.feature_right, Expression) else self.feature_right
        return pd.Series(np.where(series_cond, series_left, series_right), index=series_cond.index)

    def get_longest_back_rolling(self) -> int:
        left_br = self.feature_left.get_longest_back_rolling() if isinstance(self.feature_left, Expression) else 0
        right_br = self.feature_right.get_longest_back_rolling() if isinstance(self.feature_right, Expression) else 0
        c_br = self.condition.get_longest_back_rolling() if isinstance(self.condition, Expression) else 0
        return max(left_br, right_br, c_br)

    def get_extended_window_size(self) -> Tuple[int, int]:
        def get_ext(expr):
            return expr.get_extended_window_size() if isinstance(expr, Expression) else (0, 0)
        ll, lr = get_ext(self.feature_left)
        rl, rr = get_ext(self.feature_right)
        cl, cr = get_ext(self.condition)
        return max(ll, rl, cl), max(lr, rr, cr)


#################### Rolling Operator ####################
class Rolling(ExpressionOps):
    """Rolling Operator.

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size (0 = expanding, 0 < N < 1 = ewm)
    func : str
        rolling method name
    """

    def __init__(self, feature, N, func: str):
        self.feature = feature
        self.N = N
        self.func = func

    def __str__(self):
        return f"{type(self).__name__}({self.feature}, {self.N})"

    def _load_internal(self, instrument: str, start_index: int, end_index: int, *args) -> pd.Series:
        series = self.feature.load(instrument, start_index, end_index, *args)
        if isinstance(self.N, int) and self.N == 0:
            series = getattr(series.expanding(min_periods=1), self.func)()
        elif isinstance(self.N, float) and 0 < self.N < 1:
            series = series.ewm(alpha=self.N, min_periods=1).mean()
        else:
            series = getattr(series.rolling(self.N, min_periods=1), self.func)()
        return series

    def get_longest_back_rolling(self) -> int:
        if self.N == 0:
            return np.inf
        if 0 < self.N < 1:
            return int(np.log(1e-6) / np.log(1 - self.N))
        return self.feature.get_longest_back_rolling() + self.N - 1

    def get_extended_window_size(self) -> Tuple[int, int]:
        if self.N == 0:
            return self.feature.get_extended_window_size()
        elif 0 < self.N < 1:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            size = int(np.log(1e-6) / np.log(1 - self.N))
            return max(lft_etd + size - 1, lft_etd), rght_etd
        else:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            return max(lft_etd + self.N - 1, lft_etd), rght_etd


class Ref(Rolling):
    """Feature Reference.

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        N = 0, retrieve the first data; N > 0, retrieve data of N periods ago; N < 0, future data

    Returns
    ----------
    Expression
        a feature instance with target reference
    """

    def __init__(self, feature, N):
        super().__init__(feature, N, "ref")

    def _load_internal(self, instrument: str, start_index: int, end_index: int, *args) -> pd.Series:
        series = self.feature.load(instrument, start_index, end_index, *args)
        if series.empty:
            return series
        elif self.N == 0:
            series = pd.Series(series.iloc[0], index=series.index)
        else:
            series = series.shift(self.N)
        return series

    def get_longest_back_rolling(self) -> int:
        if self.N == 0:
            return np.inf
        return self.feature.get_longest_back_rolling() + self.N

    def get_extended_window_size(self) -> Tuple[int, int]:
        if self.N == 0:
            return self.feature.get_extended_window_size()
        else:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            lft_etd = max(lft_etd + self.N, lft_etd)
            rght_etd = max(rght_etd - self.N, rght_etd)
            return lft_etd, rght_etd


class Mean(Rolling):
    """Rolling Mean (MA)."""

    def __init__(self, feature, N):
        super().__init__(feature, N, "mean")


class Sum(Rolling):
    """Rolling Sum."""

    def __init__(self, feature, N):
        super().__init__(feature, N, "sum")


class Std(Rolling):
    """Rolling Std."""

    def __init__(self, feature, N):
        super().__init__(feature, N, "std")


class Var(Rolling):
    """Rolling Variance."""

    def __init__(self, feature, N):
        super().__init__(feature, N, "var")


class Skew(Rolling):
    """Rolling Skewness."""

    def __init__(self, feature, N):
        if N != 0 and N < 3:
            raise ValueError("The rolling window size of Skewness operation should >= 3")
        super().__init__(feature, N, "skew")


class Kurt(Rolling):
    """Rolling Kurtosis."""

    def __init__(self, feature, N):
        if N != 0 and N < 4:
            raise ValueError("The rolling window size of Kurtosis operation should >= 5")
        super().__init__(feature, N, "kurt")


class Max(Rolling):
    """Rolling Max."""

    def __init__(self, feature, N):
        super().__init__(feature, N, "max")


class Min(Rolling):
    """Rolling Min."""

    def __init__(self, feature, N):
        super().__init__(feature, N, "min")


class IdxMax(Rolling):
    """Rolling Max Index."""

    def __init__(self, feature, N):
        super().__init__(feature, N, "idxmax")

    def _load_internal(self, instrument: str, start_index: int, end_index: int, *args) -> pd.Series:
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series.expanding(min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        return series


class IdxMin(Rolling):
    """Rolling Min Index."""

    def __init__(self, feature, N):
        super().__init__(feature, N, "idxmin")

    def _load_internal(self, instrument: str, start_index: int, end_index: int, *args) -> pd.Series:
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series.expanding(min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
        return series


class Corr(Rolling):
    """Rolling Correlation."""

    def __init__(self, feature_left, feature_right, N):
        self.feature_left = feature_left
        self.feature_right = feature_right
        self.N = N

    def __str__(self):
        return f"Corr({self.feature_left}, {self.feature_right}, {self.N})"

    def _load_internal(self, instrument: str, start_index: int, end_index: int, *args) -> pd.Series:
        series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        return series_left.rolling(self.N, min_periods=1).corr(series_right)

    def get_longest_back_rolling(self) -> int:
        left_br = self.feature_left.get_longest_back_rolling() if isinstance(self.feature_left, Expression) else 0
        right_br = self.feature_right.get_longest_back_rolling() if isinstance(self.feature_right, Expression) else 0
        return max(left_br, right_br) + self.N - 1

    def get_extended_window_size(self) -> Tuple[int, int]:
        def get_ext(expr):
            return expr.get_extended_window_size() if isinstance(expr, Expression) else (0, 0)
        ll, lr = get_ext(self.feature_left)
        rl, rr = get_ext(self.feature_right)
        return max(ll, rl) + self.N - 1, max(lr, rr)


# Aliases for compatibility with QLib
RollingMean = Mean
RollingStd = Std
RollingSum = Sum


class Operators:
    """Namespace for operators to support eval-based parsing.

    This class provides a namespace for all operators so that expressions
    like 'Operators.Ref(Feature("close"), -1)' can be evaluated.
    """

    # Element-wise operators
    Abs = Abs
    Sign = Sign
    Log = Log
    Exp = Exp
    Sqrt = Sqrt
    Not = Not
    ChangeInstrument = ChangeInstrument

    # Binary operators
    Add = Add
    Sub = Sub
    Mul = Mul
    Div = Div
    Power = Power
    Greater = Greater
    Less = Less
    Gt = Gt
    Ge = Ge
    Lt = Lt
    Le = Le
    Eq = Eq
    Ne = Ne
    And = And
    Or = Or

    # Triple operators
    If = If

    # Rolling operators
    Ref = Ref
    Mean = Mean
    Sum = Sum
    Std = Std
    Var = Var
    Skew = Skew
    Kurt = Kurt
    Max = Max
    Min = Min
    IdxMax = IdxMax
    IdxMin = IdxMin
    Corr = Corr
    Rolling = Rolling

    # Aliases
    RollingMean = RollingMean
    RollingStd = RollingStd
    RollingSum = RollingSum
