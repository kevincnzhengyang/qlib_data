# Copyright (c) 2026
# Licensed under the MIT License

"""Expression system for feature calculation."""

from .base import Expression, Feature, PFeature, ExpressionOps
from .ops import Operators
from .parser import parse_expression, parse_field, parse_fields

# Import operators into namespace for eval-based parsing
from . import ops

__all__ = [
    "Expression",
    "Feature",
    "PFeature",
    "ExpressionOps",
    "Operators",
    "parse_expression",
    "parse_field",
    "parse_fields",
]
