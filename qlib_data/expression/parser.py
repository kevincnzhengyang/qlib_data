# Copyright (c) 2026
# Licensed under the MIT License

"""Expression parser for converting field strings to Expression objects."""

import re
from typing import Union, List

from .base import Expression, Feature, PFeature
from . import ops
from ..logging import get_logger


logger = get_logger(__name__)


def parse_expression(field: Union[str, Expression]) -> Union[Expression, str]:
    """Alias for parse_field for compatibility."""
    return parse_field(field)


def parse_field(field: Union[str, Expression]) -> Union[Expression, str]:
    """
    Parse a field string into an Expression object.

    Supported patterns:
    - $close -> Feature("close")
    - $$fwd_eps -> PFeature("fwd_eps")
    - Ref($close, -1) -> Operators.Ref(Feature("close"), -1)
    - RollingMean($close, 5) -> Operators.Mean(Feature("close"), 5)
    - $close + $open -> Add(Feature("close"), Feature("open"))
    - Ref($close, -1) / $close - 1 -> Sub(Div(Ref(Feature("close"), 1), Feature("close")), 1)

    Parameters
    ----------
    field : str or Expression
        Field expression string or already an Expression object

    Returns
    -------
    Expression or str
        Parsed expression object, or original value if not a string
    """
    if isinstance(field, Expression):
        return field

    if not isinstance(field, str):
        field = str(field)

    logger.debug("parser.parse_start", field=field)

    # Chinese punctuation support
    chinese_punctuation_regex = r"\u3001\uff1a\uff08\uff09"

    # Pattern replacements - order matters! $$ before $, Feature/PFeature before generic func(
    patterns = [
        # $$name -> PFeature("name")  (must be before $name)
        (rf"\$\$([\w{chinese_punctuation_regex}]+)", r'PFeature("\1")'),
        # $name -> Feature("name")
        (rf"\$([\w{chinese_punctuation_regex}]+)", r'Feature("\1")'),
    ]

    for pattern, replacement in patterns:
        field = re.sub(pattern, replacement, field)

    # Now add Operators. prefix for operators (but not for Feature/PFeature which are already converted)
    # Match function calls that are NOT already prefixed with Operators.
    def replace_operator(match):
        func_name = match.group(1).strip()
        if func_name in ("Feature", "PFeature"):
            return match.group(0)  # Keep as is
        return f"Operators.{func_name}("

    field = re.sub(r"(\w+)\s*\(", replace_operator, field)
    logger.debug("parser.after_rewrite", rewritten=field)

    # Evaluate the expression with operators and base classes in scope
    eval_globals = {
        "Operators": ops.Operators,
        "Feature": Feature,
        "PFeature": PFeature,
    }
    try:
        expr = eval(field, eval_globals)  # pylint: disable=eval-used
    except Exception as exc:
        logger.error("parser.eval_failed", field=field, error=str(exc), exc_info=True)
        raise

    logger.debug("parser.parse_done", field=field, expression_type=type(expr).__name__)
    return expr


def parse_fields(fields: Union[str, List[str], List[Expression]]) -> List[Expression]:
    """
    Parse a list of field strings into a list of Expression objects.

    Parameters
    ----------
    fields : list or str
        List of field expressions or single field expression

    Returns
    -------
    list
        List of parsed Expression objects
    """
    if isinstance(fields, str):
        fields = [fields]

    result = [parse_field(f) for f in fields]
    logger.debug("parser.parse_fields_done", count=len(result), types=[type(e).__name__ for e in result])
    return result
