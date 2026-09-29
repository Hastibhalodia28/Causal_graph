"""expr_compiler.py — safe AST-based expression compiler for KPI expressions.

Turns the machine-readable `expression` field in kpis.jsonld into a pandas
Series / scalar on a per-grain-group basis.

Supported grammar
-----------------
Row functions (applied per row, BEFORE groupby):
    days(a - b)             timedelta in days (float)
    hours(a - b)            timedelta in hours (float)
    date(col)               date-normalise a datetime (truncate to midnight)
    month(col)              integer month
    year(col)               integer year
    lower(col)              str.lower()
    notblank(col)           1.0 where col is non-empty, else 0.0
    isnull(col)             1.0 where col is NaN/None, else 0.0
    if_else(cond, a, b)     ternary (all series)
    coalesce(a, b)          first non-null

Table references:
    table.column            resolved via DatasetSpec.tables mapping

Aggregate functions (applied per grain group after a groupby):
    sum(expr)   mean(expr)  min(expr)   max(expr)
    count(expr) nunique(expr) any(expr) all(expr) mode(expr)

Comparison / arithmetic:
    ==  !=  <=  >=  <  >   +  -  *  /  and  or  not

KPI references:
    Bare KPI_ID (no table prefix) is looked up in the compiled panel frame.

Output type
-----------
compile_expression(expr, tables, spec, panel, grain) returns a pd.Series
indexed by the grain value(s), ready to be merged onto the panel.
"""
from __future__ import annotations

import ast
import operator
from typing import Any, Callable

import numpy as np
import pandas as pd

from spec_loader import DatasetSpec


# ---------------------------------------------------------------------------
# Whitelisted node visitor
# ---------------------------------------------------------------------------

_SAFE_OPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.And: lambda a, b: a & b,
    ast.Or: lambda a, b: a | b,
    ast.Not: operator.not_,
}

# Aggregate functions: (pandas groupby method, or callable(series))
_AGG_FUNCS = {"sum", "mean", "min", "max", "count", "nunique", "any", "all", "mode"}

# Row-level functions
_ROW_FUNCS = {"days", "hours", "date", "month", "year", "lower", "notblank", "isnull",
               "if_else", "coalesce"}


class _ExprCompiler(ast.NodeVisitor):
    """Walks an AST and evaluates the expression against the provided data.

    `_eval` returns either:
      - a pd.Series (indexed by the original DataFrame index), OR
      - a scalar
    Aggregates return a pd.Series indexed by the grain key(s).
    """

    def __init__(
        self,
        tables: dict[str, pd.DataFrame],  # alias -> DataFrame (already loaded, not grouped)
        grain_col: str,                    # single grain column name
        panel: pd.DataFrame,               # partially-built panel (for KPI refs)
        spec: DatasetSpec,
    ):
        self._tables = tables
        self._grain = grain_col
        self._panel = panel
        self._spec = spec
        # reverse map: csv filename -> alias
        self._file_to_alias: dict[str, str] = {v: k for k, v in spec.tables.items()}

    # --- helpers ---

    def _series_or_scalar(self, node: ast.AST) -> Any:
        return self.visit(node)

    def _to_numeric(self, s: Any) -> Any:
        if isinstance(s, pd.Series):
            # Never force-convert datetime/timedelta Series to numeric — that
            # destroys them (pd.to_numeric doesn't understand datetime64) and
            # breaks later .dt accessor calls in days()/hours().
            if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
                return s
            return pd.to_numeric(s, errors="coerce")
        return s

    def _grain_groupby(self, df: pd.DataFrame):
        return df.groupby(self._grain, sort=False)

    # --- visitors ---

    def visit_Constant(self, node: ast.Constant):
        return node.value

    def visit_Name(self, node: ast.Name):
        name = node.id
        # Check if it's a KPI already on the panel
        if name in self._panel.columns:
            return self._panel[name]
        # Check if it's a context var already on the panel
        if name in self._panel.columns:
            return self._panel[name]
        raise NameError(f"Unknown name {name!r} — not a table alias, KPI, or context var")

    def visit_Attribute(self, node: ast.Attribute):
        # table.column
        alias = node.value.id if isinstance(node.value, ast.Name) else None
        col = node.attr
        if alias and alias in self._tables:
            df = self._tables[alias]
            if col not in df.columns:
                raise KeyError(f"Column {col!r} not found in table {alias!r}")
            return df[col]
        raise NameError(f"Unknown table alias {alias!r}")

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operand = self._series_or_scalar(node.operand)
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary op {type(node.op).__name__}")
        return op(operand)

    def visit_BinOp(self, node: ast.BinOp):
        left = self._to_numeric(self._series_or_scalar(node.left))
        right = self._to_numeric(self._series_or_scalar(node.right))
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported binary op {type(node.op).__name__}")
        return op(left, right)

    def visit_Compare(self, node: ast.Compare):
        left = self._series_or_scalar(node.left)
        result = None
        for op_node, comparator in zip(node.ops, node.comparators):
            right = self._series_or_scalar(comparator)
            op = _SAFE_OPS.get(type(op_node))
            if op is None:
                raise ValueError(f"Unsupported compare op {type(op_node).__name__}")
            # Align grain-indexed Series before comparing
            if isinstance(left, pd.Series) and isinstance(right, pd.Series):
                if left.index.name == self._grain and right.index.name == self._grain:
                    left, right = left.align(right, join="left")
                elif len(left) != len(right):
                    # one is row-level, the other is grain-aggregated — merge on grain
                    grain_col = self._panel[self._grain]
                    if len(right) < len(left):
                        right = grain_col.map(right)
                    else:
                        left = grain_col.map(left)
            cmp = op(left, right)
            result = cmp if result is None else (result & cmp)
            left = right
        return result

    def visit_BoolOp(self, node: ast.BoolOp):
        def _to_bool(x):
            if isinstance(x, pd.Series):
                return x.astype(bool)
            return bool(x)

        values = [_to_bool(self._series_or_scalar(v)) for v in node.values]
        op = _SAFE_OPS.get(type(node.op))
        result = values[0]
        for v in values[1:]:
            result = op(result, v)
        return result

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls (no dotted names) allowed")
        fname = node.func.id

        # --- row-level functions ---
        if fname == "days":
            delta = self._series_or_scalar(node.args[0])
            if isinstance(delta, pd.Series):
                return delta.dt.total_seconds() / 86400.0
            return float(delta.total_seconds()) / 86400.0

        if fname == "hours":
            delta = self._series_or_scalar(node.args[0])
            if isinstance(delta, pd.Series):
                return delta.dt.total_seconds() / 3600.0
            return float(delta.total_seconds()) / 3600.0

        if fname == "date":
            s = self._series_or_scalar(node.args[0])
            if isinstance(s, pd.Series):
                return pd.to_datetime(s, errors="coerce").dt.normalize()
            return pd.Timestamp(s).normalize()

        if fname == "month":
            s = self._series_or_scalar(node.args[0])
            if isinstance(s, pd.Series):
                return pd.to_datetime(s, errors="coerce").dt.month.astype(float)
            return float(pd.Timestamp(s).month)

        if fname == "year":
            s = self._series_or_scalar(node.args[0])
            if isinstance(s, pd.Series):
                return pd.to_datetime(s, errors="coerce").dt.year.astype(float)
            return float(pd.Timestamp(s).year)

        if fname == "lower":
            s = self._series_or_scalar(node.args[0])
            if isinstance(s, pd.Series):
                return s.astype(str).str.lower()
            return str(s).lower()

        if fname == "notblank":
            s = self._series_or_scalar(node.args[0])
            if isinstance(s, pd.Series):
                return s.fillna("").astype(str).str.strip().ne("").astype(float)
            return float(bool(str(s or "").strip()))

        if fname == "isnull":
            s = self._series_or_scalar(node.args[0])
            if isinstance(s, pd.Series):
                return s.isna().astype(float)
            return float(s is None or (isinstance(s, float) and np.isnan(s)))

        if fname == "if_else":
            cond = self._series_or_scalar(node.args[0])
            a = self._series_or_scalar(node.args[1])
            b = self._series_or_scalar(node.args[2])
            if isinstance(cond, pd.Series):
                return pd.Series(np.where(cond, a, b), index=cond.index)
            return a if cond else b

        if fname == "coalesce":
            a = self._series_or_scalar(node.args[0])
            b = self._series_or_scalar(node.args[1])
            if isinstance(a, pd.Series):
                return a.fillna(b)
            return a if a is not None else b

        # --- aggregate functions ---
        if fname in _AGG_FUNCS:
            inner = self._series_or_scalar(node.args[0])
            if not isinstance(inner, pd.Series):
                raise ValueError(f"Aggregate {fname}() requires a Series (table.column)")
            # determine which table this series came from to get the grain col
            grain_s = self._grain_col_for_series(inner)
            g = inner.groupby(grain_s, sort=False)
            if fname == "sum":
                return g.sum()
            if fname == "mean":
                return g.mean()
            if fname == "min":
                return g.min()
            if fname == "max":
                return g.max()
            if fname == "count":
                return g.count()
            if fname == "nunique":
                return g.nunique()
            if fname == "any":
                # inner may be bool or 0/1 numeric
                return g.max()
            if fname == "all":
                return g.min()
            if fname == "mode":
                return g.agg(lambda s: s.mode().iloc[0] if len(s) > 0 else np.nan)

        raise NameError(f"Unknown function {fname!r}")

    def _grain_col_for_series(self, s: pd.Series) -> pd.Series:
        """Return a parallel Series of the grain value for each row of s."""
        grain = self._grain
        # Find which table this series belongs to by matching its index length
        for alias, df in self._tables.items():
            if grain in df.columns and len(df) == len(s):
                return df[grain]
        # Fall back: assume same index as panel
        if grain in self._panel.columns and len(self._panel) == len(s):
            return self._panel[grain]
        raise ValueError(
            f"Cannot determine grain column {grain!r} for the aggregate — "
            "make sure the table alias in the expression matches a loaded table."
        )

    def generic_visit(self, node: ast.AST):
        raise ValueError(f"Unsupported AST node {type(node).__name__}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_expression(
    expression: str,
    tables: dict[str, pd.DataFrame],
    spec: DatasetSpec,
    panel: pd.DataFrame,
    grain: str,
) -> pd.Series:
    """Evaluate `expression` and return a Series indexed by grain value.

    The returned Series has one row per unique grain value (e.g. order_id),
    named after the expression string for traceability. Callers merge it onto
    the panel using `panel.merge(result, on=grain, how='left')`.

    Parameters
    ----------
    expression : str
        Machine-readable expression from kpis.jsonld.
    tables : dict[str, pd.DataFrame]
        Table alias -> DataFrame (grain column must be present in every table
        that the expression references with an aggregate).
    spec : DatasetSpec
        The full spec (used for table alias resolution).
    panel : pd.DataFrame
        Partially-built panel (used to resolve bare KPI-ID references).
    grain : str
        The single grain column name (e.g. "order_id").
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Cannot parse expression {expression!r}: {exc}") from exc

    compiler = _ExprCompiler(tables=tables, grain_col=grain, panel=panel, spec=spec)
    result = compiler.visit(tree.body)

    # If the result is already grouped (pd.Series with grain as index), return it.
    # If it's a row-level Series (same length as a table), group it.
    if isinstance(result, pd.Series):
        # Check if result index looks like it could be the grain key (already grouped)
        # by comparing to panel's grain column unique values
        grain_vals = panel[grain].values if grain in panel.columns else None
        if grain_vals is not None and len(result) != len(panel):
            # Result has different length — assume it's already per-grain aggregated
            return result
        if grain_vals is not None and len(result) == len(panel):
            # Same length as panel — it's a row-level result; take it as-is since
            # boolean / comparison results on panel columns are already row-aligned
            return result
        return result
    # Scalar: broadcast to panel
    if grain in panel.columns:
        s = pd.Series(result, index=panel[grain].unique())
        s.index.name = grain
        return s
    return pd.Series([result])