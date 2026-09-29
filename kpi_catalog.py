"""kpi_catalog.py — generic KPI catalog loader and expression compiler bridge.

Replaces the Olist-specific kpi_list.py.  All knowledge about which KPIs
exist, their expressions, context variables, and grain is read from the
DatasetSpec produced by spec_loader.discover_metadata().

Public API
----------
compile_all_kpis(tables, spec, panel, grain)
    → dict[kpi_id, pd.Series]  (one series per KPI, indexed by grain value)

compile_context_vars(tables, spec, panel, grain)
    → dict[var_id, pd.Series]  (one series per context variable)

build_panel(tables, spec, grain)
    → pd.DataFrame   (full panel: grain column + all KPI cols + all context cols)
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from spec_loader import DatasetSpec, KPISpec, ContextVar
from expr_compiler import compile_expression

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _eval_one(
    expr_id: str,
    expression: str,
    tables: dict[str, pd.DataFrame],
    spec: DatasetSpec,
    panel: pd.DataFrame,
    grain: str,
) -> pd.Series | None:
    """Evaluate a single expression and return a grain-indexed Series.

    Returns None on error (logs a warning).
    """
    if not expression:
        log.warning("No expression for %s — skipped", expr_id)
        return None
    try:
        result = compile_expression(
            expression=expression,
            tables=tables,
            spec=spec,
            panel=panel,
            grain=grain,
        )
        if isinstance(result, pd.Series):
            result.name = expr_id
        return result
    except Exception as exc:
        log.warning("Failed to compile %s = %r: %s", expr_id, expression, exc)
        return None


def _merge_series(
    panel: pd.DataFrame,
    series: pd.Series,
    col_name: str,
    grain: str,
) -> pd.DataFrame:
    """Merge a grain-indexed Series onto the panel (left join on grain)."""
    if series is None:
        return panel
    s = series.rename(col_name)
    if isinstance(s.index, pd.MultiIndex):
        # Shouldn't happen with single-grain, but guard anyway
        s = s.reset_index()
        panel = panel.merge(s, on=grain, how="left")
    elif s.index.name == grain:
        df = s.reset_index()
        panel = panel.merge(df, on=grain, how="left")
    elif len(s) == len(panel):
        # Row-level result aligned to panel rows
        panel = panel.copy()
        panel[col_name] = s.values
    else:
        # Assume index already is grain values
        df = s.rename_axis(grain).reset_index()
        panel = panel.merge(df, on=grain, how="left")
    return panel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_context_vars(
    tables: dict[str, pd.DataFrame],
    spec: DatasetSpec,
    panel: pd.DataFrame,
    grain: str,
) -> dict[str, pd.Series | None]:
    """Evaluate every contextVariable expression.

    Context variables with dependencies on other context variables are
    evaluated in dependency order: if an expression references a var_id that
    appears earlier in the list, that var is already on the panel.

    Returns a dict var_id -> Series (or None on failure).
    """
    results: dict[str, pd.Series | None] = {}
    # panel is mutated as we go so later vars can reference earlier ones
    working_panel = panel.copy()

    for cv in spec.context_vars:
        s = _eval_one(cv.var_id, cv.expression, tables, spec, working_panel, grain)
        results[cv.var_id] = s
        if s is not None and cv.var_id not in working_panel.columns:
            working_panel = _merge_series(working_panel, s, cv.var_id, grain)

    return results


def compile_all_kpis(
    tables: dict[str, pd.DataFrame],
    spec: DatasetSpec,
    panel: pd.DataFrame,
    grain: str,
) -> dict[str, pd.Series | None]:
    """Evaluate every KPI expression in dependency order.

    Child KPIs that reference parent KPIs (by bare KPI_ID in the expression)
    are deferred until the parent is available on the panel.

    Returns a dict kpi_id -> Series (or None on failure).
    """
    results: dict[str, pd.Series | None] = {}
    working_panel = panel.copy()

    # Topological sort: KPIs without a parent first, then children
    # (spec.kpis is assumed to be in declaration order; we do one pass
    #  and allow a second pass for any that failed due to missing deps)
    pending = list(spec.kpis)
    max_passes = len(pending) + 1
    pass_num = 0

    while pending and pass_num < max_passes:
        pass_num += 1
        still_pending: list[KPISpec] = []
        for kpi in pending:
            # Check if all referenced KPI IDs (parent, referenced in expr) are ready
            if kpi.parent and kpi.parent not in working_panel.columns:
                still_pending.append(kpi)
                continue
            s = _eval_one(kpi.kpi_id, kpi.expression, tables, spec, working_panel, grain)
            results[kpi.kpi_id] = s
            if s is not None and kpi.kpi_id not in working_panel.columns:
                working_panel = _merge_series(working_panel, s, kpi.kpi_id, grain)
        pending = still_pending

    if pending:
        for kpi in pending:
            log.warning(
                "KPI %s could not be compiled (unresolved dependency on %s)",
                kpi.kpi_id, kpi.parent,
            )
            results[kpi.kpi_id] = None

    return results


def build_panel(
    tables: dict[str, pd.DataFrame],
    spec: DatasetSpec,
    grain: str,
) -> pd.DataFrame:
    """Build the full panel DataFrame.

    Steps:
      1. Start from the unique grain values of the grain-bearing table.
      2. Compile and merge context variables.
      3. Compile and merge KPIs (in dependency order).

    Returns a DataFrame with one row per grain value and columns:
        [grain, *context_var_ids, *kpi_ids]
    """
    # Step 1 — seed panel from grain values
    grain_col = _find_grain_series(tables, spec, grain)
    panel = pd.DataFrame({grain: grain_col.unique()})
    panel = panel.sort_values(grain).reset_index(drop=True)

    # Step 2 — context variables (may reference table.column directly)
    ctx_results = compile_context_vars(tables, spec, panel, grain)
    for var_id, s in ctx_results.items():
        if s is not None:
            panel = _merge_series(panel, s, var_id, grain)

    # Step 3 — KPIs (may reference context vars already on panel)
    kpi_results = compile_all_kpis(tables, spec, panel, grain)
    for kpi_id, s in kpi_results.items():
        if s is not None:
            panel = _merge_series(panel, s, kpi_id, grain)

    return panel


def _find_grain_series(
    tables: dict[str, pd.DataFrame],
    spec: DatasetSpec,
    grain: str,
) -> pd.Series:
    """Return a Series of all grain values from the table that has the grain column.

    Priority: the table whose CSV filename contains the grain column
    (checked via spec.tables values and the actual DataFrames).
    Falls back to any table that has the column.
    """
    # Prefer the table explicitly listed first in spec.tables
    for alias, _csv in spec.tables.items():
        df = tables.get(alias)
        if df is not None and grain in df.columns:
            return df[grain]

    raise KeyError(
        f"Grain column {grain!r} not found in any loaded table. "
        f"Available tables: {list(tables)}"
    )


# ---------------------------------------------------------------------------
# Convenience: load tables from spec + dataset directory
# ---------------------------------------------------------------------------

def load_tables(
    dataset_path,
    spec: DatasetSpec,
    dtype: dict | None = None,
    parse_dates: list[str] | None = None,
    date_cols_by_table: dict[str, list[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load every CSV listed in spec.tables from dataset_path.

    Parameters
    ----------
    dataset_path : path-like
        Directory that contains the CSV files.
    spec : DatasetSpec
        Provides alias -> filename mapping.
    dtype : optional
        Passed to pd.read_csv for all tables (column dtype overrides).
    parse_dates : optional
        Column names to parse as datetime (applied to all tables that have them).
    date_cols_by_table : optional
        dict alias -> [col, ...] of per-table date columns; preferred over
        parse_dates when both are supplied.

    Returns
    -------
    dict alias -> pd.DataFrame

    Notes
    -----
    After loading, if a table does NOT contain the grain column but another
    loaded table shares a common join key with it AND that other table does
    contain the grain column, the grain column is propagated automatically.
    This handles the common case where e.g. a "products" table joins to
    "items" via "product_id", and items has "order_id" (the grain).
    """
    from pathlib import Path

    dataset_path = Path(dataset_path)
    tables: dict[str, pd.DataFrame] = {}

    for alias, csv_name in spec.tables.items():
        csv_path = dataset_path / csv_name
        if not csv_path.exists():
            log.warning("Table %r: file %s not found — skipped", alias, csv_path)
            continue

        # Explicit per-table date columns (from Croissant or caller)
        explicit_dates: list[str] = []
        if date_cols_by_table:
            explicit_dates = list(date_cols_by_table.get(alias) or [])
        elif parse_dates:
            explicit_dates = list(parse_dates)

        try:
            df = pd.read_csv(csv_path, dtype=dtype, low_memory=False)

            # Always apply heuristic: any column whose name contains a date keyword
            _DATE_KEYWORDS = ("date", "time", "timestamp", "_at")
            heuristic = [
                c for c in df.columns
                if any(kw in c.lower() for kw in _DATE_KEYWORDS)
            ]
            # Union of explicit + heuristic, preserving order
            cols_to_parse = list(dict.fromkeys(explicit_dates + heuristic))

            for col in cols_to_parse:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            tables[alias] = df
            log.debug("Loaded %s → %s (%d rows, %d cols)", alias, csv_name, len(df), len(df.columns))
        except Exception as exc:
            log.warning("Failed to load %s (%s): %s", alias, csv_path, exc)

    # Propagate grain column into tables that lack it via shared join keys
    grain = spec.grain[0] if spec.grain else None
    if grain:
        _propagate_grain(tables, grain)

    return tables


def _propagate_grain(tables: dict[str, pd.DataFrame], grain: str) -> None:
    """For each table missing the grain column, try to inherit it from another table.

    Works iteratively: tables that already have grain can donate it to others.
    A table B gets the grain from table A when:
      - A has the grain column
      - A and B share at least one non-grain column (the join key)
    The grain is added as the per-row grain value for each row of B.
    """
    changed = True
    while changed:
        changed = False
        grain_tables = {a: df for a, df in tables.items() if grain in df.columns}
        for alias, df in list(tables.items()):
            if grain in df.columns:
                continue
            # Try to find a donor table with a shared join key
            for donor_alias, donor_df in grain_tables.items():
                shared_keys = [
                    c for c in df.columns
                    if c in donor_df.columns and c != grain
                ]
                if not shared_keys:
                    continue
                join_key = shared_keys[0]  # use first shared key
                grain_map = (
                    donor_df[[join_key, grain]]
                    .dropna(subset=[join_key])
                    .drop_duplicates(subset=[join_key])
                )
                merged = df.merge(grain_map, on=join_key, how="left")
                tables[alias] = merged
                log.debug(
                    "Propagated grain %r into table %r via join key %r (from %r)",
                    grain, alias, join_key, donor_alias,
                )
                changed = True
                break


def infer_date_columns(spec: DatasetSpec, croissant_spec=None) -> dict[str, list[str]]:
    """Infer which columns should be parsed as datetimes.

    Uses the Croissant column dtype map if available (sc:Date -> parse),
    otherwise falls back to heuristic column-name patterns.
    """
    date_cols: dict[str, list[str]] = {}

    if croissant_spec is not None:
        # Use Croissant dtype annotations
        for (csv_name, col), dtype in croissant_spec.col_dtype.items():
            dtype_lower = str(dtype).lower()
            if "date" in dtype_lower or "datetime" in dtype_lower:
                # Find alias for csv_name
                for alias, fname in spec.tables.items():
                    if fname == csv_name:
                        date_cols.setdefault(alias, []).append(col)
    else:
        # Heuristic: columns whose names contain date/time keywords
        _DATE_KEYWORDS = ("date", "time", "timestamp", "at", "_dt")
        for alias in spec.tables:
            pass  # Will be populated after loading (can't know cols before reading CSV)

    return date_cols