"""Execute a validated QuerySpec against a dataframe using only whitelisted,
read-only pandas operations.

Hard safety guarantees provided by this module:
- No eval(), no exec(), no getattr()-based dynamic dispatch of arbitrary methods.
- Every column name is checked against the real dataframe columns before use.
- Every operation is one of a fixed, small set of hand-written functions below;
  there is no code path that runs LLM-authored code.
- The input dataframe is never mutated (all operations work on copies / groupbys
  that return new objects).
- No OS, filesystem, or network calls happen anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .query_generator import DIMENSION_COLUMNS, METRIC_COLUMNS, QuerySpec


class ColumnNotFoundError(Exception):
    """Raised when a spec references a column that doesn't exist in the dataset."""

    def __init__(self, requested: str, available: List[str]):
        self.requested = requested
        self.available = available
        super().__init__(
            f"I couldn't find a '{requested}' column in your dataset. "
            f"Available columns are: {', '.join(available)}."
        )


class ValueNotFoundError(Exception):
    """Raised when a filter value doesn't exist anywhere in the dataset."""

    def __init__(self, requested: str, column_samples: dict):
        self.requested = requested
        message_lines = [f"I couldn't find '{requested}' in your data."]
        for col, values in column_samples.items():
            message_lines.append(f"Available {col} values: {', '.join(map(str, values))}.")
        super().__init__(" ".join(message_lines))


@dataclass
class ExecutionResult:
    table: pd.DataFrame
    operation: str
    dimension: Optional[str]
    metric: Optional[str]
    time_granularity: Optional[str]
    note: Optional[str] = None


def _validate_column(name: Optional[str], df: pd.DataFrame) -> None:
    if name and name not in df.columns:
        raise ColumnNotFoundError(name, sorted(df.columns))


def _apply_filters(df: pd.DataFrame, spec: QuerySpec) -> pd.DataFrame:
    result = df
    for f in spec.filters or []:
        column, value = f.get("column"), f.get("value")
        _validate_column(column, df)
        if column is None:
            continue
        match = result[result[column].astype(str).str.lower() == str(value).lower()]
        if match.empty and not result.empty:
            samples = {c: sorted(df[c].dropna().unique().tolist()) for c in DIMENSION_COLUMNS if c in df.columns}
            raise ValueNotFoundError(str(value), samples)
        result = match
    return result


def _period_column(df: pd.DataFrame, granularity: Optional[str]) -> pd.Series:
    dates = pd.to_datetime(df["date"])
    if granularity == "year":
        return dates.dt.to_period("Y").dt.to_timestamp()
    if granularity == "quarter":
        return dates.dt.to_period("Q").dt.to_timestamp()
    if granularity == "day":
        return dates.dt.normalize()
    return dates.dt.to_period("M").dt.to_timestamp()  # default: month


def execute(df: pd.DataFrame, spec: QuerySpec) -> ExecutionResult:
    """Run the QuerySpec's operation. Only whitelisted operations are dispatched."""
    metric = spec.metric or "revenue"
    _validate_column(spec.dimension, df)
    _validate_column(metric, df)
    if metric not in METRIC_COLUMNS:
        raise ColumnNotFoundError(metric, METRIC_COLUMNS)

    working = _apply_filters(df, spec)

    if spec.operation == "top_n":
        dimension = spec.dimension or "product_name"
        _validate_column(dimension, df)
        grouped = (
            working.groupby(dimension, as_index=False)
            .agg(**{metric: (metric, "sum"), "quantity": ("quantity", "sum")})
            .sort_values(metric, ascending=(spec.sort == "asc"))
        )
        limit = spec.limit or 10
        return ExecutionResult(grouped.head(limit), spec.operation, dimension, metric, None)

    if spec.operation == "filter_show":
        dimension = spec.dimension or (spec.filters[0]["column"] if spec.filters else "region")
        _validate_column(dimension, df)
        grouped = (
            working.groupby(dimension, as_index=False)
            .agg(**{metric: (metric, "sum"), "quantity": ("quantity", "sum")})
            .sort_values(metric, ascending=False)
        )
        return ExecutionResult(grouped, spec.operation, dimension, metric, None)

    if spec.operation == "trend":
        granularity = spec.time_granularity or "month"
        temp = working.copy()
        temp["period"] = _period_column(temp, granularity)
        grouped = temp.groupby("period", as_index=False)[metric].sum().sort_values("period")
        grouped = grouped.rename(columns={"period": "date"})
        return ExecutionResult(grouped, spec.operation, None, metric, granularity)

    if spec.operation == "compare":
        dimension = spec.dimension or "region"
        _validate_column(dimension, df)
        values = spec.compare_values or []
        if len(values) < 2:
            raise ValueNotFoundError(
                "two values to compare",
                {dimension: sorted(df[dimension].dropna().unique().tolist())} if dimension in df.columns else {},
            )
        subset = df[df[dimension].astype(str).str.lower().isin([v.lower() for v in values])]
        grouped = (
            subset.groupby(dimension, as_index=False)
            .agg(**{
                metric: (metric, "sum"),
                "quantity": ("quantity", "sum"),
                "profit": ("profit", "sum"),
            })
            .sort_values(metric, ascending=False)
        )
        return ExecutionResult(grouped, spec.operation, dimension, metric, None)

    if spec.operation in ("growth", "declining"):
        dimension = spec.dimension or "product_name"
        _validate_column(dimension, df)
        temp = working.copy()
        temp["period"] = _period_column(temp, spec.time_granularity or "month")
        pivot = temp.groupby([dimension, "period"], as_index=False)[metric].sum()
        periods = sorted(pivot["period"].unique())
        if len(periods) < 2:
            note = "Not enough time periods in the (filtered) data to compute growth."
            return ExecutionResult(pivot, spec.operation, dimension, metric, spec.time_granularity, note)

        latest, previous = periods[-1], periods[-2]
        latest_vals = pivot[pivot["period"] == latest].set_index(dimension)[metric]
        prev_vals = pivot[pivot["period"] == previous].set_index(dimension)[metric]
        combined = pd.DataFrame({"latest": latest_vals, "previous": prev_vals}).fillna(0)
        combined["change"] = combined["latest"] - combined["previous"]
        combined["pct_change"] = (
            (combined["change"] / combined["previous"].replace(0, pd.NA)) * 100
        ).fillna(0)
        combined = combined.reset_index().rename(columns={dimension: dimension})

        ascending = spec.operation == "declining"
        combined = combined.sort_values("pct_change", ascending=ascending)
        if spec.operation == "declining":
            combined = combined[combined["change"] < 0]
        return ExecutionResult(combined, spec.operation, dimension, metric, spec.time_granularity)

    if spec.operation == "average_metric":
        avg_value = working[metric].mean() if not working.empty else 0.0
        table = pd.DataFrame([{f"average_{metric}": round(float(avg_value), 2), "rows": len(working)}])
        note = None
        if metric == "revenue":
            note = (
                "No order-id column exists in this dataset, so this treats each sales-record "
                "row as one order line when computing the average."
            )
        return ExecutionResult(table, spec.operation, None, metric, None, note)

    if spec.operation == "summary":
        by_category = df.groupby("category", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        by_region = df.groupby("region", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        table = pd.DataFrame({
            "metric": ["total_revenue", "total_profit", "total_quantity", "top_category", "top_region"],
            "value": [
                round(float(df["revenue"].sum()), 2),
                round(float(df["profit"].sum()), 2),
                int(df["quantity"].sum()),
                by_category.iloc[0]["category"] if not by_category.empty else "N/A",
                by_region.iloc[0]["region"] if not by_region.empty else "N/A",
            ],
        })
        return ExecutionResult(table, spec.operation, None, "revenue", None)

    if spec.operation == "correlation":
        numeric = df[[c for c in METRIC_COLUMNS if c in df.columns]]
        corr = numeric.corr(numeric_only=True).round(3).reset_index().rename(columns={"index": "metric"})
        return ExecutionResult(corr, spec.operation, None, metric, None)

    raise ColumnNotFoundError(spec.operation, sorted(VALID_OPERATIONS_LIST()))


def VALID_OPERATIONS_LIST() -> List[str]:
    from .query_generator import VALID_OPERATIONS
    return list(VALID_OPERATIONS)
