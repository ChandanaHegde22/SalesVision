"""Pick and build an appropriate Plotly chart for a given ExecutionResult.

Rules (mirroring the product spec):
- Revenue/metric over time            -> line chart
- Top-N / ranked dimension             -> horizontal bar chart
- Region-style comparison (2-8 groups) -> vertical bar chart
- Category share of a whole            -> pie chart
- Two numeric columns, many rows       -> scatter (correlation)
- Very small or single-row results     -> no chart, table only
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .safe_executor import ExecutionResult

MIN_ROWS_FOR_CHART = 2


def build_chart(result: ExecutionResult) -> Optional[go.Figure]:
    table = result.table
    if table is None or len(table) < MIN_ROWS_FOR_CHART:
        return None  # never force a chart on trivial results

    op = result.operation

    if op == "trend":
        fig = px.line(table, x="date", y=result.metric, markers=True,
                       title=f"{result.metric.title()} Over Time")
        return fig

    if op == "top_n":
        ordered = table.sort_values(result.metric, ascending=True)
        fig = px.bar(ordered, x=result.metric, y=result.dimension, orientation="h",
                     title=f"Top {result.dimension.replace('_', ' ').title()} by {result.metric.title()}")
        return fig

    if op == "filter_show":
        if len(table) <= 6:
            fig = px.pie(table, names=result.dimension, values=result.metric,
                         title=f"{result.metric.title()} Share by {result.dimension.replace('_', ' ').title()}")
        else:
            fig = px.bar(table, x=result.dimension, y=result.metric,
                         title=f"{result.metric.title()} by {result.dimension.replace('_', ' ').title()}")
        return fig

    if op == "compare":
        fig = px.bar(table, x=result.dimension, y=result.metric, color=result.dimension,
                     title=f"{result.metric.title()} Comparison")
        return fig

    if op in ("growth", "declining"):
        dim_col = result.dimension
        ordered = table.head(15).sort_values("pct_change", ascending=True)
        fig = px.bar(ordered, x="pct_change", y=dim_col, orientation="h",
                     color="pct_change", color_continuous_scale="RdYlGn",
                     title="% Change vs Previous Period")
        return fig

    if op == "correlation":
        metrics = [c for c in table.columns if c != "metric"]
        fig = px.imshow(table[metrics].values, x=metrics, y=table["metric"],
                        color_continuous_scale="RdBu", zmin=-1, zmax=1,
                        title="Metric Correlation Matrix")
        return fig

    if op == "summary":
        return None  # KPI-style table, no single natural chart

    if op == "average_metric":
        return None

    return None
