"""Translate a natural-language question into a structured, validated QuerySpec.

Design decision: instead of asking an LLM to write raw pandas/Python code (which
would then need sandboxing to be safe), the LLM is only ever asked to produce a
small JSON object describing *which* whitelisted operation to run. safe_executor.py
is the only module that touches the dataframe, and it only accepts QuerySpec
objects with columns/values it can verify exist. This makes "arbitrary code
execution" structurally impossible, not just discouraged by a prompt.

Two tiers are used:
1. A fast, deterministic regex/keyword matcher covers the common question
   shapes (top N, filter by dimension value, trend, compare, growth, decline,
   average, summary). No LLM call, no cost, works even with no API key.
2. If the fast path doesn't confidently match, the question is sent to the
   LLM (see llm_client.py) with the real schema and sample values, and the
   JSON it returns is parsed into the same QuerySpec.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .llm_client import LLMClient, LLMUnavailableError
from .prompt_templates import SPEC_SYSTEM_PROMPT, SPEC_USER_TEMPLATE

VALID_OPERATIONS = {
    "top_n", "filter_show", "trend", "compare", "growth",
    "declining", "average_metric", "summary", "correlation", "unsupported",
}

DIMENSION_COLUMNS = ["product_name", "category", "region", "store"]
METRIC_COLUMNS = ["revenue", "profit", "quantity", "price", "discount"]

# Maps the way people talk about a dimension to the real column name.
# A value of None means "people ask for this, but the dataset has no such column" -
# this is intentional so we can give the exact graceful error the spec asked for,
# e.g. "salesperson" / "customer" / "state" are not present in this dataset.
DIMENSION_SYNONYMS: Dict[str, Optional[str]] = {
    "product": "product_name", "products": "product_name", "item": "product_name",
    "category": "category", "categories": "category",
    "region": "region", "regions": "region",
    "state": "region", "states": "region",
    "store": "store", "stores": "store", "outlet": "store",
    "customer": None, "customers": None, "client": None, "clients": None,
    "salesperson": None, "sales person": None, "sales rep": None, "rep": None,
    "employee": None,
}

METRIC_SYNONYMS: Dict[str, Optional[str]] = {
    "revenue": "revenue", "sales": "revenue", "turnover": "revenue",
    "profit": "profit", "margin": "profit", "profit margin": None,
    "quantity": "quantity", "units": "quantity", "volume": "quantity",
    "price": "price", "discount": "discount",
    "order value": "revenue", "average order value": "revenue", "aov": "revenue",
}


@dataclass
class QuerySpec:
    operation: str
    dimension: Optional[str] = None
    metric: Optional[str] = None
    filters: List[Dict[str, str]] = field(default_factory=list)
    compare_values: Optional[List[str]] = None
    time_granularity: Optional[str] = None
    limit: Optional[int] = None
    sort: Optional[str] = None
    clarification: Optional[str] = None
    source: str = "rule_based"  # "rule_based" or "llm", useful for debugging/UI


class SchemaContext:
    """Precomputed schema info shared by the rule-based matcher and the LLM prompt."""

    def __init__(self, df: pd.DataFrame):
        self.columns = list(df.columns)
        self.dimension_values: Dict[str, List[str]] = {
            col: sorted(df[col].dropna().unique().tolist())
            for col in DIMENSION_COLUMNS
            if col in df.columns
        }

    def find_value_column(self, raw_value: str) -> Optional[str]:
        """Find which dimension column contains a given value (case-insensitive)."""
        needle = raw_value.strip().lower()
        for col, values in self.dimension_values.items():
            for v in values:
                if str(v).strip().lower() == needle:
                    return col
        return None

    def all_known_values(self) -> Dict[str, List[str]]:
        return self.dimension_values

    def schema_description(self) -> str:
        return ", ".join(self.columns)


def _extract_limit(question: str, default: int = 10) -> int:
    match = re.search(r"\btop\s+(\d+)\b", question, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return default


def _find_dimension(question: str) -> Optional[str]:
    q = question.lower()
    for phrase, column in sorted(DIMENSION_SYNONYMS.items(), key=lambda kv: -len(kv[0])):
        if phrase in q:
            return column if column else "__unsupported__" + phrase
    return None


def _find_metric(question: str) -> Optional[str]:
    q = question.lower()
    for phrase, column in sorted(METRIC_SYNONYMS.items(), key=lambda kv: -len(kv[0])):
        if phrase in q:
            return column if column else "__unsupported__" + phrase
    return None


def _find_filter_value(question: str, schema: SchemaContext) -> Optional[Dict[str, str]]:
    """Look for a known dimension value (e.g. a region or product name) mentioned in the text."""
    for col, values in schema.all_known_values().items():
        for v in values:
            if re.search(rf"\b{re.escape(str(v))}\b", question, re.IGNORECASE):
                return {"column": col, "value": v}
    return None


def _rule_based_match(question: str, schema: SchemaContext) -> Optional[QuerySpec]:
    q = question.lower().strip()

    dimension = _find_dimension(q)
    metric = _find_metric(q)
    filter_hit = _find_filter_value(question, schema)

    # A dimension the dataset genuinely doesn't have (salesperson, customer, ...) was
    # mentioned - be honest about it immediately, regardless of which verb/phrasing
    # ("top", "maximum", "who are", ...) surrounds it.
    if dimension and dimension.startswith("__unsupported__"):
        return _unsupported_spec(dimension)
    if metric and str(metric).startswith("__unsupported__"):
        return _unsupported_spec(metric)

    # "Why did revenue decrease / drop last month" -> explain a change (growth op, negative focus)
    if re.search(r"\bwhy\b.*\b(decrease|drop|decline|fall|down)\b", q):
        return QuerySpec(
            operation="growth",
            dimension="region",
            metric=metric if metric and not metric.startswith("__unsupported__") else "revenue",
            time_granularity="month",
            sort="asc",
            source="rule_based",
        )

    # Declining products/items
    if re.search(r"\bdeclin", q) or re.search(r"\bunderperform", q):
        if dimension and dimension.startswith("__unsupported__"):
            return _unsupported_spec(dimension)
        return QuerySpec(
            operation="declining",
            dimension=dimension or "product_name",
            metric=metric if metric and not str(metric).startswith("__unsupported__") else "revenue",
            time_granularity="month",
            source="rule_based",
        )

    # Compare X and Y
    compare_match = re.search(r"\bcompare\s+([a-zA-Z ]+?)\s+(?:and|vs\.?|versus)\s+([a-zA-Z ]+)", q)
    if compare_match or "compare" in q:
        values = []
        for col, vals in schema.all_known_values().items():
            for v in vals:
                if re.search(rf"\b{re.escape(str(v))}\b", question, re.IGNORECASE):
                    values.append((col, v))
        if len(values) >= 2:
            col = values[0][0]
            return QuerySpec(
                operation="compare",
                dimension=col,
                metric=metric if metric and not str(metric).startswith("__unsupported__") else "revenue",
                compare_values=[values[0][1], values[1][1]],
                source="rule_based",
            )

    # Highest growth
    if "growth" in q and dimension:
        if dimension.startswith("__unsupported__"):
            return _unsupported_spec(dimension)
        return QuerySpec(
            operation="growth",
            dimension=dimension,
            metric=metric if metric and not str(metric).startswith("__unsupported__") else "revenue",
            time_granularity="month",
            sort="desc",
            source="rule_based",
        )

    # Top N
    if re.search(r"\btop\b", q) or re.search(r"\bhighest\b", q) or re.search(r"\bbest[- ]?selling\b", q):
        if dimension and dimension.startswith("__unsupported__"):
            return _unsupported_spec(dimension)
        return QuerySpec(
            operation="top_n",
            dimension=dimension or "product_name",
            metric=metric if metric and not str(metric).startswith("__unsupported__") else "revenue",
            limit=_extract_limit(q),
            sort="desc",
            source="rule_based",
        )

    # Average order value / average metric
    if "average" in q or "avg" in q or "aov" in q:
        if metric and str(metric).startswith("__unsupported__"):
            return _unsupported_spec(metric)
        return QuerySpec(
            operation="average_metric",
            metric=metric if metric else "revenue",
            source="rule_based",
        )

    # Monthly / trend over time
    if re.search(r"\bmonthly\b|\bmonth\b|\btrend\b|\bover time\b", q):
        return QuerySpec(
            operation="trend",
            metric=metric if metric and not str(metric).startswith("__unsupported__") else "revenue",
            time_granularity="month" if "month" in q or "monthly" in q else "day",
            filters=[filter_hit] if filter_hit else [],
            source="rule_based",
        )

    # "Show sales in <place>" / filtered lookup
    if filter_hit and re.search(r"\bshow\b|\bsales in\b|\brevenue in\b", q):
        return QuerySpec(
            operation="filter_show",
            dimension=filter_hit["column"],
            metric=metric if metric and not str(metric).startswith("__unsupported__") else "revenue",
            filters=[filter_hit],
            source="rule_based",
        )

    # A dimension value was mentioned but no clear verb - still treat as filter_show
    if filter_hit:
        return QuerySpec(
            operation="filter_show",
            dimension=filter_hit["column"],
            metric=metric if metric and not str(metric).startswith("__unsupported__") else "revenue",
            filters=[filter_hit],
            source="rule_based",
        )

    # Summary / overall performance
    if re.search(r"\bsummary\b|\boverview\b|\bperformance\b|\bhow (are|is) we doing\b", q):
        return QuerySpec(operation="summary", source="rule_based")

    # "Show sales/revenue in <place>" where <place> isn't a known value anywhere
    # (e.g. an Indian state name when the dataset only has regions) - be honest
    # about it instead of falling through to a generic "couldn't understand" message.
    place_match = re.search(r"\b(?:sales|revenue)\s+in\s+([a-zA-Z ]+?)\s*[\.\?]?$", question, re.IGNORECASE)
    if place_match:
        requested = place_match.group(1).strip()
        return QuerySpec(
            operation="unsupported",
            clarification=(
                f"I couldn't find '{requested}' in your data. "
                + " ".join(
                    f"Available {col} values: {', '.join(map(str, vals))}."
                    for col, vals in schema.all_known_values().items()
                )
            ),
            source="rule_based",
        )

    return None


def _unsupported_spec(marker: str) -> QuerySpec:
    requested = marker.replace("__unsupported__", "") or "that field"
    return QuerySpec(
        operation="unsupported",
        clarification=(
            f"I couldn't find a '{requested.strip()}' column in your dataset."
        ),
        source="rule_based",
    )


def _parse_llm_json(raw_text: str) -> QuerySpec:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    data = json.loads(cleaned)

    operation = data.get("operation")
    if operation not in VALID_OPERATIONS:
        return QuerySpec(
            operation="unsupported",
            clarification="I wasn't able to confidently map that question to the dataset.",
            source="llm",
        )

    return QuerySpec(
        operation=operation,
        dimension=data.get("dimension"),
        metric=data.get("metric"),
        filters=data.get("filters") or [],
        compare_values=data.get("compare_values"),
        time_granularity=data.get("time_granularity"),
        limit=data.get("limit"),
        sort=data.get("sort"),
        clarification=data.get("clarification"),
        source="llm",
    )


def generate_query_spec(
    question: str,
    df: pd.DataFrame,
    history: Optional[List[str]] = None,
    llm_client: Optional[LLMClient] = None,
) -> QuerySpec:
    """Turn a natural-language question into a QuerySpec.

    Tries the deterministic rule-based matcher first; only calls the LLM if
    that doesn't produce a confident match and an LLM is configured.
    """
    schema = SchemaContext(df)

    rule_spec = _rule_based_match(question, schema)
    if rule_spec is not None:
        return rule_spec

    client = llm_client or LLMClient()
    if not client.is_available():
        return QuerySpec(
            operation="unsupported",
            clarification=(
                "I couldn't map that question to a supported analysis, and no LLM is "
                "configured for free-form understanding. Try rephrasing using a product, "
                "region, category, store name, or a metric like revenue, profit, or quantity."
            ),
            source="rule_based",
        )

    history_text = "\n".join(history or []) or "(no previous turns)"
    user_prompt = SPEC_USER_TEMPLATE.format(
        schema=schema.schema_description(),
        sample_values=json.dumps(schema.all_known_values(), indent=2),
        history=history_text,
        question=question,
    )
    try:
        raw = client.complete(SPEC_SYSTEM_PROMPT, user_prompt, max_tokens=500)
        return _parse_llm_json(raw)
    except (LLMUnavailableError, json.JSONDecodeError, ValueError):
        return QuerySpec(
            operation="unsupported",
            clarification=(
                "I couldn't confidently understand that question. Try asking about a "
                "specific product, region, category, store, or metric."
            ),
            source="llm",
        )
