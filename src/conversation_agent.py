"""Top-level orchestrator for the Conversational Data Analyst.

Pipeline for every question:
    1. Check the in-session cache; return the cached bundle if this exact
       question was already asked (no LLM call, no re-execution).
    2. Resolve simple references to the previous turn (e.g. "compare with
       Tamil Nadu" after a question about Karnataka).
    3. Generate a QuerySpec (rule-based fast path, LLM fallback).
    4. If the spec is "unsupported" or execution raises a known, safe error
       (missing column / missing value), return a natural, honest message -
       never a stack trace, never a guess.
    5. Execute the spec (safe_executor), build a chart (chart_generator),
       compute grounded insights (insight_generator).
    6. Phrase a short natural-language answer from those insights (LLM if
       available, otherwise the insights are shown as-is).
    7. Suggest follow-up questions (LLM if available, otherwise static
       templates keyed off the operation).
    8. Record the turn in ChatMemory for future reference resolution and
       caching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from .chart_generator import build_chart
from .chat_memory import ChatMemory, Turn
from .insight_generator import generate_insights
from .llm_client import LLMClient, LLMUnavailableError
from .prompt_templates import (
    FOLLOWUP_SYSTEM_PROMPT,
    FOLLOWUP_USER_TEMPLATE,
    INSIGHT_PHRASING_SYSTEM_PROMPT,
    INSIGHT_PHRASING_USER_TEMPLATE,
)
from .query_generator import QuerySpec, generate_query_spec
from .safe_executor import ColumnNotFoundError, ExecutionResult, ValueNotFoundError, execute

DEFAULT_FOLLOWUPS = {
    "top_n": ["Compare these by region", "Show the trend for the top item", "Which of these are declining?"],
    "filter_show": ["Show the trend over time", "Compare with another region", "Which products drive this?"],
    "trend": ["Break this down by category", "Compare this year vs last year", "Which region drives the trend?"],
    "compare": ["Show the trend for both", "Break each down by product", "Which one is growing faster?"],
    "growth": ["Show which products are declining", "Break this down by category", "Show me a full summary"],
    "declining": ["Show the top performers instead", "Compare with last quarter", "Break down by region"],
    "average_metric": ["Show this trend over time", "Break this down by category", "Compare regions on this metric"],
    "summary": ["Show top products", "Show monthly revenue trend", "Which region is strongest?"],
    "correlation": ["Show top products by revenue", "Show monthly revenue trend"],
}


@dataclass
class AnswerBundle:
    question: str
    answer_text: str
    table: Optional[pd.DataFrame]
    chart: Optional[object]
    insights: List[str] = field(default_factory=list)
    followups: List[str] = field(default_factory=list)
    spec_operation: str = ""
    source: str = ""
    is_error: bool = False


def _resolve_reference(question: str, spec: QuerySpec, memory: ChatMemory) -> QuerySpec:
    """If this looks like a follow-up ("compare with X") reuse the previous turn's dimension/filter."""
    last = memory.last_turn()
    if not last:
        return spec

    q_lower = question.lower().strip()
    is_short_followup = len(q_lower.split()) <= 6
    references_previous = any(w in q_lower for w in ["compare", "that", "this", "it", "with", "also", "vs"])

    if spec.operation == "compare" and (not spec.compare_values or len(spec.compare_values) < 2):
        if last.filters:
            prior_value = last.filters[0]["value"]
            new_values = [prior_value]
            # try to find a second value mentioned in the current question
            for f in spec.filters or []:
                if f["value"].lower() != prior_value.lower():
                    new_values.append(f["value"])
            if len(new_values) < 2:
                # fall back: any dimension-like word in the question that isn't the prior value
                pass
            if len(new_values) >= 2:
                spec.compare_values = new_values
                spec.dimension = spec.dimension or last.filters[0]["column"]

    if is_short_followup and references_previous and not spec.filters and last.filters:
        spec.dimension = spec.dimension or last.dimension
        spec.metric = spec.metric or last.metric

    return spec


def _phrase_answer(question: str, insights: List[str], llm_client: LLMClient) -> str:
    if not insights:
        return "I couldn't derive any insights from this result."
    if not llm_client.is_available():
        return " ".join(insights[:3])
    facts = "\n".join(f"- {i}" for i in insights)
    try:
        user_prompt = INSIGHT_PHRASING_USER_TEMPLATE.format(question=question, facts=facts)
        text = llm_client.complete(INSIGHT_PHRASING_SYSTEM_PROMPT, user_prompt, max_tokens=250)
        return text or " ".join(insights[:3])
    except LLMUnavailableError:
        return " ".join(insights[:3])


def _suggest_followups(question: str, spec: QuerySpec, llm_client: LLMClient) -> List[str]:
    if llm_client.is_available():
        try:
            user_prompt = FOLLOWUP_USER_TEMPLATE.format(
                question=question, operation=spec.operation,
                dimension=spec.dimension, metric=spec.metric,
            )
            text = llm_client.complete(FOLLOWUP_SYSTEM_PROMPT, user_prompt, max_tokens=120)
            lines = [l.strip("- ").strip() for l in text.splitlines() if l.strip()]
            if lines:
                return lines[:4]
        except LLMUnavailableError:
            pass
    return DEFAULT_FOLLOWUPS.get(spec.operation, ["Show me a summary of this year's performance"])


def answer_question(
    question: str,
    df: pd.DataFrame,
    memory: ChatMemory,
    llm_client: Optional[LLMClient] = None,
) -> AnswerBundle:
    """Answer a natural-language question about `df`. Never raises to the caller."""
    llm_client = llm_client or LLMClient()
    question = question.strip()
    if not question:
        return AnswerBundle(question=question, answer_text="Please type a question to get started.",
                             table=None, chart=None, is_error=True)

    cache_key = ChatMemory.normalize_key(question)
    cached = memory.cache_get(cache_key)
    if cached:
        bundle = AnswerBundle(**cached)
        memory.add_turn(Turn(
            question=bundle.question, operation=bundle.spec_operation, dimension=None,
            metric=None, filters=[], answer=bundle.answer_text,
        ))
        return bundle

    history = memory.history_as_text()
    spec = generate_query_spec(question, df, history=history, llm_client=llm_client)
    spec = _resolve_reference(question, spec, memory)

    if spec.operation == "unsupported":
        answer_text = spec.clarification or "I couldn't understand that question."
        bundle = AnswerBundle(
            question=question, answer_text=answer_text, table=None, chart=None,
            insights=[], followups=DEFAULT_FOLLOWUPS["summary"],
            spec_operation="unsupported", source=spec.source, is_error=True,
        )
        memory.add_turn(Turn(question=question, operation="unsupported", dimension=None,
                              metric=None, filters=[], answer=answer_text))
        return bundle

    try:
        result: ExecutionResult = execute(df, spec)
    except (ColumnNotFoundError, ValueNotFoundError) as exc:
        answer_text = str(exc)
        bundle = AnswerBundle(
            question=question, answer_text=answer_text, table=None, chart=None,
            insights=[], followups=DEFAULT_FOLLOWUPS.get(spec.operation, []),
            spec_operation=spec.operation, source=spec.source, is_error=True,
        )
        memory.add_turn(Turn(question=question, operation=spec.operation, dimension=spec.dimension,
                              metric=spec.metric, filters=spec.filters, answer=answer_text))
        return bundle

    insights = generate_insights(result, df)
    chart = build_chart(result)
    answer_text = _phrase_answer(question, insights, llm_client)
    if result.note:
        answer_text = f"{answer_text}\n\n_Note: {result.note}_"
    followups = _suggest_followups(question, spec, llm_client)

    bundle = AnswerBundle(
        question=question,
        answer_text=answer_text,
        table=result.table,
        chart=chart,
        insights=insights,
        followups=followups,
        spec_operation=spec.operation,
        source=spec.source,
        is_error=False,
    )

    memory.add_turn(Turn(
        question=question, operation=spec.operation, dimension=spec.dimension,
        metric=spec.metric, filters=spec.filters, answer=answer_text,
    ))
    memory.cache_set(cache_key, {
        "question": bundle.question, "answer_text": bundle.answer_text,
        "table": bundle.table, "chart": bundle.chart, "insights": bundle.insights,
        "followups": bundle.followups, "spec_operation": bundle.spec_operation,
        "source": bundle.source, "is_error": bundle.is_error,
    })
    return bundle
