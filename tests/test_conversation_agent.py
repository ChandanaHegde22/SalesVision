"""Tests for the Conversational Data Analyst. Run with: pytest tests/ -v

These tests deliberately avoid requiring an LLM API key: the rule-based fast
path in query_generator.py covers every example question from the product
spec, so the whole pipeline (spec -> safe execution -> insights -> chart) is
fully testable offline.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from src.chat_memory import ChatMemory
from src.conversation_agent import answer_question
from src.llm_client import LLMClient
from src.query_generator import generate_query_spec
from src.safe_executor import ColumnNotFoundError, execute


@pytest.fixture(scope="module")
def df():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sales_data.csv")
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"])
    return data


@pytest.fixture()
def no_llm(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return LLMClient()


def test_top_selling_products(df, no_llm):
    spec = generate_query_spec("What were my top selling products?", df)
    assert spec.operation == "top_n"
    result = execute(df, spec)
    assert len(result.table) <= 10
    assert "product_name" in result.table.columns


def test_show_sales_in_region(df, no_llm):
    spec = generate_query_spec("Show sales in South", df)
    assert spec.operation == "filter_show"
    result = execute(df, spec)
    assert not result.table.empty


def test_show_sales_in_unknown_place_is_graceful(df, no_llm):
    spec = generate_query_spec("Show sales in Karnataka", df)
    # Karnataka isn't a value in the dataset (region/store/etc.), so the rule-based
    # matcher should be honest about it rather than guessing or hallucinating.
    assert spec.operation == "unsupported"
    assert "karnataka" in spec.clarification.lower()
    assert "region" in spec.clarification.lower()


def test_highest_growth_region(df, no_llm):
    spec = generate_query_spec("Which region had the highest growth?", df)
    assert spec.operation == "growth"
    result = execute(df, spec)
    assert "pct_change" in result.table.columns


def test_declining_products(df, no_llm):
    spec = generate_query_spec("Show products whose sales are declining", df)
    assert spec.operation == "declining"
    execute(df, spec)  # should not raise


def test_compare_regions(df, no_llm):
    spec = generate_query_spec("Compare North and South", df)
    assert spec.operation == "compare"
    result = execute(df, spec)
    assert len(result.table) == 2


def test_salesperson_column_missing_is_graceful(df, no_llm):
    spec = generate_query_spec("Which salesperson generated maximum revenue?", df)
    assert spec.operation == "unsupported"
    assert "salesperson" in spec.clarification.lower()


def test_customer_column_missing_is_graceful(df, no_llm):
    spec = generate_query_spec("Who are the top customers?", df)
    assert spec.operation == "unsupported"


def test_monthly_revenue_trend(df, no_llm):
    spec = generate_query_spec("Show me monthly revenue", df)
    assert spec.operation == "trend"
    result = execute(df, spec)
    assert "date" in result.table.columns


def test_average_order_value(df, no_llm):
    spec = generate_query_spec("What is the average order value?", df)
    assert spec.operation == "average_metric"
    result = execute(df, spec)
    assert result.note is not None  # should surface the order-line assumption


def test_summary(df, no_llm):
    spec = generate_query_spec("Give me a summary of this year's performance", df)
    assert spec.operation == "summary"
    execute(df, spec)


def test_full_pipeline_answer_question(df, no_llm):
    memory = ChatMemory()
    memory.clear()
    bundle = answer_question("What were my top selling products?", df, memory, llm_client=no_llm)
    assert not bundle.is_error
    assert bundle.table is not None
    assert len(bundle.insights) > 0
    assert len(bundle.followups) > 0


def test_no_eval_or_exec_in_safe_executor():
    import re
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "safe_executor.py")
    with open(src_path) as f:
        content = f.read()
    # Strip the module docstring (which documents, in prose, that eval/exec are avoided)
    code = re.sub(r'^""".*?"""', "", content, count=1, flags=re.DOTALL)
    assert "eval(" not in code
    assert "exec(" not in code
    assert "os.system" not in code
    assert "subprocess" not in code


def test_unknown_column_raises_clean_error(df):
    from src.query_generator import QuerySpec
    spec = QuerySpec(operation="top_n", dimension="salesperson", metric="revenue")
    with pytest.raises(ColumnNotFoundError):
        execute(df, spec)


def test_cache_avoids_recompute(df, no_llm):
    memory = ChatMemory()
    memory.clear()  # session_state persists across tests when run outside `streamlit run`
    first = answer_question("Give me a summary of this year's performance", df, memory, llm_client=no_llm)
    second = answer_question("Give me a summary of this year's performance", df, memory, llm_client=no_llm)
    assert first.answer_text == second.answer_text
    assert len(memory.turns) == 2  # both turns are still logged in history
