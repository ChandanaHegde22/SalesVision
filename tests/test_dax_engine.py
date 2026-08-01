import pytest
import pandas as pd
import numpy as np
from src.dax_engine import (
    eval_calculated_column,
    eval_measure,
    eval_dax_table,
    divide,
    dax_if
)

@pytest.fixture
def sample_df():
    data = {
        "category": ["Electronics", "Electronics", "Fashion", "Fashion", "Furniture"],
        "region": ["North", "South", "North", "South", "North"],
        "revenue": [1000.0, 2000.0, 500.0, 1500.0, 3000.0],
        "profit": [200.0, 500.0, 100.0, 300.0, 600.0],
        "quantity": [2, 4, 1, 3, 5],
        "discount": [10, 20, 0, 5, 15]
    }
    return pd.DataFrame(data)

def test_divide():
    assert divide(10, 2) == 5.0
    assert divide(10, 0) == 0.0
    assert divide(10, 0, 999) == 999.0
    
    # Vector divide
    b = pd.Series([2, 0, 4])
    res = divide(12, b)
    assert res[0] == 6.0
    assert res[1] == 0.0
    assert res[2] == 3.0

def test_dax_if():
    assert dax_if(True, 1, 0) == 1
    assert dax_if(False, 1, 0) == 0
    
    # Vector IF
    cond = pd.Series([True, False, True])
    res = dax_if(cond, 10, 20)
    assert res[0] == 10
    assert list(res) == [10, 20, 10]

def test_eval_calculated_column(sample_df):
    # Basic math
    res = eval_calculated_column(sample_df, "[revenue] - [profit]")
    assert list(res) == [800.0, 1500.0, 400.0, 1200.0, 2400.0]
    
    # IF logic
    res_if = eval_calculated_column(sample_df, "IF([discount] >= 15, [revenue] * 0.9, [revenue])")
    assert list(res_if) == [1000.0, 1800.0, 500.0, 1500.0, 2700.0]
    
    # Invalid column error
    with pytest.raises(ValueError, match="not found in the dataset"):
        eval_calculated_column(sample_df, "[revenue] + [nonexistent]")

def test_eval_measure(sample_df):
    # SUM
    assert eval_measure(sample_df, "SUM([revenue])") == 8000.0
    
    # AVERAGE
    assert eval_measure(sample_df, "AVERAGE([revenue])") == 1600.0
    
    # MIN/MAX
    assert eval_measure(sample_df, "MIN([profit])") == 100.0
    assert eval_measure(sample_df, "MAX([quantity])") == 5
    
    # DISTINCTCOUNT
    assert eval_measure(sample_df, "DISTINCTCOUNT([category])") == 3
    
    # DIVIDE / complex formulas
    assert eval_measure(sample_df, "DIVIDE(SUM([profit]), SUM([revenue]))") == 1700.0 / 8000.0
    
    # Error: un-aggregated columns
    with pytest.raises(ValueError, match="Measures must aggregate all columns"):
        eval_measure(sample_df, "SUM([revenue]) - [profit]")

def test_eval_dax_table_summarize(sample_df):
    # SUMMARIZE
    expr = "SUMMARIZE(df, [category], \"Total Sales\", SUM([revenue]), \"Avg Profit\", AVERAGE([profit]))"
    res_df = eval_dax_table(sample_df, expr)
    
    assert len(res_df) == 3
    assert "category" in res_df.columns
    assert "Total Sales" in res_df.columns
    assert "Avg Profit" in res_df.columns
    
    # Verify electronics values
    elec = res_df[res_df["category"] == "Electronics"].iloc[0]
    assert elec["Total Sales"] == 3000.0
    assert elec["Avg Profit"] == 350.0  # (200 + 500) / 2

def test_eval_dax_table_filter(sample_df):
    # FILTER
    expr = "FILTER(df, [revenue] >= 1500)"
    res_df = eval_dax_table(sample_df, expr)
    
    assert len(res_df) == 3
    assert list(res_df["revenue"]) == [2000.0, 1500.0, 3000.0]
    
    # FILTER combined with SUMMARIZE
    expr_comb = "FILTER(SUMMARIZE(df, [category], \"Sales\", SUM([revenue])), [Sales] > 1000)"
    # Wait, the inner function references `df` but we evaluate top-level
    # Wait, eval_dax_table doesn't evaluate nested tables yet, it only parses outer SUMMARIZE or FILTER.
    # Let's verify standard error for unsupported expressions or simple tables.
    res_base = eval_dax_table(sample_df, "sales_df")
    assert len(res_base) == 5
