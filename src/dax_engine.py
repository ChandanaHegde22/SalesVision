import re
import numpy as np
import pandas as pd

def dax_if(condition, val_true, val_false):
    """
    Evaluates condition and returns val_true if true, else val_false.
    Works for both scalar and vector inputs.
    """
    if isinstance(condition, (pd.Series, np.ndarray)):
        return np.where(condition, val_true, val_false)
    return val_true if condition else val_false

def divide(a, b, alt=0):
    """
    Safely divides a by b, returning alt if b is 0.
    """
    if isinstance(b, (pd.Series, np.ndarray)):
        # Vector division with safety
        return np.where(b == 0, alt, a / b)
    return alt if b == 0 else a / b

def eval_calculated_column(df: pd.DataFrame, expr: str):
    """
    Evaluates a DAX-like expression row-by-row on a DataFrame.
    Example: "[revenue] - [profit]" or "IF([discount] > 10, [revenue] * 0.9, [revenue])"
    """
    # Find all column names enclosed in square brackets [ColumnName]
    cols = re.findall(r'\[([^\]]+)\]', expr)
    
    # Validate columns
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataset.")
            
    # Replace [ColName] with df['ColName']
    eval_expr = expr
    for col in cols:
        eval_expr = eval_expr.replace(f"[{col}]", f"df['{col}']")
        
    # Replace DAX-style IF with dax_if
    eval_expr = re.sub(r'(?i)\bIF\s*\(', 'dax_if(', eval_expr)
    # Replace DIVIDE(a, b) or DIVIDE(a, b, alt)
    eval_expr = re.sub(r'(?i)\bDIVIDE\s*\(', 'divide(', eval_expr)
    
    # Establish execution environment
    allowed_names = {
        'df': df,
        'np': np,
        'pd': pd,
        'dax_if': dax_if,
        'divide': divide,
        'abs': abs,
        'round': round,
        'len': len
    }
    
    # Add numpy functions for general use (sin, cos, log, exp, etc.)
    for name in dir(np):
        if not name.startswith('_') and name not in allowed_names:
            allowed_names[name] = getattr(np, name)
            
    try:
        # Evaluate safely
        result = eval(eval_expr, {"__builtins__": None}, allowed_names)
        return result
    except Exception as e:
        raise ValueError(f"Error evaluating formula '{expr}': {str(e)}")

def eval_measure(df: pd.DataFrame, expr: str):
    """
    Evaluates a DAX-like measure aggregation expression.
    Example: "SUM([revenue])" or "DIVIDE(SUM([profit]), SUM([revenue]))"
    """
    # Verify that there are no un-aggregated columns left. E.g. [revenue] without SUM()
    check_expr = expr
    check_expr = re.sub(r'(?i)\bSUM\(\s*\[([^\]]+)\]\s*\)', '', check_expr)
    check_expr = re.sub(r'(?i)\bAVERAGE\(\s*\[([^\]]+)\]\s*\)', '', check_expr)
    check_expr = re.sub(r'(?i)\bAVG\(\s*\[([^\]]+)\]\s*\)', '', check_expr)
    check_expr = re.sub(r'(?i)\bMIN\(\s*\[([^\]]+)\]\s*\)', '', check_expr)
    check_expr = re.sub(r'(?i)\bMAX\(\s*\[([^\]]+)\]\s*\)', '', check_expr)
    check_expr = re.sub(r'(?i)\bCOUNT\(\s*\[([^\]]+)\]\s*\)', '', check_expr)
    check_expr = re.sub(r'(?i)\bDISTINCTCOUNT\(\s*\[([^\]]+)\]\s*\)', '', check_expr)
    
    raw_cols = re.findall(r'\[([^\]]+)\]', check_expr)
    if raw_cols:
        raise ValueError(
            f"Measures must aggregate all columns. Found un-aggregated column(s): {', '.join(raw_cols)}. "
            "Please wrap them in SUM, AVERAGE, MIN, MAX, COUNT, or DISTINCTCOUNT."
        )

    # Replace aggregations: SUM([col]) -> df['col'].sum()
    eval_expr = expr
    eval_expr = re.sub(r'(?i)\bSUM\(\s*\[([^\]]+)\]\s*\)', r"df['\1'].sum()", eval_expr)
    eval_expr = re.sub(r'(?i)\bAVERAGE\(\s*\[([^\]]+)\]\s*\)', r"df['\1'].mean()", eval_expr)
    eval_expr = re.sub(r'(?i)\bAVG\(\s*\[([^\]]+)\]\s*\)', r"df['\1'].mean()", eval_expr)
    eval_expr = re.sub(r'(?i)\bMIN\(\s*\[([^\]]+)\]\s*\)', r"df['\1'].min()", eval_expr)
    eval_expr = re.sub(r'(?i)\bMAX\(\s*\[([^\]]+)\]\s*\)', r"df['\1'].max()", eval_expr)
    eval_expr = re.sub(r'(?i)\bCOUNT\(\s*\[([^\]]+)\]\s*\)', r"df['\1'].count()", eval_expr)
    eval_expr = re.sub(r'(?i)\bDISTINCTCOUNT\(\s*\[([^\]]+)\]\s*\)', r"df['\1'].nunique()", eval_expr)
    
    # Replace DAX-style IF with dax_if
    eval_expr = re.sub(r'(?i)\bIF\s*\(', 'dax_if(', eval_expr)
    # Replace DIVIDE
    eval_expr = re.sub(r'(?i)\bDIVIDE\s*\(', 'divide(', eval_expr)
    
    allowed_names = {
        'df': df,
        'np': np,
        'pd': pd,
        'dax_if': dax_if,
        'divide': divide,
        'abs': abs,
        'round': round
    }
        
    try:
        result = eval(eval_expr, {"__builtins__": None}, allowed_names)
        # If it's a series or array, take mean or raise error. Measures must be scalar.
        if isinstance(result, (pd.Series, np.ndarray)):
            raise ValueError("Measure formula did not evaluate to a single numeric/scalar value.")
        return result
    except Exception as e:
        raise ValueError(f"Error evaluating measure '{expr}': {str(e)}")

def split_dax_args(args_str: str):
    """
    Splits arguments by comma, respecting quotes and parentheses.
    """
    tokens = []
    current_token = []
    paren_depth = 0
    in_quotes = False
    quote_char = None
    
    for char in args_str:
        if char in ('"', "'"):
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None
            current_token.append(char)
        elif char == '(' and not in_quotes:
            paren_depth += 1
            current_token.append(char)
        elif char == ')' and not in_quotes:
            paren_depth -= 1
            current_token.append(char)
        elif char == ',' and not in_quotes and paren_depth == 0:
            tokens.append("".join(current_token).strip())
            current_token = []
        else:
            current_token.append(char)
    if current_token:
        tokens.append("".join(current_token).strip())
        
    return tokens

def parse_summarize(df: pd.DataFrame, args_str: str) -> pd.DataFrame:
    """
    Evaluates SUMMARIZE(df, GroupByCol1, GroupByCol2, "Name", Formula, ...)
    """
    tokens = split_dax_args(args_str)
    if not tokens:
        raise ValueError("SUMMARIZE requires arguments.")
        
    # The first token is the table name, we ignore it since we pass `df` directly
    # Subsequent tokens are group by columns, or pairs of (new_column_name, aggregation_expression)
    groupby_cols = []
    agg_specs = []
    
    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token.startswith('[') and token.endswith(']'):
            col_name = token[1:-1]
            if col_name not in df.columns:
                raise ValueError(f"Group-by column '{col_name}' not found in the dataset.")
            groupby_cols.append(col_name)
            i += 1
        else:
            # We expect a string literal for the new column name, followed by its calculation formula
            new_name = token.strip('"\'')
            if i + 1 >= len(tokens):
                raise ValueError(f"SUMMARIZE expected a formula after new column name '{new_name}'.")
            formula = tokens[i+1]
            agg_specs.append((new_name, formula))
            i += 2
            
    if not groupby_cols:
        raise ValueError("SUMMARIZE requires at least one group-by column (e.g., [category]).")
        
    # Group by the specified columns
    grouped = df.groupby(groupby_cols)
    
    if not agg_specs:
        # Just return unique grouped values
        return df[groupby_cols].drop_duplicates().reset_index(drop=True)
        
    # Build list of results for each group
    # We will iterate through each group, evaluate formulas on that group's subset, and combine them.
    # To run efficiently and correctly:
    records = []
    for keys, group_df in grouped:
        record = {}
        # Fill group-by values
        if isinstance(keys, tuple):
            for col_idx, col_name in enumerate(groupby_cols):
                record[col_name] = keys[col_idx]
        else:
            record[groupby_cols[0]] = keys
                
        # Evaluate aggregations on this group's subset
        for new_name, formula in agg_specs:
            try:
                val = eval_measure(group_df, formula)
                record[new_name] = val
            except Exception as e:
                raise ValueError(f"Error calculating group field '{new_name}' with formula '{formula}': {str(e)}")
        records.append(record)
        
    return pd.DataFrame(records)

def parse_filter(df: pd.DataFrame, args_str: str) -> pd.DataFrame:
    """
    Evaluates FILTER(df, [revenue] > 500)
    """
    tokens = split_dax_args(args_str)
    if len(tokens) < 2:
        raise ValueError("FILTER requires a table and a boolean expression.")
        
    # Rejoin the condition tokens if they were split by commas inside functions
    condition = ",".join(tokens[1:])
    
    try:
        # Evaluate the filter condition row-by-row
        mask = eval_calculated_column(df, condition)
        # Handle series mask
        if isinstance(mask, pd.Series):
            return df[mask.fillna(False)].reset_index(drop=True)
        else:
            # Scalar result or list-like
            mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
            return df[mask].reset_index(drop=True)
    except Exception as e:
        raise ValueError(f"Error evaluating FILTER condition '{condition}': {str(e)}")

def eval_dax_table(df: pd.DataFrame, expr: str) -> pd.DataFrame:
    """
    Main entry point for evaluating DAX calculated table expressions.
    Supported functions: SUMMARIZE, FILTER, or simply referencing the table.
    """
    expr = expr.strip()
    
    # Check if they just want the base table
    if expr.lower() in ("df", "dataset", "sales_df"):
        return df.copy()
        
    # Match SUMMARIZE(df, ...)
    summarize_match = re.match(r'(?i)^SUMMARIZE\s*\((.*)\)$', expr)
    if summarize_match:
        return parse_summarize(df, summarize_match.group(1))
        
    # Match FILTER(df, ...)
    filter_match = re.match(r'(?i)^FILTER\s*\((.*)\)$', expr)
    if filter_match:
        return parse_filter(df, filter_match.group(1))
        
    raise ValueError(
        "Unsupported DAX Table expression. Only SUMMARIZE(df, ...) and FILTER(df, ...) "
        "functions are supported at this time."
    )
