# Conversational Data Analyst

A new enterprise-level feature for SalesVision that lets users ask questions
about their sales data in plain English instead of using filters and charts.
It behaves like a business analyst, not a chatbot: every answer is grounded
in numbers actually computed from the dataset, never invented.

This feature is additive. `app.py` and the existing dashboard are untouched.

---

## 1. Folder structure

```
salesvision/
├── app.py                                   # existing dashboard (unchanged)
├── pages/
│   └── 1_💬_Conversational_Analyst.py        # NEW - sidebar page (Streamlit auto-detects pages/)
├── src/
│   ├── forecast.py, generate_data.py, ...    # existing (unchanged)
│   ├── conversation_agent.py                 # NEW - orchestrates the full pipeline
│   ├── query_generator.py                    # NEW - NL question -> QuerySpec (JSON)
│   ├── safe_executor.py                      # NEW - QuerySpec -> pandas result (whitelisted ops only)
│   ├── insight_generator.py                  # NEW - result -> grounded business insights
│   ├── chart_generator.py                    # NEW - result -> appropriate Plotly chart
│   ├── chat_memory.py                        # NEW - session conversation history + cache
│   ├── llm_client.py                         # NEW - provider-agnostic LLM wrapper (Claude/Gemini)
│   └── prompt_templates.py                   # NEW - all prompt text in one place
├── tests/
│   └── test_conversation_agent.py            # NEW - offline tests (no API key required)
├── .env.example                               # NEW - configuration template
└── requirements.txt                           # updated with new optional dependencies
```

Each module has a single responsibility, as requested:

| Module | Responsibility |
|---|---|
| `query_generator.py` | Turn a question into a structured `QuerySpec`. Rule-based fast path first, LLM fallback second. |
| `safe_executor.py` | Execute a `QuerySpec` against the dataframe. Only hand-written, whitelisted pandas operations - no `eval`/`exec`/dynamic dispatch. |
| `insight_generator.py` | Compute 2-5 business insights **directly from the result table**, never from the LLM, so they can't be fabricated. |
| `chart_generator.py` | Pick a chart type (line / bar / pie / scatter / none) based on the operation and result shape. |
| `chat_memory.py` | Session-scoped turn history + a question -> answer cache, backed by `st.session_state`. |
| `llm_client.py` | One place that talks to Claude or Gemini. Everything else is provider-agnostic. |
| `conversation_agent.py` | Wires all of the above together for one `answer_question(...)` call. |

---

## 2. How a question is answered (pipeline)

```
question
   │
   ├─▶ cache lookup (chat_memory) ──▶ if hit: return cached answer (no LLM call)
   │
   ▼
query_generator.generate_query_spec()
   │  1. deterministic rule-based matcher (regex/keyword) tries first
   │  2. if no confident match AND an LLM key is configured, ask the LLM
   │     for a JSON QuerySpec (schema + known values are given to it)
   ▼
QuerySpec  (operation, dimension, metric, filters, ...)
   │
   ├─▶ operation == "unsupported"? → return an honest message, no execution
   │
   ▼
safe_executor.execute()
   │  - validates every column against the real dataframe
   │  - validates every filter value against the real data
   │  - raises ColumnNotFoundError / ValueNotFoundError on a miss
   │    (caught by conversation_agent and turned into a plain-English message)
   ▼
ExecutionResult (a pandas DataFrame + metadata)
   │
   ├─▶ chart_generator.build_chart()      → Plotly figure or None (small results stay table-only)
   ├─▶ insight_generator.generate_insights() → 2-5 grounded facts
   ▼
conversation_agent
   │  - phrases a short answer from the facts (LLM if available, else joins them)
   │  - suggests 3-4 follow-up questions (LLM if available, else static per-operation list)
   │  - records the turn in chat_memory (for history + reference resolution + export)
   ▼
AnswerBundle → rendered by pages/1_💬_Conversational_Analyst.py
```

### Why the LLM never touches the dataframe
The LLM is only ever asked for one of three things, and every one of them is
validated before use:
1. A JSON `QuerySpec` (question → operation) - checked against a fixed set of
   operations and the dataframe's real columns/values.
2. A short natural-language rephrasing of **facts you already computed**
   (the prompt explicitly says "do not alter the numbers").
3. A list of follow-up question strings (used as-is, purely cosmetic).

There is no code path where LLM output becomes a pandas expression, a
filesystem call, or a shell command. `tests/test_conversation_agent.py`
includes a regression test that greps `safe_executor.py` for `eval(`,
`exec(`, `os.system`, and `subprocess` to keep it that way.

---

## 3. Chat memory & references

`chat_memory.ChatMemory` keeps a list of `Turn`s (question, operation,
dimension, metric, filters, answer) for the session, plus a small cache
keyed by the normalized question text so identical questions never
re-trigger an LLM call or re-execution.

`conversation_agent._resolve_reference()` handles the example from the
spec:
```
User: Show Karnataka sales.
User: Compare with Tamil Nadu.
```
If a short follow-up mentions "compare"/"that"/"it" and doesn't carry its
own filter, it reuses the previous turn's dimension/filter as one side of
the comparison.

---

## 4. Handling missing data honestly

The sample dataset (`data/sales_data.csv`) has these columns: `date,
product_name, category, region, store, quantity, price, discount, revenue,
profit, promotion, holiday`. It has **no** `customer`, `salesperson`, or
`state` column - `region` (South/North/East/West) is the closest existing
field, and Karnataka/Tamil Nadu (Indian states) aren't in it.

Rather than guessing or hallucinating, the assistant is explicit:

- **"Which salesperson generated maximum revenue?"** → *"I couldn't find a
  'salesperson' column in your dataset."*
- **"Show sales in Karnataka."** → *"I couldn't find 'Karnataka' in your
  data. Available region values are: East, North, South, West..."*

This matches the product requirement to never invent a missing column, and
extends the same honesty to missing dimension **values**.

---

## 5. Installation

```bash
cd salesvision
pip install -r requirements.txt
```

To enable free-form question understanding beyond the built-in patterns,
copy `.env.example` to `.env` and add a key:

```bash
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

Load it however your environment normally loads `.env` files (e.g.
`python-dotenv`, or export the variables in your shell) before running
Streamlit. **Without a key, the feature still works** for the question
types in the product spec via the rule-based matcher - it just can't
generalize to arbitrary new phrasings.

Run the app as usual:

```bash
streamlit run app.py
```

Streamlit automatically lists every file under `pages/` in the sidebar, so
"Conversational Analyst" appears alongside the existing dashboard with no
changes to `app.py`.

---

## 6. Testing

```bash
cd salesvision
pytest tests/ -v
```

All 15 tests run **without any API key**, because the rule-based matcher
covers every example question from the product spec. They cover: each
example question type, the missing-column/missing-value graceful paths,
the cache, and a static check that `safe_executor.py` contains no
`eval`/`exec`/OS calls.

---

## 7. Example conversations

These transcripts are real output from the rule-based path (no LLM key)
running against `data/sales_data.csv`, captured while building this feature:

**Top products**
```
User: What were my top selling products?
AI:   Laptop leads with revenue of 685,439,153, 50.7% of total revenue. The
      top 10 product name entries together account for 100.0% of total
      revenue. The gap between the highest and lowest shown is 673,336,004
      in revenue.
      [Horizontal bar chart + table of all 10 products]
      Follow-ups: Compare these by region · Show the trend for the top item · Which of these are declining?
```

**Filtering + comparison**
```
User: Show sales in South
AI:   Combined revenue across the filtered selection is 336,147,339.
      [Pie chart by category for the South region]

User: Compare South and North
AI:   North recorded 337,578,851 in revenue versus 336,147,339 for South
      (1,431,512 difference). Profit was 92,501,677 for North and
      89,947,501 for South.
      [Bar chart comparing South vs North]
```

**Missing column, handled honestly**
```
User: Which salesperson generated maximum revenue?
AI:   I couldn't find a 'salesperson' column in your dataset.
```

**Missing value, handled honestly**
```
User: Show sales in Karnataka.
AI:   I couldn't find 'Karnataka' in your data. Available product_name
      values: Coffee Maker, Headphones, ... Available category values:
      Electronics, Fashion, Furniture, Home Appliances. Available region
      values: East, North, South, West. Available store values: Store A,
      Store B, Store C, Store D, Store E.
```

**Trend + summary**
```
User: Show me monthly revenue
AI:   Revenue increased from 34,206,290 to 47,752,745 across the shown
      periods. The highest point in the series was 53,324,006.
      [Line chart of monthly revenue]

User: Give me a summary of this year's performance
AI:   Total revenue is 1,350,897,319. Total profit is 368,465,214.
      Top category is Electronics, top region is East.
```

---

## 8. Notes & assumptions

- **Average order value**: the dataset has no order-id column, so each row
  is treated as one order line. The UI surfaces this assumption explicitly
  in the answer rather than silently guessing at a different definition.
- **"Why did revenue decrease last month?"** is mapped to the `growth`
  operation over `region` at monthly granularity, sorted ascending, so the
  most negative contributors surface first.
- Adding a real `customer`, `salesperson`, or `state` column to the CSV
  later requires no code changes to the fast path beyond removing the
  entry from `DIMENSION_SYNONYMS` in `query_generator.py` and mapping it to
  the new column name - the rest of the pipeline (execution, charts,
  insights) is fully generic.
