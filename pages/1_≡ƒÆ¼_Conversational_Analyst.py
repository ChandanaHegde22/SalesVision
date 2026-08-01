"""Conversational Data Analyst - a new SalesVision page.

This file lives under pages/ so Streamlit automatically adds it to the
sidebar navigation alongside the existing app.py dashboard, without any
changes to app.py or its behavior.
"""

import os
import sys

import pandas as pd
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.chat_memory import ChatMemory
from src.conversation_agent import answer_question
from src.llm_client import LLMClient

DATA_PATH = os.path.join(ROOT_DIR, "data", "sales_data.csv")

st.set_page_config(page_title="SalesVision - Conversational Analyst", page_icon="💬", layout="wide")

st.title("💬 Conversational Data Analyst")
st.write(
    "Ask questions about your sales data in plain English. This assistant reads your data, "
    "never modifies it, and grounds every answer in the numbers it actually finds."
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Run: python src/generate_data.py")
    st.stop()

sales_df = load_data()
memory = ChatMemory()
llm_client = LLMClient()

if not llm_client.is_available():
    st.info(
        "Running in rule-based mode (no LLM API key detected). Common question types "
        "(top N, filters, trends, comparisons, growth, summaries) still work. Set "
        "`ANTHROPIC_API_KEY` or `GEMINI_API_KEY` as an environment variable to enable "
        "free-form question understanding for anything outside those patterns.",
        icon="ℹ️",
    )

SUGGESTED_QUESTIONS = [
    "What were my top selling products?",
    "Show sales in South",
    "Which region had the highest growth?",
    "Show products whose sales are declining",
    "Compare North and South",
    "Show me monthly revenue",
    "What is the average order value?",
    "Give me a summary of this year's performance",
]

top_bar_left, top_bar_right = st.columns([3, 1])
with top_bar_right:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        memory.clear()
        st.rerun()

    if memory.turns:
        export_text = "\n\n".join(
            f"Q: {t.question}\nA: {t.answer}" for t in memory.turns
        )
        st.download_button(
            "⬇️ Export Conversation",
            data=export_text,
            file_name="salesvision_conversation.txt",
            mime="text/plain",
            use_container_width=True,
        )

st.subheader("Suggested Questions")
suggestion_cols = st.columns(4)
clicked_suggestion = None
for i, question in enumerate(SUGGESTED_QUESTIONS):
    with suggestion_cols[i % 4]:
        if st.button(question, key=f"suggestion_{i}", use_container_width=True):
            clicked_suggestion = question

st.divider()

# Render prior conversation
for turn in memory.turns:
    with st.chat_message("user"):
        st.write(turn.question)
    with st.chat_message("assistant"):
        st.write(turn.answer)

user_question = st.chat_input("Ask about your sales data...")
question_to_process = clicked_suggestion or user_question

if question_to_process:
    with st.chat_message("user"):
        st.write(question_to_process)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            bundle = answer_question(question_to_process, sales_df, memory, llm_client=llm_client)

        if bundle.is_error:
            st.warning(bundle.answer_text)
        else:
            st.write(bundle.answer_text)

            if bundle.table is not None and not bundle.table.empty:
                if bundle.chart is not None:
                    st.plotly_chart(bundle.chart, use_container_width=True)
                st.dataframe(bundle.table, use_container_width=True)

            if bundle.insights:
                with st.expander("📊 Business Insights", expanded=True):
                    for insight in bundle.insights:
                        st.markdown(f"- {insight}")

        if bundle.followups:
            st.caption("Follow-up questions:")
            followup_cols = st.columns(len(bundle.followups))
            for i, followup in enumerate(bundle.followups):
                with followup_cols[i]:
                    st.button(followup, key=f"followup_{len(memory.turns)}_{i}", use_container_width=True)

    st.rerun()
