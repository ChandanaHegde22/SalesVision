"""Session-scoped conversation memory for the Conversational Data Analyst.

Backed by st.session_state so it naturally persists for the browser session
and resets when the user clicks "Clear Chat" or starts a new session. Keeping
this as a small, dependency-free class (rather than scattering session_state
keys through the UI file) makes it easy to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:  # allows chat_memory to be unit-tested without streamlit installed
    _HAS_STREAMLIT = False


@dataclass
class Turn:
    question: str
    operation: str
    dimension: Optional[str]
    metric: Optional[str]
    filters: List[Dict[str, str]] = field(default_factory=list)
    answer: str = ""


class ChatMemory:
    """Wraps a list of `Turn`s, persisted in Streamlit's session_state when available."""

    _STATE_KEY = "conversational_analyst_turns"
    _CACHE_KEY = "conversational_analyst_cache"

    def __init__(self):
        if _HAS_STREAMLIT:
            if self._STATE_KEY not in st.session_state:
                st.session_state[self._STATE_KEY] = []
            if self._CACHE_KEY not in st.session_state:
                st.session_state[self._CACHE_KEY] = {}
        else:
            self._local_turns: List[Turn] = []
            self._local_cache: Dict[str, dict] = {}

    @property
    def turns(self) -> List[Turn]:
        return st.session_state[self._STATE_KEY] if _HAS_STREAMLIT else self._local_turns

    @property
    def cache(self) -> Dict[str, dict]:
        return st.session_state[self._CACHE_KEY] if _HAS_STREAMLIT else self._local_cache

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)

    def clear(self) -> None:
        self.turns.clear()
        self.cache.clear()

    def history_as_text(self, max_turns: int = 5) -> List[str]:
        lines = []
        for t in self.turns[-max_turns:]:
            lines.append(f"Q: {t.question}")
            if t.answer:
                lines.append(f"A: {t.answer}")
        return lines

    def last_turn(self) -> Optional[Turn]:
        return self.turns[-1] if self.turns else None

    def cache_get(self, key: str) -> Optional[dict]:
        return self.cache.get(key)

    def cache_set(self, key: str, value: dict) -> None:
        self.cache[key] = value

    @staticmethod
    def normalize_key(question: str) -> str:
        return " ".join(question.strip().lower().split())
