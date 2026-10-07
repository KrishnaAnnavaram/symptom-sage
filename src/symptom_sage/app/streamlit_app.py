"""Streamlit chat (optional extra ``app``). Run: ``streamlit run src/symptom_sage/app/streamlit_app.py``.

Build the index first (``symptom-sage index``). Research demo only.
"""
from __future__ import annotations

import streamlit as st

from symptom_sage.cli import make_assistant
from symptom_sage.config import Settings
from symptom_sage.safety import DISCLAIMER


@st.cache_resource
def _assistant():
    return make_assistant(Settings.from_env())


def main() -> None:
    st.set_page_config(page_title="symptom-sage (research demo)")
    st.title("symptom-sage")
    st.warning(DISCLAIMER)
    history = st.session_state.setdefault("history", [])
    for role, text in history:
        st.chat_message(role).write(text)
    q = st.chat_input("Describe your symptoms")
    if q:
        st.chat_message("user").write(q)
        ans = _assistant().answer(q)
        if ans.kind == "emergency":
            st.error(ans.text)
        else:
            st.chat_message("assistant").write(ans.text)
        if ans.retrieved:
            with st.expander("Similar cases used"):
                st.table([{"case": c, "assessment": cond, "score": s} for c, cond, s in ans.retrieved])
        history += [("user", q), ("assistant", ans.text)]


main()
