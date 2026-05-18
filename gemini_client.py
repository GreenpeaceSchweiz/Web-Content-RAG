import os
from functools import lru_cache

from google import genai


def _load_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    try:
        import streamlit as st

        api_key = (
            api_key
            or st.secrets.get("GEMINI_API_KEY")
            or st.secrets.get("gemini", {}).get("GEMINI_API_KEY")
            or st.secrets.get("Gemini", {}).get("GEMINI_API_KEY")
        )
    except Exception:
        pass

    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Set it as an environment variable or in .streamlit/secrets.toml."
        )

    return api_key


@lru_cache(maxsize=1)
def get_genai_client():
    return genai.Client(api_key=_load_api_key())