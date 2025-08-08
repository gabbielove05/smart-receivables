# ai_client.py — OpenRouter client using the OpenAI SDK style (per docs)
# https://openrouter.ai/docs/quickstart

import os
import streamlit as st
from openai import OpenAI

def _secret(name, default=""):
    try:
        return st.secrets.get(name, os.environ.get(name, default))
    except Exception:
        return os.environ.get(name, default)

def get_client():
    api_key = _secret("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in Streamlit Secrets.")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

def call_ai(messages, model="openai/gpt-4o", max_tokens=700, temperature=0.2):
    """
    Minimal wrapper. Returns the assistant's text.
    """
    try:
        client = get_client()
        resp = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": _secret("APP_REFERRER", "https://smart-receivables-glove.streamlit.app"),
                "X-Title": _secret("APP_TITLE", "Smart Receivables"),
            },
            model=model,  # keep to a known model; can swap to "openrouter/auto"
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"AI call failed: {e}")

# Backward compatibility
def chat(messages, model="openai/gpt-4o", max_tokens=700, temperature=0.2):
    return call_ai(messages, model, max_tokens, temperature)
