# InvoiceFlow/ai_client.py
# OpenRouter API client for Streamlit app

import os
import streamlit as st
from openai import OpenAI

def _secret(name, default=""):
    """Get a secret from Streamlit Cloud or environment variables."""
    try:
        return st.secrets.get(name, os.environ.get(name, default))
    except Exception:
        return os.environ.get(name, default)

def get_client():
    """Initialize and return an OpenAI client pointed to OpenRouter."""
    api_key = (_secret("OPENROUTER_API_KEY") or "").strip()
    
    # Debug: Print what we found
    print(f"🔍 API Key found: {'Yes' if api_key else 'No'}")
    if api_key:
        print(f"🔍 API Key starts with: {api_key[:10]}...{api_key[-4:]}")
    else:
        print("🔍 No API key found in secrets")
    
    if not api_key.startswith("sk-or-"):
        raise RuntimeError("OPENROUTER_API_KEY not found or invalid in Streamlit Secrets.")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

def chat(messages, model="openai/gpt-4o", max_tokens=700, temperature=0.2):
    """
    Send a chat request to OpenRouter and return the assistant's reply text.
    - messages: list of {"role": ..., "content": ...} dicts
    - model: OpenRouter model name
    """
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_headers={
            "HTTP-Referer": _secret("APP_REFERRER", "https://smart-receivables-glove.streamlit.app"),
            "X-Title": _secret("APP_TITLE", "Smart Receivables"),
        }
    )
    return resp.choices[0].message.content.strip()

# Backward compatibility
def call_ai(messages, model="openai/gpt-4o", max_tokens=700, temperature=0.2):
    return chat(messages, model, max_tokens, temperature)
