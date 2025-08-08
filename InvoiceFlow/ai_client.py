import os, time, requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def _get_secret(name, default=None):
    try:
        import streamlit as st
        return st.secrets.get(name, os.environ.get(name, default))
    except Exception:
        return os.environ.get(name, default)

def call_ai(messages, model="openrouter/auto", max_tokens=700, temperature=0.2, retries=2, timeout=30):
    api_key = _get_secret("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": _get_secret("APP_REFERRER", ""),
        "X-Title": _get_secret("APP_TITLE", "Receivables App"),
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 401:
                raise RuntimeError("Unauthorized (401). Check OPENROUTER_API_KEY or domain restrictions.")
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                last_err = RuntimeError("Rate limited (429). Retrying…")
                continue
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            last_err = RuntimeError("AI request timed out.")
        except requests.exceptions.RequestException as e:
            last_err = RuntimeError(f"Network/API error: {e}")
        except Exception as e:
            last_err = RuntimeError(f"AI parsing error: {e}")
        time.sleep(2 ** attempt)

    raise last_err or RuntimeError("Unknown AI error")
