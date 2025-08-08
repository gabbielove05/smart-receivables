import os, time, requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def _get_secret(name, default=None):
    try:
        import streamlit as st
        value = st.secrets.get(name, os.environ.get(name, default))
        if name == "OPENROUTER_API_KEY" and value:
            print(f"✅ Found {name}: {value[:10]}...{value[-4:]}")
        elif name == "OPENROUTER_API_KEY" and not value:
            print(f"❌ {name} not found in secrets or environment")
        return value
    except Exception as e:
        print(f"⚠️ Error getting {name}: {e}")
        return os.environ.get(name, default)

def call_ai(messages, model="openrouter/auto", max_tokens=700, temperature=0.2, retries=2, timeout=30):
    api_key = _get_secret("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    
    # Debug: Print what we're using
    print(f"🔍 Using API key: {api_key[:10]}...{api_key[-4:]}")
    print(f"🔍 Using model: {model}")
    print(f"🔍 Using referer: {_get_secret('APP_REFERRER', 'None')}")

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
