import streamlit as st
import os
import requests

def test_api_call():
    """Test the API call with the current secrets"""
    try:
        api_key = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            st.error("❌ Missing OPENROUTER_API_KEY")
            return
        
        st.info(f"🔍 Testing API key: {api_key[:10]}...{api_key[-4:]}")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": st.secrets.get("APP_REFERRER", ""),
            "X-Title": st.secrets.get("APP_TITLE", ""),
        }
        
        payload = {
            "model": "openrouter/auto",
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 5
        }
        
        st.info("🔄 Making API call...")
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        
        st.write(f"**Status Code:** {r.status_code}")
        st.write(f"**Response:** {r.text}")
        
        if r.status_code == 200:
            st.success("✅ API call successful!")
        elif r.status_code == 401:
            st.error("❌ 401 Unauthorized - Check API key or domain restrictions")
        else:
            st.warning(f"⚠️ Unexpected status: {r.status_code}")
            
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Add this to your app to test
if st.button("🧪 Test API Call"):
    test_api_call()
