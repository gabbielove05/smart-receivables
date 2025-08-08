import streamlit as st
import os

# Test the API key loading logic
try:
    api_key = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ No API key found")
        print("Available secrets keys:", list(st.secrets.keys()) if hasattr(st.secrets, 'keys') else "No secrets available")
    else:
        print("✅ API key found")
        print(f"Key starts with: {api_key[:10]}...{api_key[-4:]}")
        
except Exception as e:
    print(f"❌ Error loading API key: {e}")
    print("This might be because we're not in a Streamlit context")
