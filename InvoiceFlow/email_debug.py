#!/usr/bin/env python3
"""
Debug script for email configuration
"""

import os
import sys
sys.path.append('.')

def debug_email_config():
    print("🔍 Email Configuration Debug")
    print("=" * 40)
    
    # Check environment variables
    print("1. Environment Variables:")
    email_user = os.getenv("EMAIL_USER", "GLoveEmailTest@gmail.com")
    email_pass = os.getenv("EMAIL_APP_PASSWORD", "shutzfeeqtpdbqnp")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "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")
    
    print(f"   EMAIL_USER: {email_user}")
    print(f"   EMAIL_APP_PASSWORD: {'*' * len(email_pass) if email_pass else 'NOT SET'}")
    print(f"   OPENROUTER_API_KEY: {openrouter_key[:20]}..." if openrouter_key else "NOT SET")
    
    # Check secrets file
    print("\n2. Streamlit Secrets:")
    try:
        import streamlit as st
        print("   Streamlit available")
        try:
            email_secret = st.secrets["email"]["user"]
            print(f"   Email from secrets: {email_secret}")
        except:
            print("   No email secrets found")
    except:
        print("   Streamlit not available")
    
    # Test OpenRouter API
    print("\n3. OpenRouter API Test:")
    try:
        import requests
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": "openai/gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                               headers=headers, data=body, timeout=10)
        if response.status_code == 200:
            print("   ✅ OpenRouter API working")
        else:
            print(f"   ❌ OpenRouter API error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ OpenRouter API error: {e}")
    
    # Test Gmail SMTP
    print("\n4. Gmail SMTP Test:")
    try:
        import smtplib
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(email_user, email_pass)
            print("   ✅ Gmail SMTP working")
    except Exception as e:
        print(f"   ❌ Gmail SMTP error: {e}")
    
    print("\n🔧 Troubleshooting Tips:")
    print("   - If Gmail fails: Check app password in Gmail settings")
    print("   - If OpenRouter fails: Check API key")
    print("   - If Streamlit secrets fail: Check .streamlit/secrets.toml")

if __name__ == "__main__":
    debug_email_config()
