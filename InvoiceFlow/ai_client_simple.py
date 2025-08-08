import os
import time
import requests

def simple_ai_call(messages, max_tokens=300):
    """Simplified AI call without complex error handling"""
    api_key = "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"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "openrouter/auto",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"Network error: {e}"
