import requests
import os

# Test the API key directly
api_key = "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"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "openai/gpt-4o-mini",
    "messages": [
        {"role": "user", "content": "Hello, this is a test message."}
    ],
    "max_tokens": 50
}

try:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("✅ API key is working!")
    else:
        print("❌ API key is not working")
        
except Exception as e:
    print(f"Error: {e}")
