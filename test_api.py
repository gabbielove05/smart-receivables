import requests
import json

# Test the API key
api_key = "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"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "gpt-4o-mini",
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
