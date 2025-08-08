import requests

# Test the new API key
api_key = "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"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://smart-receivables-glove.streamlit.app",
    "X-Title": "Smart Receivables"
}

data = {
    "model": "openrouter/auto",
    "messages": [
        {"role": "user", "content": "Hello, this is a test message."}
    ],
    "max_tokens": 50,
    "temperature": 0.2
}

try:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=30
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("✅ NEW API KEY IS WORKING!")
    elif response.status_code == 401:
        print("❌ API key is still invalid")
    else:
        print(f"❌ Unexpected error: {response.status_code}")
        
except Exception as e:
    print(f"Error: {e}")
