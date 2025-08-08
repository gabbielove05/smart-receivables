import requests

# Test the current API key
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
        print("✅ API key is working!")
    elif response.status_code == 401:
        print("❌ API key is invalid or has domain restrictions")
        print("This could mean:")
        print("1. The key is incorrect")
        print("2. The key has domain restrictions")
        print("3. The key has been disabled")
    else:
        print(f"❌ Unexpected error: {response.status_code}")
        
except Exception as e:
    print(f"Error: {e}")
