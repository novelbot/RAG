import os
from openai import OpenAI

# Test OpenAI directly
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not found in environment")
    exit(1)

print(f"✅ API Key found: {api_key[:10]}...")

client = OpenAI(api_key=api_key)

print("\n=== Testing OpenAI Streaming ===")
try:
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Count from 1 to 5 slowly."}
        ],
        stream=True
    )
    
    print("Streaming response:")
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n✅ Streaming successful\!")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\n=== Testing OpenAI Non-Streaming ===")
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Say hello in Korean."}
        ]
    )
    print(f"Response: {response.choices[0].message.content}")
    print("✅ Non-streaming successful\!")
    
except Exception as e:
    print(f"❌ Error: {e}")
