import requests
import json


URL = "https://openrouter.ai/api/v1/chat/completions"


def print_divider(title):
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


# 🔹 First API call
print_divider("FIRST REQUEST")

payload_1 = {
    "model": "openrouter/elephant-alpha",
    "messages": [
        {
            "role": "user",
            "content": "How many r's are in the word 'strawberry'?"
        }
    ],
    "reasoning": {"enabled": True}
}

response = requests.post(
    url=URL,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    data=json.dumps(payload_1)
)

print(f"Status Code: {response.status_code}")

if response.status_code != 200:
    print("❌ Error:", response.text)
    exit()

data = response.json()
message = data['choices'][0]['message']

print("\n🧠 Model Answer:")
print(message.get("content"))

print("\n🔍 Reasoning Details:")
print(message.get("reasoning_details"))


# 🔹 Prepare messages for second call
messages = [
    {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
    {
        "role": "assistant",
        "content": message.get('content'),
        "reasoning_details": message.get('reasoning_details')
    },
    {"role": "user", "content": "Are you sure? Think carefully."}
]


# 🔹 Second API call
print_divider("SECOND REQUEST")

payload_2 = {
    "model": "minimax/minimax-m2.5:free",
    "messages": messages,
    "reasoning": {"enabled": True}
}

response2 = requests.post(
    url=URL,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    data=json.dumps(payload_2)
)

print(f"Status Code: {response2.status_code}")

if response2.status_code != 200:
    print("❌ Error:", response2.text)
    exit()

data2 = response2.json()
message2 = data2['choices'][0]['message']

print("\n🧠 Final Answer:")
print(message2.get("content"))

print("\n🔍 Continued Reasoning:")
print(message2.get("reasoning_details"))