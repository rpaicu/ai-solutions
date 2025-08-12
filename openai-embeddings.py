from openai import OpenAI

client = OpenAI(
    api_key="https://t.me/evilfreelancer",
    base_url="https://api.rpa.icu"
)

response = client.embeddings.create(
    model="FRIDA",
    input="Привет! Как дела?"
)

print(response)
