from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://172.28.102.11:8011/v1"
)

r = client.chat.completions.create(
    model="Qwen/Qwen3-14B-30",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"}
    ]
    
)
print(r.choices[0].message.content)