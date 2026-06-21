from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="meta/llama-3.3-70b-instruct",
  messages=[{"role":"user","content":"Explain machine learning in simple terms."}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=False
)

print(completion.choices[0].message)