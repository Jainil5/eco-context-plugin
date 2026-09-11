from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

response = client.responses.create(
  model="openai/gpt-oss-20b",
  input="Which number is larger, 9.11 or 9.8?",
  max_output_tokens=4096,
  top_p=1,
  temperature=1,
  stream=False
)


print(response)