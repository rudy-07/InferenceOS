# OpenAI-Compatible REST API Specification (`server/api/routes/`)

InferenceOS provides a high-throughput, drop-in replacement for OpenAI endpoints built on FastAPI.

---

## Supported Endpoints

### 1. Chat Completions (`POST /v1/chat/completions`)
Supports non-streaming and streaming (`stream=true`) Server-Sent Events (SSE).

**Request Body**:
```json
{
  "model": "llama-3-8b-instruct",
  "messages": [
    {"role": "system", "content": "You are an AI assistant."},
    {"role": "user", "content": "Explain adaptive microbatching."}
  ],
  "temperature": 0.7,
  "max_tokens": 512,
  "stream": true
}
```

### 2. Text Completions (`POST /v1/completions`)
Legacy prompt completion endpoint.

### 3. Embeddings (`POST /v1/embeddings`)
Generates text embedding vectors.

### 4. Models List (`GET /v1/models`)
Lists all registered models in the local library.

---

## Python Client Usage

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="inferenceos")

response = client.chat.completions.create(
    model="llama-3-8b-instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response.choices[0].message.content)
```
