# Ollama-Compatible REST API Specification (`server/api/routes/ollama.py`)

InferenceOS supports native compatibility with Ollama ecosystem tools, frameworks, and web UIs (such as Open WebUI).

---

## Supported Endpoints

- **`POST /api/generate`**: Single-prompt text generation stream.
- **`POST /api/chat`**: Structured chat conversation endpoint.
- **`GET /api/tags`**: List all locally downloaded or registered models.
- **`POST /api/show`**: Inspect model architecture, quantization, and parameters.
- **`GET /api/ps`**: List active running inference models and memory consumption.

---

## Example Curl Request

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama-3-8b",
  "prompt": "Why is local inference important?"
}'
```
