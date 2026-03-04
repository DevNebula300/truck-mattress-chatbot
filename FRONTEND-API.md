# Chat API — Integration for Frontend

Use this to plug the Estee mattress chatbot into your app.

---

## Base URL

- **Production:** `https://truck-mattress-chatbot.vercel.app` (or whatever URL the backend is deployed to)
- **Local:** `http://localhost:8000`

---

## Endpoint: Chat

**POST** `/chat`

**Headers**

| Header       | Required | Description                          |
|-------------|----------|--------------------------------------|
| `Content-Type` | Yes  | `application/json`                  |
| `X-API-Key`    | Yes* | API key you received from the project owner |

\* Only required if the backend is configured with an API key. If you get `401 Invalid or missing API key`, add the key to this header.

**Body (JSON)**

```json
{
  "message": "I drive a Volvo and need a firm mattress",
  "history": [
    { "role": "user", "content": "Hi" },
    { "role": "assistant", "content": "Hello! How can I help you today?" }
  ]
}
```

- `message` (string) — the user’s latest message  
- `history` (array) — previous turns in the conversation. Each item: `{ "role": "user" | "assistant", "content": "..." }`. Can be empty `[]` for the first message.

**Response (200)**

```json
{
  "reply": "Here’s what I’d suggest..."
}
```

**Errors**

- `400` — empty or invalid body (e.g. empty `message`)
- `401` — missing or invalid API key (when the backend uses one)
- `503` — backend not configured (e.g. missing OpenAI key)

---

## Example (JavaScript)

```js
const API_BASE = "https://truck-mattress-chatbot.vercel.app";
const API_KEY = "your-api-key-here";  // from project owner, or omit if not used

async function sendMessage(message, history = []) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(API_KEY && { "X-API-Key": API_KEY }),
    },
    body: JSON.stringify({ message, history }),
  });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.reply;
}
```

---

## Getting the API key

If the backend is protected, the project owner will set an `API_KEY` in the server environment and share that value with you. Put it in your app’s config or env (e.g. `NEXT_PUBLIC_CHAT_API_KEY` or similar) and send it in the `X-API-Key` header on every `/chat` request. Do not commit the key to public repos.
