# Webflow integration – step by step

Use this on your Estee Bedding Webflow site so the chat box talks to your Vercel API.

---

## What you need

- Your **Vercel API URL** (e.g. `https://truck-mattress-chatbot.vercel.app`)
- Your **API key** (if you set `API_KEY` in Vercel) – otherwise leave it as `""`

---

## Step 1: Add three IDs in Webflow

In the **Navigator**, select each element below and give it the **ID** shown.  
(Select element → in the right panel under **Element Settings** find **ID** → type the value.)

| Element | ID to add | Where it is in your structure |
|--------|-----------|-------------------------------|
| **Messages container** | `chat-messages` | The **chat-wrap** div (the one that contains the greeting and the two demo bubbles). |
| **Input field** | `chat-input` | The **input** or **textarea** where users type (inside **chat-field-wrap** / **chat-field**). If you only see a div with “Ask our expert…”, click inside it; the actual field might be inside that div – select that input/textarea and add the ID to it. |
| **Send button** | `chat-send` | The **primary-btn** that says “Ask..” (the one next to the input). |

You should end up with:

- One element with ID `chat-messages` (the chat-wrap).
- One element with ID `chat-input` (the input/textarea).
- One element with ID `chat-send` (the Ask button).

---

## Step 2: Add the script in Webflow

1. Open **Page Settings** (click the gear icon for the page, or use the Pages panel).
2. Go to **Custom Code**.
3. In **Footer Code** (or “Before `</body>` tag”), paste the code below.
4. **Edit the two lines at the top** of the script:
   - `API_BASE` = your Vercel URL (no trailing slash).
   - `API_KEY` = your API key, or `""` if you don’t use one.
5. Click **Save**, then **Publish** the site.

---

## The code to paste

Replace `YOUR_VERCEL_URL` and `YOUR_API_KEY` (or `""`) then paste this into Footer Code:

```html
<script>
(function () {
  var API_BASE = "YOUR_VERCEL_URL";
  var API_KEY  = "YOUR_API_KEY";

  var messagesEl = document.getElementById("chat-messages");
  var inputEl    = document.getElementById("chat-input");
  var sendBtn    = document.getElementById("chat-send");

  if (!messagesEl || !inputEl || !sendBtn) {
    console.warn("Chat: missing #chat-messages, #chat-input, or #chat-send");
    return;
  }

  var history = [];

  function addMessage(role, content) {
    var bubble = document.createElement("div");
    bubble.className = "chat-mini chat-msg-" + role;
    bubble.style.cssText = role === "user"
      ? "margin-left:auto;margin-right:0;max-width:85%;padding:10px 14px;border-radius:12px;background:#1e3a6e;color:#fff;"
      : "margin-right:auto;margin-left:0;max-width:85%;padding:10px 14px;border-radius:12px;background:#e8e8ea;color:#1d1d1d;";
    bubble.textContent = content;
    messagesEl.appendChild(bubble);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  var staticKids = messagesEl.querySelectorAll(".chat-mini");
  for (var i = 0; i < staticKids.length; i++) staticKids[i].remove();
  addMessage("assistant", "Hello! I can help you find the perfect Estee Bedding mattress. What are you looking for?");

  function send() {
    var text = (inputEl.value || "").trim();
    if (!text) return;

    addMessage("user", text);
    history.push({ role: "user", content: text });
    inputEl.value = "";

    var thinking = document.createElement("div");
    thinking.className = "chat-mini chat-msg-assistant";
    thinking.style.cssText = "margin-right:auto;max-width:85%;padding:10px 14px;border-radius:12px;background:#e8e8ea;color:#777;font-style:italic;";
    thinking.textContent = "Thinking…";
    messagesEl.appendChild(thinking);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    var headers = { "Content-Type": "application/json" };
    if (API_KEY) headers["X-API-Key"] = API_KEY;

    fetch(API_BASE.replace(/\/$/, "") + "/chat", {
      method: "POST",
      headers: headers,
      body: JSON.stringify({ message: text, history: history })
    })
      .then(function (r) {
        if (!r.ok) return r.text().then(function (t) { throw new Error(t || r.statusText); });
        return r.json();
      })
      .then(function (data) {
        thinking.remove();
        var reply = (data && data.reply) || "";
        addMessage("assistant", reply);
        history.push({ role: "assistant", content: reply });
      })
      .catch(function (err) {
        thinking.remove();
        addMessage("assistant", "Something went wrong. Please try again.");
        console.error("Chat error", err);
      });
  }

  sendBtn.addEventListener("click", send);
  inputEl.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
})();
</script>
```

---

## Step 3: Publish and test

1. **Publish** the site in Webflow.
2. Open the live page (e.g. `https://estee-bedding.webflow.io/`).
3. Type in the chat and click **Ask..** (or press Enter). You should get a reply from your API.

If it doesn’t work, open the browser **Developer Tools** (F12) → **Console** and check for errors. Ensure the three IDs are on the correct elements and that `API_BASE` and `API_KEY` are set correctly.

---

## Optional: style new messages in Webflow

The script adds classes `chat-mini`, `chat-msg-user`, and `chat-msg-assistant`. You can add a **combo class** in Webflow (e.g. on a sample bubble) and style `.chat-msg-user` and `.chat-msg-assistant` so new messages match your design. The inline styles in the script will still work if you don’t change anything.
