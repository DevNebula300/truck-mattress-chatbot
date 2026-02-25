/**
 * Embeddable chat widget for Mattress & Truck RAG agent.
 * Usage: <script src="https://your-domain.com/widget.js" data-api-base="https://api.your-domain.com" data-widget-title="Help"></script>
 */
(function () {
  var script = document.currentScript;
  var apiBase = (script && script.getAttribute("data-api-base")) || "http://localhost:8000";
  var title = (script && script.getAttribute("data-widget-title")) || "Chat";

  var root = document.createElement("div");
  root.id = "mattress-rag-widget-root";
  root.innerHTML = [
    "<div id='mattress-rag-toggle' aria-label='Open chat'>",
    "  <svg width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'><path d='M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z'/></svg>",
    "</div>",
    "<div id='mattress-rag-panel' hidden>",
    "  <div id='mattress-rag-header'>",
    "    <span id='mattress-rag-title'>" + escapeHtml(title) + "</span>",
    "    <button id='mattress-rag-close' aria-label='Close'>×</button>",
    "  </div>",
    "  <div id='mattress-rag-messages'></div>",
    "  <div id='mattress-rag-input-area'>",
    "    <textarea id='mattress-rag-input' rows='1' placeholder='Ask about mattresses or truck fit...'></textarea>",
    "    <button id='mattress-rag-send' aria-label='Send'>Send</button>",
    "  </div>",
    "</div>"
  ].join("");

  var deepBlue = "#1e3a5f";
  var deepBlueHover = "#16304d";
  var style = document.createElement("style");
  style.textContent = [
    "#mattress-rag-widget-root { position: fixed; bottom: 24px; right: 24px; z-index: 999999; font-family: system-ui, -apple-system, \"Segoe UI\", sans-serif; font-size: 14px; }",
    "#mattress-rag-toggle { width: 56px; height: 56px; border-radius: 50%; background: " + deepBlue + "; color: #fff; display: flex; align-items: center; justify-content: center; cursor: pointer; box-shadow: 0 4px 16px rgba(30, 58, 95, 0.35); }",
    "#mattress-rag-toggle:hover { background: " + deepBlueHover + "; box-shadow: 0 6px 20px rgba(30, 58, 95, 0.4); }",
    "#mattress-rag-panel { position: absolute; bottom: 72px; right: 0; width: 400px; max-width: calc(100vw - 48px); height: 520px; max-height: 75vh; background: #fff; border-radius: 16px; box-shadow: 0 12px 40px rgba(0,0,0,0.12); display: flex; flex-direction: column; overflow: hidden; }",
    "#mattress-rag-header { padding: 16px 18px; background: " + deepBlue + "; color: #fff; display: flex; justify-content: space-between; align-items: center; }",
    "#mattress-rag-title { font-weight: 600; font-size: 1rem; }",
    "#mattress-rag-close { background: none; border: none; color: #fff; font-size: 24px; cursor: pointer; line-height: 1; padding: 0 4px; opacity: 0.9; }",
    "#mattress-rag-close:hover { opacity: 1; }",
    "#mattress-rag-messages { flex: 1; overflow-y: auto; padding: 18px; display: flex; flex-direction: column; gap: 14px; background: #f5f5f7; }",
    "#mattress-rag-messages .msg { max-width: 88%; padding: 12px 16px; border-radius: 14px; line-height: 1.5; }",
    "#mattress-rag-messages .msg.user { align-self: flex-end; background: " + deepBlue + "; color: #fff; }",
    "#mattress-rag-messages .msg.assistant { align-self: flex-start; background: #e8e8ea; color: #1d1d1d; border: none; }",
    "#mattress-rag-messages .msg.assistant .msg-heading { font-weight: 700; font-size: 0.95rem; margin: 0 0 6px 0; color: " + deepBlue + "; }",
    "#mattress-rag-messages .msg.assistant .msg-highlight { font-weight: 700; color: " + deepBlue + "; }",
    "#mattress-rag-messages .msg.assistant.thinking { color: #666; font-style: italic; }",
    "#mattress-rag-input-area { padding: 14px; background: #fff; border-top: 1px solid #e0e0e0; display: flex; gap: 10px; align-items: flex-end; }",
    "#mattress-rag-input { flex: 1; resize: none; border: 1px solid #d0d0d0; border-radius: 10px; padding: 12px 14px; font: inherit; min-height: 46px; max-height: 120px; }",
    "#mattress-rag-input:focus { outline: none; border-color: " + deepBlue + "; }",
    "#mattress-rag-send { padding: 12px 18px; background: " + deepBlue + "; color: #fff; border: none; border-radius: 10px; cursor: pointer; font-weight: 500; font-size: 0.95rem; }",
    "#mattress-rag-send:hover { background: " + deepBlueHover + "; }",
    "#mattress-rag-send:disabled { opacity: 0.6; cursor: not-allowed; }"
  ].join("\n");

  document.head.appendChild(style);
  document.body.appendChild(root);

  var panel = document.getElementById("mattress-rag-panel");
  var messagesEl = document.getElementById("mattress-rag-messages");
  var inputEl = document.getElementById("mattress-rag-input");
  var sendBtn = document.getElementById("mattress-rag-send");

  function escapeHtml(s) {
    var div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  var WELCOME_MSG = "Hello! I can help you find the perfect Estee Bedding mattress. What are you looking for?";

  function openPanel() {
    panel.hidden = false;
    inputEl.focus();
  }
  function closePanel() {
    panel.hidden = true;
  }

  document.getElementById("mattress-rag-toggle").onclick = openPanel;
  document.getElementById("mattress-rag-close").onclick = closePanel;

  function renderMarkdown(text) {
    if (!text) return "";
    var s = escapeHtml(text);
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong class=\"msg-highlight\">$1</strong>");
    s = s.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    s = s.replace(/^##\s+(.+)$/gm, "<div class=\"msg-heading\">$1</div>");
    s = s.replace(/\n/g, "<br>");
    return s;
  }

  function addMessage(role, content, className) {
    var div = document.createElement("div");
    div.className = "msg " + role + (className ? " " + className : "");
    if (role === "assistant" && !className) {
      div.innerHTML = renderMarkdown(content);
      div.setAttribute("data-raw-content", content);
    } else {
      div.textContent = content;
    }
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  addMessage("assistant", WELCOME_MSG);

  function send() {
    var text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = "";
    addMessage("user", text);
    var thinking = addMessage("assistant", "Thinking…", "thinking");
    sendBtn.disabled = true;

    var history = [];
    var msgNodes = messagesEl.querySelectorAll(".msg:not(.thinking)");
    for (var i = 0; i < msgNodes.length - 1; i++) {
      var node = msgNodes[i];
      var content = node.getAttribute("data-raw-content");
      if (content == null) content = node.textContent;
      history.push({ role: node.classList.contains("user") ? "user" : "assistant", content: content });
    }

    fetch(apiBase.replace(/\/$/, "") + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, history: history })
    })
      .then(function (r) {
        if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || r.statusText); });
        return r.json();
      })
      .then(function (data) {
        thinking.remove();
        addMessage("assistant", data.reply || "");
      })
      .catch(function (err) {
        thinking.remove();
        addMessage("assistant", "Sorry, something went wrong: " + (err.message || "Please try again."));
      })
      .finally(function () {
        sendBtn.disabled = false;
        messagesEl.scrollTop = messagesEl.scrollHeight;
      });
  }

  sendBtn.onclick = send;
  inputEl.onkeydown = function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };
})();
