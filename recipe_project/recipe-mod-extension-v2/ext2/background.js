// background.js — Service Worker (module)
console.log("[RecipeMod] background.js started");

const API_BASE = "https://danielddds-hebrew-recipe-mod-api.hf.space";

async function predictSingle(text) {
  const postRes = await fetch(`${API_BASE}/gradio_api/call/v2/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });
  const { event_id } = await postRes.json();
  const getRes = await fetch(`${API_BASE}/gradio_api/call/predict/${event_id}`);
  const raw = await getRes.text();
  let predictions = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("data:")) {
      try {
        const parsed = JSON.parse(line.slice(5));
        if (Array.isArray(parsed) && parsed.length > 0 && parsed[0].results) {
          predictions = parsed[0].results;
        }
      } catch (_) {}
    }
  }
  return predictions;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  console.log("[RecipeMod] Message received:", msg.type);

  if (msg.type === "PING") {
    sendResponse({ ok: true });
    return false;
  }

  if (msg.type === "PREDICT_BATCH") {
    (async () => {
      try {
        const data = [];
        for (const { text, index } of msg.comments) {
          const preds = await predictSingle(text);
          data.push({ text, index, preds });
        }
        console.log("[RecipeMod] Predictions ready, sending response");
        sendResponse({ ok: true, data });
      } catch (err) {
        console.error("[RecipeMod] error:", err.message);
        sendResponse({ ok: false, error: err.message });
      }
    })();
    return true; // keep channel open
  }
});