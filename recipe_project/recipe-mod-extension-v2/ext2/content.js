// content.js — Main content script (v2.4 – infinite‑scroll fix)
// Scrolls the YT comments container, waits for new threads, then collects all.
console.log("[RecipeMod] content.js injected");

const HEBREW_REGEX = /[\u0590-\u05FF]/;
const MAX_COMMENTS = 300;   // high safety cap

const ASPECT_LABELS = {
  SUBSTITUTION: { he: "תחליפים", icon: "⇄", color: "#FF6B35" },
  ADDITION:     { he: "תוספות",  icon: "+", color: "#4CAF50" },
  QUANTITY:     { he: "כמויות",  icon: "⚖", color: "#2196F3" },
  TECHNIQUE:    { he: "טכניקות", icon: "✦", color: "#9C27B0" },
};

let sidebarEl        = null;
let scanDone         = false;
let currentUrl       = location.href;
let scanTimeout      = null;
let innerScanTimeout = null;

// ── DOM helpers ──
function el(tag, props = {}) {
  const e = document.createElement(tag);
  if (props.className) e.className = props.className;
  if (props.id)        e.id = props.id;
  if (props.text)      e.textContent = props.text;
  if (props.title)     e.title = props.title;
  if (props.style)     e.style.cssText = props.style;
  return e;
}

function getVideoTitle() {
  return document.querySelector("h1 yt-formatted-string, h1.ytd-watch-metadata yt-formatted-string")
             ?.innerText?.trim()
      || document.title.replace(" - YouTube","").trim()
      || "מתכון";
}

function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

// ── Infinite‑scroll loading ──
async function loadAllComments() {
  const commentsEl = document.querySelector("ytd-comments");
  if (!commentsEl) {
    console.warn("[RecipeMod] ytd-comments not found, scrolling full page instead");
    return fallbackScroll();
  }

  // Scroll the comments container into view first
  commentsEl.scrollIntoView({ behavior: "smooth", block: "start" });
  await sleep(1500);

  const container = commentsEl.querySelector("#contents") || commentsEl;
  let lastThreadCount = 0;
  let stableCount = 0;

  // Keep scrolling to bottom of the container until no new threads appear 3 times in a row
  for (let i = 0; i < 50; i++) {
    container.scrollTo(0, container.scrollHeight);
    await sleep(1200);

    // Click any "View more replies" that may have appeared (inside threads)
    const replyButtons = document.querySelectorAll(
      "ytd-button-renderer#more-replies button, ytd-comment-replies-renderer #more-replies button"
    );
    for (const btn of replyButtons) {
      if (btn.offsetParent !== null) { btn.click(); await sleep(500); }
    }

    const currentCount = document.querySelectorAll("ytd-comment-thread-renderer").length;
    if (currentCount === lastThreadCount) {
      stableCount++;
      if (stableCount >= 3) break;   // no more loading
    } else {
      stableCount = 0;
    }
    lastThreadCount = currentCount;
  }

  // Final wait for any late renders
  await sleep(1500);
}

async function fallbackScroll() {
  let prevCount = 0, stable = 0;
  for (let i = 0; i < 30; i++) {
    window.scrollTo(0, document.body.scrollHeight);
    await sleep(1500);
    const currentCount = document.querySelectorAll("ytd-comment-thread-renderer").length;
    if (currentCount === prevCount) {
      stable++;
      if (stable >= 3) break;
    } else {
      stable = 0;
    }
    prevCount = currentCount;
  }
  await sleep(1500);
}

// ── Collect ──
function collectComments() {
  const comments = [];
  const textNodes = document.querySelectorAll(
    "ytd-comment-thread-renderer #content-text, " +
    "ytd-comment-replies-renderer #content-text"
  );
  textNodes.forEach((node, i) => {
    if (i >= MAX_COMMENTS) return;
    const text = node.innerText?.trim();
    if (text && text.length > 10 && HEBREW_REGEX.test(text)) {
      comments.push({ text, index: i });
    }
  });
  console.log(`[RecipeMod] Collected ${comments.length} comments`);
  return comments;
}

// ── Parsing ──
function cleanWord(word) {
  return word.replace(/##/g, "").replace(/\s+/g, " ").trim();
}

function parseEntities(results) {
  const grouped = { SUBSTITUTION: [], ADDITION: [], QUANTITY: [], TECHNIQUE: [] };
  for (const { text, preds } of results) {
    if (!Array.isArray(preds)) continue;
    for (const item of preds) {
      const label = item.entity_group;
      if (!grouped[label] || item.score < 0.45) continue;
      const word = cleanWord(item.word);
      if (word.length < 2) continue;
      if (!grouped[label].find(e => e.text === word)) {
        grouped[label].push({ text: word, score: item.score, comment: text });
      }
    }
  }
  return grouped;
}

// ── Sidebar UI ──
function createSidebar() {
  const sidebar = el("div", { id: "recipe-mod-sidebar" });

  const header = el("div", { className: "rms-header" });
  header.appendChild(el("span", { className: "rms-logo", text: "🧑‍🍳" }));
  const titleWrap = el("div", { className: "rms-title-wrap" });
  titleWrap.appendChild(el("div", { className: "rms-title", text: "Recipe Mods" }));
  const subtitle = el("div", { className: "rms-subtitle", id: "rms-video-title" });
  titleWrap.appendChild(subtitle);
  header.appendChild(titleWrap);
  const closeBtn = el("button", { className: "rms-close", id: "rms-close", text: "✕" });
  closeBtn.addEventListener("click", () => sidebar.classList.add("rms-hidden"));
  header.appendChild(closeBtn);
  sidebar.appendChild(header);

  const body = el("div", { className: "rms-body", id: "rms-body" });
  const loading = el("div", { className: "rms-loading", id: "rms-loading" });
  loading.appendChild(el("div", { className: "rms-spinner" }));
  loading.appendChild(el("div", { className: "rms-loading-text", text: "סורק תגובות…" }));
  loading.appendChild(el("div", { className: "rms-loading-sub", id: "rms-loading-sub", text: "טוען מודל AI" }));
  body.appendChild(loading);
  sidebar.appendChild(body);
  document.body.appendChild(sidebar);
  return sidebar;
}

function showLoading(msg) { const sub = document.getElementById("rms-loading-sub"); if (sub) sub.textContent = msg; }

function showResults(grouped, commentCount) {
  const body = document.getElementById("rms-body");
  if (!body) return;
  while (body.firstChild) body.removeChild(body.firstChild);

  const totalMods = Object.values(grouped).reduce((a, b) => a + b.length, 0);
  if (totalMods === 0) {
    const empty = el("div", { className: "rms-empty" });
    empty.appendChild(el("div", { className: "rms-empty-icon", text: "🔍" }));
    empty.appendChild(el("div", { className: "rms-empty-text", text: `לא נמצאו שינויי מתכון ב-${commentCount} תגובות` }));
    body.appendChild(empty);
    return;
  }

  body.appendChild(el("div", { className: "rms-meta", text: `${totalMods} שינויים ב-${commentCount} תגובות` }));

  for (const [aspect, info] of Object.entries(ASPECT_LABELS)) {
    const items = grouped[aspect];
    if (!items || items.length === 0) continue;
    const section = el("div", { className: "rms-section" });
    const sHeader = el("div", { className: "rms-section-header" });
    sHeader.style.setProperty("--accent", info.color);
    sHeader.appendChild(el("span", { className: "rms-section-icon", text: info.icon }));
    sHeader.appendChild(el("span", { className: "rms-section-title", text: info.he }));
    sHeader.appendChild(el("span", { className: "rms-section-count", text: String(items.length) }));
    section.appendChild(sHeader);

    const itemsWrap = el("div", { className: "rms-section-items" });
    for (const item of items) {
      const row = el("div", { className: "rms-item", title: item.comment.slice(0, 120) });
      row.appendChild(el("span", { className: "rms-item-text", text: item.text }));
      const score = el("span", { className: "rms-item-score", text: Math.round(item.score * 100) + "%" });
      score.style.setProperty("--accent", info.color);
      row.appendChild(score);
      itemsWrap.appendChild(row);
    }
    section.appendChild(itemsWrap);
    body.appendChild(section);
  }
}

function showError(msg) {
  const body = document.getElementById("rms-body");
  if (!body) return;
  while (body.firstChild) body.removeChild(body.firstChild);
  body.appendChild(el("div", { className: "rms-error", text: msg }));
}

// ── Scan flow ──
async function scanComments() {
  if (scanDone) return;
  await loadAllComments();
  let comments = collectComments();
  if (comments.length === 0) { showError("לא נמצאו תגובות בעברית"); return; }
  showLoading(`מנתח ${comments.length} תגובות…`);
  chrome.runtime.sendMessage({ type: "PREDICT_BATCH", comments }, (response) => {
    if (chrome.runtime.lastError) { showError("טיימאאוט — המודל עדיין טוען. רענן את הדף בעוד 30 שניות."); return; }
    if (!response?.ok) { showError("שגיאה בטעינת המודל. נסה שוב."); return; }
    const grouped = parseEntities(response.data);
    showResults(grouped, comments.length);
    scanDone = true;
  });
}

function initSidebar() {
  if (sidebarEl) sidebarEl.remove();
  scanDone = false;
  sidebarEl = createSidebar();
  const titleEl = document.getElementById("rms-video-title");
  if (titleEl) titleEl.textContent = getVideoTitle();
  chrome.runtime.sendMessage({ type: "PING" });
  scanComments();
}

// ── SPA navigation ──
function checkPageChange() {
  if (location.href !== currentUrl) {
    currentUrl = location.href;
    scanDone = false;
    clearTimeout(scanTimeout);
    clearTimeout(innerScanTimeout);
    scanTimeout = setTimeout(() => {
      if (location.pathname === "/watch") {
        innerScanTimeout = setTimeout(() => {
          if (!document.getElementById("recipe-mod-sidebar")) initSidebar();
        }, 2000);
      }
    }, 1500);
  }
}

const navObserver = new MutationObserver(checkPageChange);
navObserver.observe(document.body, { childList: true, subtree: true });

chrome.runtime.sendMessage({ type: "PING" });
setTimeout(() => { if (location.pathname === "/watch") initSidebar(); }, 2500);