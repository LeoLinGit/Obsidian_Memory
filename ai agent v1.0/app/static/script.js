/* ============================================================
   MemFlow 前端交互脚本
   负责：会话列表管理、SSE 流式对话渲染、Markdown 渲染、
        深色/浅色主题切换、模型选择、移动端侧边栏。
   依赖（CDN）：marked（Markdown）、DOMPurify（XSS 过滤）、highlight.js（代码高亮）。
   ============================================================ */

// ===== 元素引用 =====
const sidebar = document.getElementById("sidebar");
const sidebarOverlay = document.getElementById("sidebar-overlay");
const conversationList = document.getElementById("conversation-list");
const messagesBox = document.getElementById("messages");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");
const newChatBtn = document.getElementById("new-chat-btn");
const clearBtn = document.getElementById("clear-btn");
const menuBtn = document.getElementById("menu-btn");
const headerModel = document.getElementById("header-model");
const modelSelect = document.getElementById("model-select");
const settingsBtn = document.getElementById("settings-btn");
const settingsPopover = document.getElementById("settings-popover");
const themeToggle = document.getElementById("theme-toggle");
const obsidianConfigBtn = document.getElementById("obsidian-config-btn");
const obsidianSyncBtn = document.getElementById("obsidian-sync-btn");
const exportAllBtn = document.getElementById("export-all-btn");
const obsidianModal = document.getElementById("obsidian-modal");
const obsidianPathInput = document.getElementById("obsidian-path-input");
const obsidianCancel = document.getElementById("obsidian-cancel");
const obsidianSave = document.getElementById("obsidian-save");
const toast = document.getElementById("toast");
const writeModal = document.getElementById("write-modal");
const writeModalPath = document.getElementById("write-modal-path");
const writeModalContent = document.getElementById("write-modal-content");
const writeAllow = document.getElementById("write-allow");
const writeDeny = document.getElementById("write-deny");

// ===== 常量与状态 =====
const STORAGE_MODEL = "memflow_model";
const STORAGE_THEME = "memflow_theme";
const DEFAULT_MODEL = "qwen2.5:7b";

let conversations = [];
let currentConversationId = null;
let currentHasMessages = false;
let isGenerating = false;
let currentAbort = null;

// ===== 工具函数 =====
function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function nearBottom() {
    return messagesBox.scrollHeight - messagesBox.scrollTop - messagesBox.clientHeight < 80;
}
function scrollToBottom() { messagesBox.scrollTop = messagesBox.scrollHeight; }

// ===== 主题切换 =====
function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    themeToggle.textContent = theme === "dark" ? "已开启" : "已关闭";
    localStorage.setItem(STORAGE_THEME, theme);
}
function initTheme() {
    const saved = localStorage.getItem(STORAGE_THEME) || "dark";
    applyTheme(saved);
}

// ===== Markdown 渲染 =====
function enhanceCodeBlocks(container) {
    if (!window.hljs) return;
    container.querySelectorAll("pre code").forEach((code) => {
        hljs.highlightElement(code);
    });
}

function renderMarkdown(contentEl, rawText) {
    let html;
    if (window.marked) {
        html = marked.parse(rawText);
        if (window.DOMPurify) html = DOMPurify.sanitize(html);
    } else {
        html = escapeHtml(rawText).replace(/\n/g, "<br>");
    }
    contentEl.innerHTML = html;
    enhanceCodeBlocks(contentEl);
    if (nearBottom()) scrollToBottom();
}

// ===== 消息渲染 =====
function createStreamingAIMessage() {
    const div = document.createElement("div");
    div.className = "message ai";
    const bubble = document.createElement("div");
    bubble.className = "bubble markdown";
    bubble.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    div.appendChild(bubble);
    messagesBox.appendChild(div);
    scrollToBottom();
    return bubble;
}

function appendUserMessage(text) {
    const div = document.createElement("div");
    div.className = "message user";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    div.appendChild(bubble);
    messagesBox.appendChild(div);
    scrollToBottom();
}

function renderWelcome() {
    messagesBox.innerHTML = "";
    const w = document.createElement("div");
    w.className = "welcome";
    w.innerHTML =
        '<div class="welcome-icon">🧠</div>' +
        "<h2>MemFlow 记忆助手</h2>" +
        "<p>基于本地 Ollama 运行，支持长期记忆检索。有什么可以帮你的？</p>";
    messagesBox.appendChild(w);
}

function renderMessages(messages) {
    messagesBox.innerHTML = "";
    if (!messages.length) { renderWelcome(); return; }
    for (const m of messages) {
        if (m.role === "user") {
            appendUserMessage(m.content);
        } else if (m.role === "tool" || (m.role === "assistant" && m.tool_calls)) {
            // 工具调用轨迹（工具结果 / 调用意图）：不在界面渲染，仅供后端多轮上下文与调试
            continue;
        } else {
            const bubble = createStreamingAIMessage();
            renderMarkdown(bubble, m.content);
        }
    }
    scrollToBottom();
}

// ===== 侧边栏 / 会话列表 =====
function toggleSidebar() {
    sidebar.classList.toggle("open");
    sidebarOverlay.classList.toggle("show");
}
function closeSidebar() {
    sidebar.classList.remove("open");
    sidebarOverlay.classList.remove("show");
}

async function refreshConversations() {
    conversations = await (await fetch("/api/conversations")).json();
    renderConversationList();
}

function renderConversationList() {
    conversationList.innerHTML = "";
    if (!conversations.length) {
        conversationList.innerHTML = '<div class="empty-hint">暂无对话</div>';
        return;
    }
    for (const c of conversations) {
        const item = document.createElement("div");
        item.className = "conversation-item" + (c.id === currentConversationId ? " active" : "");
        item.dataset.id = c.id;
        item.title = c.title || "新对话";

        const title = document.createElement("div");
        title.className = "conv-title";
        title.textContent = c.title || "新对话";

        const exp = document.createElement("button");
        exp.className = "conv-export";
        exp.textContent = "⬇";
        exp.title = "导出到 Obsidian";
        exp.addEventListener("click", (e) => {
            e.stopPropagation();
            exportConversation(c.id);
        });

        const del = document.createElement("button");
        del.className = "conv-delete";
        del.textContent = "✕";
        del.title = "删除对话";
        del.addEventListener("click", (e) => {
            e.stopPropagation();
            deleteConversation(c.id);
        });

        item.appendChild(title);
        item.appendChild(exp);
        item.appendChild(del);
        item.addEventListener("click", () => selectConversation(c.id));
        conversationList.appendChild(item);
    }
}

async function selectConversation(id) {
    currentConversationId = id;
    const res = await fetch("/api/conversations/" + id);
    if (!res.ok) return;
    const conv = await res.json();
    currentHasMessages = conv.messages.length > 0;
    renderMessages(conv.messages);
    renderConversationList();
    closeSidebar();
}

async function newConversation() {
    if (currentConversationId && !currentHasMessages) {
        renderMessages([]);
        renderConversationList();
        return;
    }
    const res = await fetch("/api/conversations", { method: "POST" });
    const conv = await res.json();
    currentConversationId = conv.id;
    currentHasMessages = false;
    renderMessages([]);
    await refreshConversations();
}

async function deleteConversation(id) {
    await fetch("/api/conversations/" + id, { method: "DELETE" });
    if (currentConversationId === id) {
        currentConversationId = null;
        currentHasMessages = false;
        renderMessages([]);
    }
    await refreshConversations();
    if (!currentConversationId && conversations.length) {
        await selectConversation(conversations[0].id);
    }
}

// ===== 发送 / 流式 =====
function setGenerating(gen) {
    isGenerating = gen;
    sendBtn.disabled = gen;
}

async function sendMessage() {
    if (isGenerating) return;
    const text = input.value.trim();
    if (!text) return;

    if (!currentConversationId) {
        const res = await fetch("/api/conversations", { method: "POST" });
        const conv = await res.json();
        currentConversationId = conv.id;
        currentHasMessages = false;
    }

    appendUserMessage(text);
    input.value = "";
    autoResize();
    setGenerating(true);

    const contentEl = createStreamingAIMessage();
    const messageDiv = contentEl.parentElement;
    let toolHint = null;
    let fullText = "";
    let firstToken = true;
    let renderQueued = false;

    const model = modelSelect.value || DEFAULT_MODEL;
    currentAbort = new AbortController();

    function flushRender() {
        renderQueued = false;
        renderMarkdown(contentEl, fullText);
    }
    function queueRender() {
        if (renderQueued) return;
        renderQueued = true;
        requestAnimationFrame(flushRender);
    }

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text, conversation_id: currentConversationId, model: model }),
            signal: currentAbort.signal
        });

        if (!response.ok) {
            showError(contentEl, "HTTP " + response.status);
            return;
        }
        if (!response.body) {
            showError(contentEl, "浏览器不支持流式读取。");
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            let idx;
            while ((idx = buffer.indexOf("\n\n")) !== -1) {
                const raw = buffer.slice(0, idx).trim();
                buffer = buffer.slice(idx + 2);
                if (raw.indexOf("data:") !== 0) continue;

                let event;
                try { event = JSON.parse(raw.slice(5).trim()); } catch (_) { continue; }

                if (event.type === "token") {
                    if (firstToken) { contentEl.innerHTML = ""; firstToken = false; }
                    fullText += event.content;
                    queueRender();
                } else if (event.type === "tool_call") {
                    firstToken = false;
                    if (!toolHint) {
                        toolHint = document.createElement("div");
                        toolHint.className = "tool-call-hint";
                        messageDiv.insertBefore(toolHint, contentEl);
                    }
                    const argsText = event.args && Object.keys(event.args).length
                        ? " · " + JSON.stringify(event.args) : "";
                    toolHint.innerHTML = "";
                    toolHint.textContent = "🔧 调用 " + (event.name || "工具") + argsText + " …";
                    scrollToBottom();
                } else if (event.type === "tool_result") {
                    firstToken = false;
                    if (!toolHint) {
                        toolHint = document.createElement("div");
                        toolHint.className = "tool-call-hint";
                        messageDiv.insertBefore(toolHint, contentEl);
                    }
                    const res = String(event.result || "");
                    const short = res.length > 300 ? res.slice(0, 300) + "…" : res;
                    const resultEl = document.createElement("div");
                    resultEl.className = "tool-result";
                    resultEl.textContent = "↳ " + short;
                    toolHint.appendChild(resultEl);
                    scrollToBottom();
                } else if (event.type === "confirm_write") {
                    firstToken = false;
                    showWriteConfirm(event);
                } else if (event.type === "done") {
                    currentHasMessages = true;
                    if (event.conversation_id) currentConversationId = event.conversation_id;
                    if (toolHint) {
                        toolHint.textContent = "🔧 已调用 " + (event.tool_used || "工具") + " 工具";
                    }
                    if (firstToken) { contentEl.innerHTML = ""; firstToken = false; }
                    renderMarkdown(contentEl, fullText);
                    await refreshConversations();
                } else if (event.type === "error") {
                    showError(contentEl, friendlyError(event));
                }
            }
        }
    } catch (error) {
        if (error.name !== "AbortError") {
            showError(contentEl, "无法连接到后端服务。请确认已运行 `python app.py`，且访问 http://127.0.0.1:5000。");
        } else {
            // 用户主动停止：保留已生成的部分
            renderMarkdown(contentEl, fullText);
        }
    } finally {
        setGenerating(false);
        currentAbort = null;
        if (nearBottom()) scrollToBottom();
    }
}

function showError(el, text) {
    el.closest(".message").classList.add("error");
    el.innerHTML = "";
    el.textContent = text;
    scrollToBottom();
}

function friendlyError(event) {
    switch (event.code) {
        case "ollama_down":
            return "Ollama 服务未启动。请在终端运行 `ollama serve` 后重试。";
        case "model_not_found":
            return "模型未安装：" + (event.detail || "") + "\n请在 `ollama list` 中确认，或换个已安装的模型。";
        default:
            return event.detail || "未知错误";
    }
}

// ===== Obsidian 导出 =====
let toastTimer = null;
function showToast(msg, type) {
    toast.textContent = msg;
    toast.className = "toast " + (type || "info");
    toast.hidden = false;
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toast.hidden = true; }, 4000);
}

async function openObsidianModal() {
    try {
        const res = await fetch("/api/obsidian/config");
        const data = await res.json();
        obsidianPathInput.value = data.obsidian_vault_path || "";
    } catch (e) {
        obsidianPathInput.value = "";
    }
    obsidianModal.hidden = false;
    obsidianPathInput.focus();
}

async function saveObsidianConfig() {
    const path = obsidianPathInput.value.trim();
    if (!path) { showToast("路径不能为空", "error"); return; }
    try {
        const res = await fetch("/api/obsidian/config", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ obsidian_vault_path: path }),
        });
        const data = await res.json();
        if (res.ok && data.status === "ok") {
            obsidianModal.hidden = true;
            showToast("已保存 Obsidian 路径：" + path, "success");
        } else {
            showToast("保存失败：" + (data.detail || res.status), "error");
        }
    } catch (e) {
        showToast("保存失败：无法连接后端", "error");
    }
}

async function exportConversation(id) {
    try {
        const res = await fetch("/api/obsidian/export/" + id);
        const data = await res.json();
        if (res.ok && data.status === "ok") {
            showToast("已导出（" + data.message_count + " 条消息）：" + data.file_path, "success");
        } else {
            showToast("导出失败：" + (data.detail || res.status), "error");
        }
    } catch (e) {
        showToast("导出失败：无法连接后端", "error");
    }
}

async function exportAll() {
    try {
        const res = await fetch("/api/obsidian/export-all");
        const data = await res.json();
        if (res.ok && data.status === "ok") {
            showToast("已导出 " + data.exported + " 个对话", "success");
        } else {
            showToast("导出失败：" + (data.detail || res.status), "error");
        }
    } catch (e) {
        showToast("导出失败：无法连接后端", "error");
    }
}

async function syncObsidian() {
    showToast("正在同步笔记…", "info");
    try {
        const res = await fetch("/api/obsidian/sync");
        const data = await res.json();
        if (data.status === "ok") {
            showToast("同步完成：新增 " + data.added + "，更新 " + data.updated +
                "，删除 " + data.deleted + "，未变 " + data.unchanged, "success");
        } else {
            showToast("同步失败：" + (data.detail || res.status), "error");
        }
    } catch (e) {
        showToast("同步失败：无法连接后端", "error");
    }
}

// ===== file_write 确认 =====
let pendingWriteId = null;

function showWriteConfirm(evt) {
    pendingWriteId = evt.id;
    writeModalPath.textContent = "路径：" + (evt.path || "") + "（模式：" + (evt.mode || "write") + "）";
    writeModalContent.textContent = evt.content || "";
    writeModal.hidden = false;
}

async function respondWrite(allow) {
    if (!pendingWriteId) return;
    const id = pendingWriteId;
    pendingWriteId = null;
    writeModal.hidden = true;
    try {
        await fetch("/api/confirm_write", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id: id, allow: allow })
        });
    } catch (e) {
        console.warn("确认写入请求失败", e);
    }
}

// ===== 输入框自适应 =====
function autoResize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
}

// ===== 模型选择 =====
async function initModelSelect() {
    let models = [];
    try {
        const res = await fetch("/api/models");
        const data = await res.json();
        models = data.models || [];
    } catch (e) {
        console.warn("获取模型列表失败，使用默认模型", e);
    }
    if (!models.length) models = [DEFAULT_MODEL];

    models.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        modelSelect.appendChild(opt);
    });

    const saved = localStorage.getItem(STORAGE_MODEL);
    modelSelect.value = models.includes(saved) ? saved : models[0];
    headerModel.textContent = modelSelect.value;
    modelSelect.addEventListener("change", () => {
        localStorage.setItem(STORAGE_MODEL, modelSelect.value);
        headerModel.textContent = modelSelect.value;
    });
}

// ===== 事件绑定 =====
newChatBtn.addEventListener("click", newConversation);
clearBtn.addEventListener("click", newConversation);
menuBtn.addEventListener("click", toggleSidebar);
sidebarOverlay.addEventListener("click", closeSidebar);
sendBtn.addEventListener("click", () => {
    if (isGenerating && currentAbort) currentAbort.abort();
    else sendMessage();
});
input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
input.addEventListener("input", autoResize);
settingsBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    settingsPopover.hidden = !settingsPopover.hidden;
});
document.addEventListener("click", (e) => {
    if (!settingsPopover.hidden && !settingsPopover.contains(e.target) && e.target !== settingsBtn) {
        settingsPopover.hidden = true;
    }
});
themeToggle.addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(next);
});

obsidianConfigBtn.addEventListener("click", openObsidianModal);
obsidianSyncBtn.addEventListener("click", syncObsidian);
exportAllBtn.addEventListener("click", exportAll);
obsidianCancel.addEventListener("click", () => { obsidianModal.hidden = true; });
obsidianSave.addEventListener("click", saveObsidianConfig);
writeAllow.addEventListener("click", () => respondWrite(true));
writeDeny.addEventListener("click", () => respondWrite(false));
obsidianModal.addEventListener("click", (e) => {
    if (e.target === obsidianModal) obsidianModal.hidden = true;
});
obsidianPathInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") saveObsidianConfig();
});

// ===== 初始化 =====
async function init() {
    initTheme();
    await initModelSelect();
    await refreshConversations();
    if (conversations.length) {
        await selectConversation(conversations[0].id);
    } else {
        renderMessages([]);
    }
}
document.addEventListener("DOMContentLoaded", init);
