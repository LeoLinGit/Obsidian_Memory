# -*- coding: utf-8 -*-
"""
context_manager.py —— 智能上下文管理（滑动窗口 + 早期摘要 + token 截断）

组装最终发给 Ollama 的完整 prompt，按优先级：
    System Prompt（第一优先级，永不截断）
    RAG 检索到的历史记忆（第二优先级，超预算时优先被截断）
    早期对话摘要（第三优先级，长对话时用摘要替代更早的消息，避免 AI 遗忘开头）
    当前会话最近 N 条消息（第四优先级）
    用户当前问题（永不截断）
"""
import hashlib
import json
import logging

import requests

import db
import rag

logger = logging.getLogger("memflow.context")

SYSTEM_PROMPT = "你是本地知识库里的 AI 记忆助手，请始终使用简体中文、简洁直接地回答用户的问题。"

RECENT_MESSAGE_COUNT = 8   # 滑动窗口大小：最近 8 条消息（旧 build_prompt 使用）
MAX_TOKENS = 4096          # token 预算上限（旧 build_prompt 使用）
TOP_K = 5                  # RAG 检索 top-k

# ---- 早期摘要相关配置 ----
OLLAMA_BASE = "http://127.0.0.1:11434"   # 与 app.py 中的 OLLAMA_BASE 保持一致
SUMMARY_TIMEOUT = 30                     # 摘要生成超时（秒），避免阻塞聊天过久
MAX_RECENT = 100                         # build_prompt_with_summary 默认滑动窗口大小（放大到 100，几乎不再触发早期摘要）
MAX_CONTEXT_TOKENS = 16000               # build_prompt_with_summary 默认 token 预算（配合 Ollama num_ctx=32768）
SUMMARY_PROMPT = (
    "请将以下对话内容总结为一段简洁的摘要，保留关键信息、决策和结论，不超过200字：\n\n"
)


def estimate_tokens(text):
    """简单估算 token 数：内容长度 × 0.75（中英混合的粗略估算）。"""
    return int(len(text or "") * 0.75)


def truncate_to_token_limit(messages, max_tokens=4096):
    """
    按 token 预算截断消息列表。
    规则：
      1) 首条 system 与末条 query（当前问题）永不截断；
      2) 超预算时优先截断注入的上下文（RAG 记忆 tag=="rag" / 早期摘要 tag=="summary"）；
      3) 仍超预算则从最旧的历史消息（tag=="history"）开始丢弃。
    返回新的消息列表。
    """
    if not messages:
        return []

    def _total(msgs):
        return sum(estimate_tokens(m.get("content", "")) for m in msgs)

    result = [dict(m) for m in messages]
    if _total(result) <= max_tokens:
        return result

    # 1) 先去掉注入的上下文（RAG 记忆 / 早期摘要）
    result = [m for m in result if m.get("tag") not in ("rag", "summary")]

    # 2) 仍超预算：丢弃最旧的历史消息（索引 1，保留首条 system 与末条 query）
    while _total(result) > max_tokens and len(result) > 2:
        result.pop(1)

    return result


def build_prompt(conversation_id, user_query):
    """
    组装最终发给 Ollama 的完整消息列表。
    返回 list[{"role","content"}]（已去掉内部 tag 字段）。
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 1) RAG 记忆（第二优先级）
    if rag.is_available():
        try:
            hits = rag.search_similar_messages(user_query, top_k=TOP_K)
            if hits:
                messages.append({"role": "system", "content": _format_memory(hits), "tag": "rag"})
        except Exception as e:
            logger.warning("RAG 检索失败，忽略：%s", e)

    # 2) 滑动窗口历史（第三优先级，最近 8 条）
    try:
        history = db.get_recent_messages(conversation_id, limit=RECENT_MESSAGE_COUNT)
    except Exception as e:
        logger.warning("读取历史消息失败：%s", e)
        history = []
    for m in history:
        messages.append({"role": m["role"], "content": m["content"], "tag": "history"})

    # 3) 用户当前问题
    messages.append({"role": "user", "content": user_query, "tag": "query"})

    # 4) 按 token 预算截断
    truncated = truncate_to_token_limit(messages, max_tokens=MAX_TOKENS)
    # 去掉内部 tag，只保留 Ollama 需要的 role / content
    return [{"role": m["role"], "content": m["content"]} for m in truncated]


def _format_memory(hits):
    """把检索结果格式化为【相关记忆】文本块，按来源区分标注。"""
    lines = ["【相关记忆】"]
    role_label = {"user": "用户", "assistant": "AI"}
    for h in hits:
        content = (h.get("content") or "").strip()
        if not content:
            continue
        if h.get("source") == "obsidian":
            # Obsidian 笔记：标注文件名
            name = h.get("file_name") or "未命名"
            lines.append("- [笔记: %s] %s" % (name, content))
        else:
            # 聊天记录：标注历史对话，并保留角色与日期
            label = role_label.get(h.get("role"), "")
            date = (h.get("timestamp") or "")[:10]
            tag = "历史对话"
            if date:
                tag += " " + date
            if label:
                tag += " " + label
            lines.append("- [%s] %s" % (tag, content))
    return "\n".join(lines)


# ==================== 早期消息摘要 ====================
_summary_cache = {}   # conversation_id -> {"hash": str, "summary": str}，避免重复生成


def _messages_hash(messages):
    """对消息内容列表生成稳定哈希，作为摘要缓存 key（内容不变则不重新生成）。"""
    return hashlib.md5(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()


def generate_summary(messages, model_name):
    """
    调用 Ollama /api/generate，把一组旧消息总结成一段简洁摘要（不超过 200 字）。
    失败（连接失败 / 超时 / 模型未加载）时返回降级摘要，绝不抛异常，保证聊天可用。
    """
    if not messages:
        return ""
    lines = []
    for m in messages:
        role = "用户" if (m.get("role") == "user") else "AI"
        content = (m.get("content") or "").strip()
        if content:
            lines.append("%s: %s" % (role, content))
    dialogue = "\n".join(lines)
    if not dialogue:
        return ""

    prompt = SUMMARY_PROMPT + dialogue
    try:
        resp = requests.post(
            OLLAMA_BASE + "/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=SUMMARY_TIMEOUT,
        )
        if resp.status_code == 200:
            summary = (resp.json().get("response") or "").strip()
            if summary:
                return summary
    except requests.exceptions.RequestException as e:
        logger.warning("摘要生成失败（Ollama 不可用或超时）：%s", e)

    # 降级摘要：明确告知有 N 条消息被省略
    return "[早期对话摘要不可用，共 %d 条消息被省略]" % len(messages)


def _get_or_build_summary(conversation_id, early_messages, model_name):
    """
    按 conversation_id 缓存早期摘要：消息内容哈希没变则直接复用缓存，否则重新生成。
    （用内容哈希替代消息总数做失效信号——总数每轮必然增加，会导致缓存永远不命中。）
    """
    key = _messages_hash(early_messages)
    cached = _summary_cache.get(conversation_id)
    if cached and cached.get("hash") == key:
        return cached["summary"]
    summary = generate_summary(early_messages, model_name)
    _summary_cache[conversation_id] = {"hash": key, "summary": summary}
    return summary


def build_prompt_with_summary(conversation_id, user_message, model_name,
                              max_recent=MAX_RECENT, max_context_tokens=MAX_CONTEXT_TOKENS,
                              include_rag=False):
    """
    组装最终发给 Ollama 的完整消息列表（滑动窗口 + 早期摘要）。

    优先级（返回结构与 build_prompt 一致，list[{"role","content"}]）：
      1) System Prompt
      2) RAG 检索到的相关记忆（仅 include_rag=True 时注入，默认关闭、改由 memory_search 工具按需检索）
      3) 早期对话摘要（仅当对话超过 max_recent 条时）
      4) 最近 max_recent 条完整消息
      5) 当前用户问题
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 1) RAG 记忆（默认关闭：改由模型通过 memory_search 工具按需检索，避免与工具调用重复。
    #    传 include_rag=True 可恢复旧的强制注入行为）
    if include_rag and rag.is_available():
        try:
            hits = rag.search_similar_messages(user_message, top_k=TOP_K)
            if hits:
                messages.append({"role": "system", "content": _format_memory(hits), "tag": "rag"})
        except Exception as e:
            logger.warning("RAG 检索失败，忽略：%s", e)

    # 2) 读取该对话全部历史消息（按时间升序，不含当前这条用户消息）
    try:
        conv = db.get_conversation(conversation_id)
        history = (conv or {}).get("messages") or []
    except Exception as e:
        logger.warning("读取对话消息失败：%s", e)
        history = []

    total = len(history) + 1   # 历史 + 当前用户消息

    # 3) 判断是否需要早期摘要
    summary_text = ""
    if total > max_recent and history:
        # 最近窗口保留最近 (max_recent - 1) 条历史，当前用户消息占掉最后 1 个位置
        keep = max(max_recent - 1, 0)
        early = history[:-keep] if keep > 0 else history
        recent = history[-keep:] if keep > 0 else []
        if early:
            summary_text = _get_or_build_summary(conversation_id, early, model_name)
    else:
        recent = history

    # 4) 早期摘要（第三优先级，超预算时优先截断）
    if summary_text:
        messages.append({"role": "system", "content": "【早期对话摘要】\n" + summary_text, "tag": "summary"})

    # 5) 最近完整消息（含工具调用轨迹，供模型在多轮对话中引用之前的工具结果）
    for m in recent:
        msg = {"role": m["role"], "content": m["content"], "tag": "history"}
        if m.get("tool_calls"):
            msg["tool_calls"] = m["tool_calls"]
        if m.get("tool_name"):
            msg["tool_name"] = m["tool_name"]
        messages.append(msg)

    # 6) 当前用户问题
    messages.append({"role": "user", "content": user_message, "tag": "query"})

    # 7) 按 token 预算截断（system 与 query 永不截断，先丢 rag/summary，再丢最旧 history）
    truncated = truncate_to_token_limit(messages, max_tokens=max_context_tokens)
    result = []
    for m in truncated:
        out = {"role": m["role"], "content": m["content"]}
        if m.get("tool_calls"):
            out["tool_calls"] = m["tool_calls"]
        if m.get("tool_name"):
            out["tool_name"] = m["tool_name"]
        result.append(out)
    return result
