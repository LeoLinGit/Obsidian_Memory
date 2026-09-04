# -*- coding: utf-8 -*-
"""
app.py —— MemFlow 后端主入口
Flask 应用：REST API + SSE 流式对话 + RAG 长期记忆 + 工具调用能力
"""
import hashlib
import json
import logging
import os
import re
import threading
import time
import requests
from urllib.parse import quote
from html import unescape
from flask import Flask, request, jsonify, render_template, Response, make_response
from datetime import datetime  # 新增：用于获取时间工具

import db
import rag
from context_manager import build_prompt_with_summary

logger = logging.getLogger(__name__)
app = Flask(__name__)

# ---------- 基础配置 ----------
# ★ 模型白名单：前端下拉列表只显示这里的模型，增删模型直接改这个列表。
ALLOWED_MODELS = ["qwen2.5:7b", "qwen3.5:9b"]
# Ollama 基础地址
OLLAMA_BASE = "http://127.0.0.1:11434"
OLLAMA_URL = OLLAMA_BASE + "/api/chat"
DEFAULT_MODEL = ALLOWED_MODELS[0]
# ReAct 推理循环最大轮次
MAX_REACT_ROUNDS = 5
# 熔断阈值：连续工具调用失败（参数错误/执行异常）达到该次数则终止循环，避免无效循环浪费资源
MAX_CONSECUTIVE_TOOL_FAILURES = 2
# Ollama 请求的上下文窗口（token 数）：qwen2.5:7b / qwen3.5:9b 均支持 32768，
# 显式指定避免 Ollama 默认 4096 截断长对话历史（配合 context_manager 的 MAX_CONTEXT_TOKENS）。
OLLAMA_NUM_CTX = 32768
# ReAct 规划与反思系统指令（作为 messages 第一条注入，引导 7b 模型拆解任务 + 失败重试）
REACT_SYSTEM_PROMPT = (
    "你是一个智能助手。面对复杂任务时，请先简要列出执行计划（格式：计划：1.xxx 2.xxx 3.xxx），"
    "然后逐步执行，每一步可以调用工具获取信息。"
    "如果工具调用失败，请分析失败原因，调整参数或使用其他工具重试，不要直接将错误抛给用户。"
    "当用户询问当前时间、日期时，必须调用 get_current_time 工具获取准确时间，禁止凭空编造。"
    "当用户询问实时信息（新闻、天气等）时，必须调用 web_search 工具，禁止凭记忆回答。"
    "当用户一次提出多个问题时，必须逐一分析每个问题，分别调用对应的工具，不能遗漏任何一个。"
    "每个问题独立处理，先列出所有需要处理的问题，再逐个执行。"
    "当用户的请求包含多个子任务时，必须先列出执行计划（比如：1. 检索本周笔记 2. 筛选关键内容 3. 生成周报），"
    "再逐步调用工具执行，每完成一步简要说明进度。"
)
# file_read 相对路径的基准目录：项目根目录（app.py 位于 app/ 子目录，其上一级即项目根）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# file_read 单次读取最大字符数（超出则截断）
FILE_READ_MAX_CHARS = 10000

def get_ollama_models():
    """ 获取 Ollama 模型列表（带重试机制）。 """
    for i in range(5):
        try:
            resp = requests.get(OLLAMA_BASE + "/api/tags", timeout=5)
            if resp.status_code == 200:
                return resp.json().get("models", [])
        except requests.exceptions.ConnectionError:
            if i == 4:
                logger.error("连接 Ollama 失败：请检查 Ollama 是否已启动 (http://127.0.0.1:11434)")
            else:
                time.sleep(1)
        except Exception as e:
            logger.warning("获取 Ollama 模型列表异常: %s", e)
            break
    return []

# ---------- 安全：刻意不开启 CORS ----------
# 前端由本服务同源提供（"/" 与 "/static"），无需跨域。若设置 Access-Control-Allow-Origin: *，
# 浏览器中任意第三方网页即可跨站读取本地聊天记录、触发笔记导出（本地服务 + CORS:* 是常见隐私泄露面）。

def make_title(text):
    """用 AI 回复的第一句话生成标题。"""
    text = (text or "").strip()
    if not text: return db.DEFAULT_TITLE
    first = re.split(r"[。！？!?\n\r]+", text)[0].strip()
    if not first: first = text
    first = first.lstrip("#*- `>").strip()
    if not first: return db.DEFAULT_TITLE
    return first[:20] + ("…" if len(first) > 20 else "")

def sse(payload):
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

def _parse_args(args):
    """把 Ollama 返回的 arguments 归一化为 dict（可能是 dict 或 JSON 字符串）。"""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def _tool_memory_search(args):
    """memory_search 工具：按需检索 Obsidian 笔记 / 历史记忆。"""
    if not rag.is_available():
        return "记忆检索不可用（ChromaDB 未初始化或依赖缺失）。请改用其他工具，或直接基于已有知识回答。"
    query = (args.get("query") or "").strip()
    if not query:
        return "未提供检索关键词。请提供 query 参数后重新调用 memory_search。"
    hits = rag.search_similar_messages(query, top_k=5)
    if not hits:
        return "未找到相关记忆。请尝试更换更简洁或更具体的检索关键词后重试。"
    lines = []
    for h in hits:
        content = (h.get("content") or "").strip()
        if not content:
            continue
        if h.get("source") == "obsidian":
            lines.append("[笔记:%s] %s" % (h.get("file_name") or "未命名", content))
        else:
            label = "用户" if h.get("role") == "user" else "AI"
            date = (h.get("timestamp") or "")[:10]
            tag = "历史对话" + (" " + date if date else "")
            lines.append("[%s %s] %s" % (tag, label, content))
    return "\n".join(lines) if lines else "未找到相关记忆"


# file_read / file_write 的路径白名单：只允许访问以下目录，其余一律拒绝。
# 用 os.path.commonpath 校验，防止相对路径（如 ..\..）跳出白名单。
def _allowed_dirs():
    """返回允许访问的目录白名单：项目根目录 + Obsidian Vault（环境变量 OBSIDIAN_VAULT_PATH 或 config）。"""
    dirs = [PROJECT_ROOT]
    vault = (os.environ.get("OBSIDIAN_VAULT_PATH") or "").strip() or \
            (_load_config().get("obsidian_vault_path") or "").strip()
    if vault:
        dirs.append(vault)
    return [os.path.normcase(os.path.abspath(d)) for d in dirs if d]


def _resolve_within_whitelist(path):
    """把用户提供的路径解析为绝对路径并校验在白名单内。返回 (abs_path, None) 或 (None, 错误提示)。"""
    if os.path.isabs(path):
        abs_path = os.path.abspath(path)
    else:
        abs_path = os.path.abspath(os.path.join(PROJECT_ROOT, path))
    norm = os.path.normcase(os.path.normpath(abs_path))
    allowed = _allowed_dirs()
    for d in allowed:
        try:
            if os.path.commonpath([d, norm]) == d:
                return abs_path, None
        except ValueError:
            continue  # 盘符不同，不在该白名单目录下
    allowed_str = "、".join(allowed)
    return None, "拒绝访问：路径 %s 不在允许范围内。允许访问的目录：%s。" % (path, allowed_str)


# file_read 模糊搜索时要跳过的目录（隐藏/系统/缓存目录，避免误匹配到无关文件）
_SEARCH_SKIP_DIRS = {
    ".obsidian", ".claudian", ".claude", ".git", ".pytest_cache",
    "__pycache__", "chroma_db", "node_modules", ".venv", "venv",
}


def _read_file_text(abs_path, display_path):
    """读取单个文本文件（utf-8、坏字节用 � 替换），超过上限截断；失败返回错误提示字符串。"""
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except PermissionError:
        return "权限不足，无法读取：%s。请检查文件读取权限。" % display_path
    except OSError as e:
        return "读取失败：%s。请检查文件是否损坏或编码是否正常。" % e
    if len(content) > FILE_READ_MAX_CHARS:
        content = content[:FILE_READ_MAX_CHARS] + "\n\n...(文件过长，已截断前 %d 字符)" % FILE_READ_MAX_CHARS
    return content


def _fuzzy_find_files(keyword, max_results=20):
    """在允许目录（项目根 + Obsidian Vault）内递归搜索文件名与 keyword 匹配的文件。

    匹配规则：取 keyword 的文件名部分（去掉目录与扩展名），与目标文件名（去掉扩展名）
    做不区分大小写的「互相包含」判断——既支持只传「简历」命中「简历及面试锻炼.md」，
    也支持传完整文件名或带多余描述文字。返回 [(绝对路径, 原始文件名), ...]，去重排序。
    """
    kw = os.path.basename(keyword.replace("\\", "/")).strip()
    kw = os.path.splitext(kw)[0].casefold()
    if not kw:
        return []
    found = {}
    for base in _allowed_dirs():
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in _SEARCH_SKIP_DIRS]
            for fn in files:
                stem = os.path.splitext(fn)[0].casefold()
                if not stem:
                    continue
                if kw in stem or stem in kw:
                    full = os.path.join(root, fn)
                    found.setdefault(os.path.normcase(full), (full, fn))
    return sorted(found.values(), key=lambda x: x[0].lower())[:max_results]


def _tool_file_read(args):
    """file_read 工具：读取本地文件（白名单限制：仅项目目录 + Obsidian Vault），带长度截断。

    优先按给定路径精确解析（绝对路径 / 相对项目根目录）；解析不到时，在允许目录内按文件名
    模糊搜索，支持只传文件名或关键词（如「简历」命中「简历及面试锻炼.md」）。
    """
    path = (args.get("file_path") or "").strip()
    if not path:
        return "未提供文件路径。请提供 file_path 参数（文件名或相对/绝对路径）后重新调用。"

    # 1) 精确解析（仅当 path 含路径成分时）：绝对路径或相对路径，越界直接拒绝
    is_bare = (not os.path.isabs(path)) and ("/" not in path and "\\" not in path)
    if not is_bare:
        abs_path, err = _resolve_within_whitelist(path)
        if err:
            return err
        if os.path.isfile(abs_path):
            return _read_file_text(abs_path, path)
        if os.path.isdir(abs_path):
            return "目标是目录而非文件：%s。请改用该目录内的具体文件名。" % path
        # 路径在允许范围内但文件不存在 → 落到下面按文件名模糊搜索兜底

    # 2) 模糊搜索：纯文件名/关键词，或精确路径未命中时兜底
    matches = _fuzzy_find_files(path)
    if not matches:
        return ("未找到该文件：%s。已在项目目录与 Obsidian 笔记库中按文件名搜索均未命中，"
                "请确认文件名是否正确。" % path)
    if len(matches) == 1:
        full, fn = matches[0]
        return "已自动匹配文件：%s\n\n%s" % (full, _read_file_text(full, fn))
    # 多个匹配：列出所有候选路径，让用户/模型明确指定后重读
    lines = ["找到 %d 个匹配文件，请指定要读取的具体文件（重新调用 file_read 并传入完整路径）：" % len(matches)]
    for i, (full, _fn) in enumerate(matches, 1):
        lines.append("%d. %s" % (i, full))
    return "\n".join(lines)


def _tool_file_write(args):
    """file_write 工具：创建/追加写入文件（白名单限制，与 file_read 一致）。"""
    path = (args.get("file_path") or "").strip()
    if not path:
        return "未提供文件路径。请提供 file_path 参数后重新调用。"
    content = args.get("content")
    if content is None or content == "":
        return "未提供要写入的内容。请提供 content 参数后重新调用。"
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    mode = (args.get("mode") or "write").strip().lower()
    if mode not in ("write", "append"):
        return "不支持的写入模式：%s。mode 仅支持 write（覆盖）或 append（追加）。" % mode
    abs_path, err = _resolve_within_whitelist(path)
    if err:
        return err
    try:
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(abs_path, "a" if mode == "append" else "w", encoding="utf-8") as f:
            f.write(content)
    except PermissionError:
        return "权限不足，无法写入：%s。请检查目录写入权限。" % path
    except OSError as e:
        return "写入失败：%s。请检查路径与权限。" % e
    action = "追加到" if mode == "append" else "写入"
    return "已成功%s文件：%s（共 %d 字符）" % (action, path, len(content))


# web_search 结果缓存：以关键词哈希为 key，10 分钟内重复请求直接复用，避免重复请求被风控/限流。
_search_cache = {}
_search_cache_lock = threading.Lock()
SEARCH_CACHE_TTL = 600   # 缓存有效期（秒）= 10 分钟

# Bing 网页抓取：国内可访问（无需 API key）。cn.bing.com 优先，bing.com 兜底。
_BING_DOMAINS = ("https://cn.bing.com", "https://www.bing.com")
_BING_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def _strip_tags(text):
    """去掉 HTML 标签。"""
    return re.sub(r"<[^>]+>", "", text or "")


def _parse_bing_html(html, max_results):
    """从 Bing 搜索结果页 HTML 解析出 (标题, 摘要, 链接) 列表。"""
    results = []
    for b in re.findall(r'<li class="b_algo".*?</li>', html, re.S):
        link = title = snip = ""
        h2 = re.search(r'<h2.*?</h2>', b, re.S)
        if h2:
            a = re.search(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', h2.group(0), re.S)
            if a:
                link = unescape(a.group(1)).strip()
                title = unescape(_strip_tags(a.group(2))).strip()
        # 摘要：优先 <p>，其次 b_caption 内的 <p>
        p = re.search(r'<p[^>]*>(.*?)</p>', b, re.S) or \
            re.search(r'<div class="b_caption".*?<p[^>]*>(.*?)</p>', b, re.S)
        if p:
            snip = unescape(_strip_tags(p.group(1))).strip()
        if not title and not snip:
            continue
        results.append((title, snip, link))
        if len(results) >= max_results:
            break
    return results


def _search_bing(query, max_results):
    """用 Bing 网页抓取搜索，多域名兜底。返回 (results, error)。"""
    last_err = ""
    got_valid_page = False
    for domain in _BING_DOMAINS:
        url = "%s/search?q=%s&count=10&setlang=zh-cn" % (domain, quote(query))
        try:
            resp = requests.get(url, headers=_BING_HEADERS, timeout=10)
        except requests.exceptions.RequestException as e:
            last_err = str(e) or type(e).__name__
            continue
        if resp.status_code != 200:
            last_err = "HTTP %d" % resp.status_code
            continue
        got_valid_page = True
        results = _parse_bing_html(resp.text, max_results)
        if results:
            return results, ""
    if got_valid_page:
        return [], ""          # 页面正常但没解析出结果 → 未找到
    return [], last_err        # 全部请求失败 → 报错


def _search_duckduckgo(query, max_results):
    """用 DuckDuckGo 搜索（duckduckgo_search / ddgs 库）。

    国内直连 DuckDuckGo 常被墙或超时，故仅作为首选尝试；失败/无结果时由 _search_bing 兜底。
    返回 (results, error)，results 为 [(标题, 摘要, 链接), ...]。
    """
    try:
        try:
            from ddgs import DDGS                  # 维护中的新包名（duckduckgo-search 已改名）
        except ImportError:
            from duckduckgo_search import DDGS    # 旧包名（pip install duckduckgo-search）
    except ImportError:
        return [], "duckduckgo-search 未安装（请 pip install ddgs）"

    try:
        # 单后端 + 短超时：国内直连 DuckDuckGo 被墙时约 3 秒即可快速失败，交由 Bing 兜底，
        # 避免默认 auto 后端扇出 google/mojeek/startpage 等多个引擎、拖慢到十几秒。
        raw = DDGS(timeout=3).text(query, max_results=max_results, backend="duckduckgo")
    except Exception as e:
        return [], (str(e) or type(e).__name__)

    results = []
    for item in raw or []:
        title = (item.get("title") or "").strip()
        link = (item.get("href") or item.get("url") or "").strip()
        snip = (item.get("body") or item.get("description") or "").strip()
        if not snip:            # 摘要正文为空则跳过（模型靠摘要获取信息，空摘要无用）
            continue
        if not title:
            title = "（无标题）"
        results.append((title, snip, link))
        if len(results) >= max_results:
            break
    return results, ""


# ---- 天气实时数据（wttr.in，国内可直连、原生支持中文城市名）----
# 搜索引擎返回的「摘要」只是网页 SEO 元信息，不含实际气温/降水，模型无法据此回答“明天会下雨吗”。
# 因此天气类查询不再依赖搜索摘要，改为直接拉取真实天气数据。
_WTTR_CODE_CN = {
    113: "晴", 116: "局部多云", 119: "多云", 122: "阴",
    143: "雾", 200: "雷阵雨", 248: "雾", 260: "冻雾",
    176: "局地阵雨", 179: "局地雪", 182: "局地雨夹雪",
    227: "吹雪", 230: "暴风雪", 266: "毛毛雨", 281: "冻毛毛雨",
    284: "冻雨", 293: "小雨", 296: "小雨", 299: "中雨", 302: "中雨",
    305: "大雨", 308: "大雨", 311: "冻雨", 314: "冻雨",
    320: "阵雨", 350: "冰雹", 353: "阵雨", 356: "中雨", 359: "大雨",
    362: "雨夹雪", 365: "雨夹雪", 368: "雨夹雪", 371: "雪", 374: "雪", 377: "雪",
    386: "雷阵雨", 389: "雷阵雨", 392: "雷阵雨", 395: "雷暴",
}
# weatherCode 未命中时的英文描述 → 中文回退（按“更具体者在前”排列，避免被短词提前命中）
_WTTR_DESC_EN_CN = (
    ("smoky", "霾"), ("haze", "霾"), ("fog", "雾"), ("mist", "薄雾"),
    ("freezing", "冻"), ("sleet", "雨夹雪"), ("blizzard", "暴风雪"),
    ("light drizzle", "毛毛雨"), ("drizzle", "毛毛雨"),
    ("light rain shower", "阵雨"), ("rain shower", "阵雨"),
    ("light rain", "小雨"), ("moderate rain", "中雨"), ("heavy rain", "大雨"),
    ("patchy rain", "局地阵雨"), ("rain", "雨"),
    ("thundery", "雷阵雨"), ("thunder", "雷暴"),
    ("light snow", "小雪"), ("heavy snow", "大雪"), ("snow", "雪"),
    ("sunny", "晴"), ("clear", "晴"),
    ("partly cloudy", "局部多云"), ("cloudy", "多云"), ("overcast", "阴"),
)
# 天气类查询关键词（命中即走天气数据源）
_WEATHER_KEYWORDS = (
    "天气", "下雨", "降雨", "降水", "气温", "温度", "晴", "阴", "多云", "台风",
    "刮风", "风力", "湿度", "预报", "多少度", "几度", "会不会下雨", "降温", "升温",
)
# 提取城市名时要剔除的停用词（疑问词/时间词/天气词本身等）
_WEATHER_STOPWORDS = (
    "天气", "天气预报", "预报", "今天", "明天", "后天", "大后天", "今日", "明日",
    "本周", "这周", "下周", "未来", "最近", "目前", "现在", "怎么样", "如何",
    "吗", "呢", "啊", "的", "是", "会", "有没有", "会不会", "星期", "周末",
    "气温", "温度", "下雨", "降雨", "降水", "多少度", "几度", "降温", "升温",
)


def _wttr_desc(item):
    """取 wttr.in 的天气描述：weatherCode 映射中文 > 英文描述关键词映射中文 > 原始英文。"""
    try:
        code = int(item.get("weatherCode"))
    except (TypeError, ValueError):
        code = None
    if code is not None and code in _WTTR_CODE_CN:
        return _WTTR_CODE_CN[code]
    wd = item.get("weatherDesc")
    text = (wd[0].get("value") or "").strip() if isinstance(wd, list) and wd else ""
    low = text.lower()
    for en, cn in _WTTR_DESC_EN_CN:
        if en in low:
            return cn
    return text


def _extract_city(query):
    """从天气类 query 里尽量提取城市名（去停用词/数字标点/行政后缀）。"""
    q = query
    for w in _WEATHER_STOPWORDS:
        q = q.replace(w, "")
    q = re.sub(r"[0-9０-９\s，。？！、；：·℃°%\-—]", "", q)
    q = q.rstrip("省市县区")
    return q.strip()


def _fetch_weather(city, days=3):
    """用 wttr.in 拉取真实天气（JSON），返回给模型的可读文本；失败返回 None。"""
    if not city:
        return None
    try:
        resp = requests.get("https://wttr.in/%s?format=j1" % quote(city),
                            headers={"User-Agent": "curl/8"}, timeout=10)
    except requests.exceptions.RequestException:
        return None
    if resp.status_code != 200 or "Unknown location" in resp.text or "Sorry" in resp.text:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    try:
        parts = []
        cur = data["current_condition"][0]
        wind = "%s风 %s km/h" % (cur.get("winddir16Point", ""), cur.get("windspeedKmph", "?"))
        parts.append("当前：%s，气温 %s°C，体感 %s°C，湿度 %s%%，%s" % (
            _wttr_desc(cur), cur.get("temp_C", "?"), cur.get("FeelsLikeC", "?"),
            cur.get("humidity", "?"), wind))
        for day in data["weather"][:days]:
            max_rain = 0
            desc = ""
            for h in day.get("hourly", []):
                try:
                    rp = int(h.get("chanceofrain") or 0)
                except (TypeError, ValueError):
                    rp = 0
                if rp >= max_rain:
                    max_rain = rp
                    desc = _wttr_desc(h)
            parts.append("%s：%s，最高 %s°C / 最低 %s°C，降水概率 %d%%" % (
                day.get("date"), desc, day.get("maxtempC"), day.get("mintempC"), max_rain))
        return "「%s」实时天气（数据源 wttr.in）：\n%s" % (city, "\n".join(parts))
    except (KeyError, IndexError, TypeError):
        return None


def _tool_web_search(args, max_results=5):
    """web_search 工具：天气类查询直接返回 wttr.in 真实数据；否则 DuckDuckGo 搜索、失败回退 Bing。

    结果按关键词缓存 10 分钟；返回格式：标题 / 摘要 / 链接（摘要正文保证非空）。
    """
    query = (args.get("query") or "").strip()
    if not query:
        return "未提供搜索关键词。请提供 query 参数后重新调用 web_search。"

    key = hashlib.md5(query.encode("utf-8")).hexdigest()
    now = time.time()
    with _search_cache_lock:
        cached = _search_cache.get(key)
        if cached and now - cached["ts"] < SEARCH_CACHE_TTL:
            return cached["result"]

    # 天气类查询：不依赖搜索摘要，直接拉真实天气数据（气温/降水概率），模型可直接据此回答
    if any(w in query for w in _WEATHER_KEYWORDS):
        city = _extract_city(query)
        weather_text = _fetch_weather(city)
        if weather_text:
            with _search_cache_lock:
                _search_cache[key] = {"ts": now, "result": weather_text}
            return weather_text

    # 普通搜索：首选 DuckDuckGo（duckduckgo_search 库）；国内直连失败或无结果时回退 Bing
    results, error = _search_duckduckgo(query, max_results)
    if not results:
        results, error = _search_bing(query, max_results)

    # 归一化：过滤空结果与空摘要，保证每条结果标题/摘要/链接三者齐全
    clean = []
    for t, s, l in results or []:
        t = (t or "").strip()
        s = (s or "").strip()
        l = (l or "").strip()
        if not s:               # 摘要正文为空则跳过
            continue
        if not t:
            t = "（无标题）"
        clean.append((t, s, l))
    results = clean

    if error and not results:
        result_str = "搜索请求失败：%s。请稍后重试，或更换更简洁的关键词。" % error
    elif not results:
        result_str = "未找到相关搜索结果。请更换更简洁或更具体的关键词后重试。"
    else:
        lines = ["标题：%s\n摘要：%s\n链接：%s" % (t, s, l) for t, s, l in results]
        result_str = "\n\n".join(lines)

    # 只缓存成功（有结果且无错误）的结果，错误/空结果不缓存以便稍后重试
    if not error and results:
        with _search_cache_lock:
            _search_cache[key] = {"ts": now, "result": result_str}
    return result_str


def _tool_get_current_time(args):
    """get_current_time 工具：返回当前日期时间（无参数）。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 工具注册表：工具名 -> {"handler": 执行函数, "schema": JSON Schema}。
# 单一数据源：chat() 里的 tools 列表由此自动生成，新增工具只需在此注册一处。
TOOL_REGISTRY = {
    "get_current_time": {
        "handler": _tool_get_current_time,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "获取当前的日期和时间。当用户询问时间相关问题时使用。",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
    },
    "memory_search": {
        "handler": _tool_memory_search,
        "schema": {
            "type": "function",
            "function": {
                "name": "memory_search",
                "description": "从用户的 Obsidian 笔记库中搜索相关记忆和知识。当用户的问题可能涉及其个人笔记、过往记录或已有知识时使用此工具。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词，描述你想查找的内容"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
    },
    "file_read": {
        "handler": _tool_file_read,
        "schema": {
            "type": "function",
            "function": {
                "name": "file_read",
                "description": "读取本地文件的内容。用于查看代码、配置文件、笔记等。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "要读取的文件路径（相对路径或绝对路径）"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        },
    },
    "file_write": {
        "handler": _tool_file_write,
        "schema": {
            "type": "function",
            "function": {
                "name": "file_write",
                "description": "创建或追加写入本地文件（仅限白名单目录）。写入前需用户确认。用于帮用户创建、修改笔记或保存内容。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "要写入的文件路径（相对或绝对路径，仅限白名单目录）"
                        },
                        "content": {
                            "type": "string",
                            "description": "要写入的文件内容"
                        },
                        "mode": {
                            "type": "string",
                            "description": "写入模式：write=覆盖，append=追加，默认 write"
                        }
                    },
                    "required": ["file_path", "content"]
                }
            }
        },
    },
    "web_search": {
        "handler": _tool_web_search,
        "schema": {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "在互联网上搜索实时信息。当需要查询最新资讯、事实核查、或模型知识截止日期之后的信息时使用此工具。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词。请使用简洁关键词（如“温州天气”），不要传完整问句。"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
    },
}


def _execute_tool(tool_name, args=None):
    """执行工具，返回结果字符串。分发由 TOOL_REGISTRY 字典映射完成，无需 if-elif。"""
    args = _parse_args(args)
    entry = TOOL_REGISTRY.get(tool_name)
    if entry is None:
        available = "、".join(TOOL_REGISTRY.keys())
        return "未知工具：%s。当前可用工具：%s，请选择正确的工具重试。" % (tool_name or "（空）", available)
    try:
        return entry["handler"](args)
    except Exception as e:
        return "工具 %s 执行出错：%s，请分析原因后调整参数重试。" % (tool_name, str(e) or type(e).__name__)


def _is_tool_error(result):
    """判断工具返回是否为「执行失败」（参数错误/异常），而非正常的空结果（如「未找到」）。"""
    if not isinstance(result, str) or not result:
        return False
    if result.startswith("工具 ") and "执行出错" in result:
        return True
    return result.startswith((
        "未提供", "未知工具", "拒绝访问", "搜索请求失败", "搜索依赖缺失",
        "记忆检索不可用", "权限不足", "读取失败", "写入失败", "不支持的",
    ))


# ---------- file_write 用户确认机制 ----------
# file_write 是唯一有副作用的写操作：模型发起写入时，先通过 SSE 把内容推给前端，
# 前端弹框让用户点「允许/拒绝」，再通过 /api/confirm_write 回传决定，此处阻塞等待。
_pending_writes = {}          # confirm_id -> {"event": threading.Event, "allow": bool}
_pending_writes_lock = threading.Lock()
_write_confirm_seq = [0]


def _new_confirm_id():
    with _pending_writes_lock:
        _write_confirm_seq[0] += 1
        return "w%d" % _write_confirm_seq[0]


def _await_write_decision(confirm_id, timeout=180):
    """阻塞等待前端对 file_write 的确认；返回 {"allow": bool} 或 None（超时视为拒绝）。"""
    with _pending_writes_lock:
        entry = {"event": threading.Event(), "allow": False}
        _pending_writes[confirm_id] = entry
    decided = entry["event"].wait(timeout)
    with _pending_writes_lock:
        _pending_writes.pop(confirm_id, None)
    if not decided:
        return None
    return {"allow": entry["allow"]}


# ---------- Obsidian 导出相关 (保持不变) ----------
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def _load_config():
    if not os.path.exists(CONFIG_PATH): return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (ValueError, OSError) as e:
        logger.warning("读取 config.json 失败：%s", e)
        return {}

def _save_config(data):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _sanitize_filename(name):
    return re.sub(r'[\\/:*?"<>|]', "_", name or "")

def _yaml_quote(s):
    s = (s or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return '"' + s + '"'

def _obsidian_dir():
    vault = (_load_config().get("obsidian_vault_path") or "").strip()
    return os.path.join(vault, "MemFlow") if vault else None

def _generate_markdown(conv):
    title = conv.get("title") or db.DEFAULT_TITLE
    messages = conv.get("messages") or []
    lines = [
        "---",
        "title: " + _yaml_quote(title),
        "created: " + _yaml_quote(conv.get("created_at") or ""),
        "updated: " + _yaml_quote(conv.get("updated_at") or ""),
        "message_count: %d" % len(messages),
        "tags: [memflow, ai-chat]",
        "---",
        "",
        "# " + title,
        "",
    ]
    for m in messages:
        role = m.get("role") or "user"
        if role == "tool" or (role == "assistant" and m.get("tool_calls")):
            continue  # 工具调用轨迹不导出，保持导出文件干净
        label = "User" if role == "user" else "Assistant"
        lines.append("## %s (%s)" % (label, m.get("timestamp") or ""))
        lines.append(m.get("content") or "")
        lines.append("")
    return "\n".join(lines)

# ---------- 路由 ----------
@app.route("/")
def index():
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

@app.route("/api/models", methods=["GET"])
def api_list_models():
    installed_models = get_ollama_models()
    installed_names = {m.get("name") for m in installed_models}
    if installed_names:
        models = [m for m in ALLOWED_MODELS if m in installed_names]
    else:
        models = list(ALLOWED_MODELS)
    return jsonify({"models": models})

@app.route("/api/models/cleanup", methods=["POST"])
def api_cleanup_models():
    try:
        resp = requests.get(OLLAMA_BASE + "/api/tags", timeout=5)
        resp.raise_for_status()
        installed = [m.get("name") for m in (resp.json().get("models") or [])]
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"无法获取模型列表：{e}"}), 502
    
    to_delete = [m for m in installed if m not in ALLOWED_MODELS]
    deleted, failed = [], []
    for name in to_delete:
        try:
            d = requests.delete(OLLAMA_BASE + "/api/delete", json={"name": name}, timeout=600)
            if d.status_code == 200:
                deleted.append(name)
            else:
                failed.append({"model": name, "detail": d.text[:300]})
        except requests.exceptions.RequestException as e:
            failed.append({"model": name, "detail": str(e)})
    return jsonify({"to_delete": to_delete, "deleted": deleted, "failed": failed})

@app.route("/api/health", methods=["GET"])
def api_health():
    try:
        with db.get_conn() as conn:
            conn.execute("SELECT 1").fetchone()
            db_status = "ok"
            db_message = "SQLite connected"
    except Exception as e:
        db_status = "error"
        db_message = str(e)

    models_count = 0
    try:
        resp = requests.get(OLLAMA_BASE + "/api/tags", timeout=5)
        if resp.status_code == 200:
            ollama_status = "ok"
            ollama_message = "Ollama running"
            models_count = len(resp.json().get("models") or [])
        else:
            ollama_status = "error"
            ollama_message = "Ollama /api/tags 返回 HTTP %d" % resp.status_code
    except requests.exceptions.RequestException as e:
        ollama_status = "error"
        ollama_message = str(e)

    obsidian_files = obsidian_chunks = 0
    if rag.is_available():
        rag_status = "ok"
        rag_message = "ChromaDB available"
        stats = rag.get_obsidian_stats()
        obsidian_files = stats.get("files", 0)
        obsidian_chunks = stats.get("chunks", 0)
    else:
        rag_status = "error"
        rag_message = "ChromaDB 不可用（依赖缺失或初始化失败）"

    if db_status == "error":
        overall = "error"
    elif ollama_status == "error" or rag_status == "error":
        overall = "degraded"
    else:
        overall = "ok"

    return jsonify({
        "status": overall,
        "timestamp": _now_str(),
        "database": {"status": db_status, "message": db_message},
        "ollama": {"status": ollama_status, "message": ollama_message, "models_count": models_count},
        "rag": {
            "status": rag_status,
            "message": rag_message,
            "obsidian_files": obsidian_files,
            "obsidian_chunks": obsidian_chunks,
        },
    })

@app.route("/api/conversations", methods=["GET"])
def api_list_conversations():
    return jsonify(db.list_conversations())

@app.route("/api/conversations", methods=["POST"])
def api_create_conversation():
    conv_id = db.create_conversation()
    return jsonify({"id": conv_id}), 201

@app.route("/api/conversations/<int:conv_id>", methods=["GET"])
def api_get_conversation(conv_id):
    conv = db.get_conversation(conv_id)
    if conv is None:
        return jsonify({"error": "对话不存在"}), 404
    return jsonify(conv)

@app.route("/api/conversations/<int:conv_id>", methods=["DELETE"])
def api_delete_conversation(conv_id):
    db.delete_conversation(conv_id)
    return jsonify({"status": "success"})

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_input = (data.get("message") or "").strip()
    if not user_input:
        return jsonify({"error": "消息不能为空"}), 400

    # MODEL_NAME 环境变量优先级最高：设置后强制使用该模型（如 MODEL_NAME=qwen3.5:9b 切到 9b，解决 7b 漏调工具）
    model = (os.environ.get("MODEL_NAME") or data.get("model") or DEFAULT_MODEL).strip()

    # 定位对话
    conv_id = data.get("conversation_id")
    conv = db.get_conversation(conv_id) if conv_id else None
    if conv is None:
        conv_id = db.create_conversation()
        conv = db.get_conversation(conv_id)

    should_rename = conv["title"] == db.DEFAULT_TITLE

    # 组装基础上下文（ReAct 过程中会在此列表上追加 tool_calls / tool 消息）
    messages = build_prompt_with_summary(conv_id, user_input, model)

    # 注入规划与反思系统指令（作为 messages 第一条，在用户消息之前）
    messages.insert(0, {"role": "system", "content": REACT_SYSTEM_PROMPT})

    # --- 定义工具（由 TOOL_REGISTRY 自动生成，单一数据源）---
    tools = [entry["schema"] for entry in TOOL_REGISTRY.values()]

    def generate():
        used_tools = []
        tool_trace = []   # 本轮的工具调用轨迹（assistant tool_calls + tool 结果），落库供多轮引用与调试
        consecutive_failures = 0   # 连续工具调用失败计数，用于熔断

        try:
            # ReAct 循环：最多 MAX_REACT_ROUNDS 轮
            for _ in range(MAX_REACT_ROUNDS):
                # a. 流式请求：逐 chunk 接收并实时推给前端（消除首字等待），
                #    同时累积正文，等完整响应拼完后判断本轮是调用工具还是最终回答
                resp = requests.post(
                    OLLAMA_URL,
                    json={"model": model, "messages": messages, "tools": tools,
                          "stream": True, "options": {"num_ctx": OLLAMA_NUM_CTX}},
                    timeout=300,
                    stream=True,
                )
                if resp.status_code != 200:
                    yield sse({"type": "error", "code": "ollama_error", "detail": resp.text[:500]})
                    return

                content_parts = []
                tool_calls = None
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except ValueError:
                        continue
                    message = chunk.get("message") or {}
                    content = message.get("content") or ""
                    if content:
                        content_parts.append(content)
                        # 真流式：正文 chunk 一到达就推给前端
                        yield sse({"type": "token", "content": content})
                    if message.get("tool_calls"):
                        tool_calls = message["tool_calls"]
                    if chunk.get("done"):
                        break

                # b. 有 tool_calls → 执行工具并回填结果，继续下一轮（不向前端输出正文）
                #    Ollama 工具调用时 content 为空，上面的循环不会向前端输出任何正文。
                if tool_calls:
                    # 先把模型这轮的“调用意图”作为 assistant 消息加入上下文（并记录到落库轨迹）
                    assistant_tool_msg = {
                        "role": "assistant",
                        "content": "".join(content_parts),
                        "tool_calls": tool_calls,
                    }
                    messages.append(assistant_tool_msg)
                    tool_trace.append(assistant_tool_msg)
                    for tc in tool_calls:
                        fn = tc.get("function") or {}
                        name = fn.get("name") or ""
                        args = fn.get("arguments") or {}
                        parsed_args = _parse_args(args)
                        # 通知前端：正在调用工具（附参数，方便调试）
                        yield sse({"type": "tool_call", "name": name, "args": parsed_args})
                        used_tools.append(name)

                        # file_write 需要用户确认：推送待写入内容并阻塞等待决定
                        if name == "file_write":
                            confirm_id = _new_confirm_id()
                            yield sse({"type": "confirm_write", "id": confirm_id,
                                       "path": parsed_args.get("file_path") or "",
                                       "content": parsed_args.get("content") or "",
                                       "mode": parsed_args.get("mode") or "write"})
                            decision = _await_write_decision(confirm_id)
                            if not decision or not decision.get("allow"):
                                tool_result = "用户已取消写入"
                            else:
                                tool_result = _tool_file_write(parsed_args)
                        else:
                            tool_result = _execute_tool(name, args)

                        # 通知前端：工具执行结果（方便调试）
                        yield sse({"type": "tool_result", "name": name, "result": tool_result})

                        # 熔断：连续两次工具调用失败（参数错误/异常）则终止
                        if _is_tool_error(tool_result):
                            consecutive_failures += 1
                        else:
                            consecutive_failures = 0

                        tool_msg = {
                            "role": "tool",
                            "content": tool_result,
                            "tool_name": name,
                        }
                        messages.append(tool_msg)
                        tool_trace.append(tool_msg)

                        if consecutive_failures >= MAX_CONSECUTIVE_TOOL_FAILURES:
                            yield sse({"type": "error", "code": "tool_failures",
                                       "detail": "连续 %d 次工具调用失败（参数错误或执行异常），已终止本次任务。" % consecutive_failures})
                            return
                    continue

                # c. 没有 tool_calls → 正文已在上面逐 chunk 实时推给前端，这里用累积结果收尾
                full_reply = "".join(content_parts)

                # 收尾：标题 + 落库 + 向量化 + done 事件
                title = make_title(full_reply) if should_rename else None
                # 落库时连同工具调用轨迹一起保存，使多轮对话能引用之前的工具结果
                persist = [{"role": "user", "content": user_input}]
                persist.extend(tool_trace)
                persist.append({"role": "assistant", "content": full_reply})
                new_messages = db.append_messages(conv_id, persist, new_title=title)
                # 向量化只入 user/assistant 正文，跳过工具调用/结果（避免污染长期记忆）
                _index_messages_async([
                    m for m in new_messages
                    if m["role"] in ("user", "assistant") and not m.get("tool_calls")
                ])
                updated = db.get_conversation(conv_id)
                done_payload = {
                    "type": "done",
                    "conversation_id": conv_id,
                    "title": updated["title"] if updated else db.DEFAULT_TITLE,
                }
                if used_tools:
                    done_payload["tool_used"] = ", ".join(used_tools)
                yield sse(done_payload)
                return

            # 达到最大轮次仍未得到最终回答 → 强制退出
            yield sse({"type": "error", "code": "max_rounds", "detail": "已达到最大推理轮次"})

        except requests.exceptions.ConnectionError:
            logger.error("Ollama 连接失败：请检查 Ollama 是否运行 (http://127.0.0.1:11434)")
            yield sse({"type": "error", "code": "ollama_down", "detail": "Ollama 服务未启动，请检查后台是否运行"})
        except requests.exceptions.Timeout:
            yield sse({"type": "error", "code": "unknown", "detail": "请求超时"})
        except Exception as e:
            msg = str(e) or type(e).__name__
            low = msg.lower()
            if "not found" in low:
                code = "model_not_found"
            elif "connection" in low or "refused" in low or "connecterror" in low:
                code = "ollama_down"
            else:
                code = "unknown"
            yield sse({"type": "error", "code": code, "detail": msg})

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })

@app.route("/api/confirm_write", methods=["POST"])
def api_confirm_write():
    """前端回传 file_write 的用户确认决定，唤醒阻塞中的生成器。"""
    data = request.get_json(silent=True) or {}
    confirm_id = (data.get("id") or "").strip()
    allow = bool(data.get("allow"))
    with _pending_writes_lock:
        entry = _pending_writes.get(confirm_id)
    if entry is None:
        return jsonify({"status": "error", "detail": "确认请求不存在或已过期"}), 404
    entry["allow"] = allow
    entry["event"].set()
    return jsonify({"status": "ok"})

@app.route("/api/sync_chroma", methods=["GET"])
def api_sync_chroma():
    """手动触发全量同步：把 messages 表所有消息增量同步到 ChromaDB。"""
    if not rag.is_available():
        return jsonify({"status": "disabled", "detail": "ChromaDB 不可用（依赖缺失或初始化失败）"})
    result = rag.sync_all_messages_to_chroma()
    return jsonify(result)

@app.route("/api/obsidian/config", methods=["GET"])
def api_obsidian_get_config():
    """返回当前配置的 Obsidian Vault 路径。"""
    path = _load_config().get("obsidian_vault_path") or ""
    return jsonify({"obsidian_vault_path": path})

@app.route("/api/obsidian/config", methods=["POST"])
def api_obsidian_set_config():
    """保存 Obsidian Vault 路径，并自动创建 MemFlow 子目录。"""
    data = request.get_json(silent=True) or {}
    vault_path = (data.get("obsidian_vault_path") or "").strip()
    if not vault_path:
        return jsonify({"status": "error", "detail": "路径不能为空"}), 400
    cfg = _load_config()
    cfg["obsidian_vault_path"] = vault_path
    _save_config(cfg)
    memflow_dir = os.path.join(vault_path, "MemFlow")
    try:
        os.makedirs(memflow_dir, exist_ok=True)
    except OSError as e:
        logger.warning("创建 Obsidian 目录失败：%s", e)
        return jsonify({"status": "error", "detail": "无法创建目录：%s" % e}), 500
    # 后台增量导入笔记到向量库，供 memory_search 工具检索（不阻塞响应）
    _sync_obsidian_async(vault_path)
    return jsonify({"status": "ok", "path": memflow_dir})

@app.route("/api/obsidian/export/<int:conv_id>", methods=["GET"])
def api_obsidian_export(conv_id):
    """导出单个对话为 Markdown 文件（UTF-8 with BOM）。"""
    conv = db.get_conversation(conv_id)
    if conv is None:
        return jsonify({"status": "error", "detail": "对话不存在"}), 404
    obs_dir = _obsidian_dir()
    if obs_dir is None:
        return jsonify({"status": "error", "detail": "尚未配置 Obsidian Vault 路径，请先设置"}), 400
    try:
        os.makedirs(obs_dir, exist_ok=True)
        title = conv.get("title") or db.DEFAULT_TITLE
        filename = "%s_%d.md" % (_sanitize_filename(title), conv_id)
        file_path = os.path.join(obs_dir, filename)
        with open(file_path, "w", encoding="utf-8-sig") as f:
            f.write(_generate_markdown(conv))
    except OSError as e:
        logger.warning("导出对话 %s 失败：%s", conv_id, e)
        return jsonify({"status": "error", "detail": "写入失败：%s" % e}), 500
    return jsonify({
        "status": "ok",
        "file_path": file_path,
        "message_count": len(conv.get("messages") or []),
    })

@app.route("/api/obsidian/export-all", methods=["GET"])
def api_obsidian_export_all():
    """导出全部对话，每个对话生成一个 .md 文件。"""
    obs_dir = _obsidian_dir()
    if obs_dir is None:
        return jsonify({"status": "error", "detail": "尚未配置 Obsidian Vault 路径，请先设置"}), 400
    try:
        os.makedirs(obs_dir, exist_ok=True)
    except OSError as e:
        return jsonify({"status": "error", "detail": "无法创建目录：%s" % e}), 500
    exported = []
    for c in db.list_conversations():
        conv = db.get_conversation(c["id"])
        if conv is None: continue
        title = conv.get("title") or db.DEFAULT_TITLE
        filename = "%s_%d.md" % (_sanitize_filename(title), c["id"])
        file_path = os.path.join(obs_dir, filename)
        try:
            with open(file_path, "w", encoding="utf-8-sig") as f:
                f.write(_generate_markdown(conv))
            exported.append(file_path)
        except OSError as e:
            logger.warning("导出对话 %s 失败：%s", c["id"], e)
    return jsonify({"status": "ok", "exported": len(exported), "files": exported})


@app.route("/api/obsidian/sync", methods=["GET"])
def api_obsidian_sync():
    """把 Obsidian Vault 笔记增量导入向量库，供 memory_search 工具检索笔记。"""
    if not rag.is_available():
        return jsonify({"status": "disabled", "detail": "ChromaDB 不可用（依赖缺失或初始化失败）"})
    vault_path = _load_config().get("obsidian_vault_path") or ""
    if not vault_path:
        return jsonify({"status": "error", "detail": "尚未配置 Obsidian Vault 路径，请先设置"}), 400
    return jsonify(rag.sync_obsidian_vault(vault_path))


# ---------- 后台向量化 ----------
def _index_messages_async(messages):
    """后台线程：将新消息存入 ChromaDB 向量库（RAG 未就绪时懒加载，不阻塞聊天）。"""
    def _task():
        try:
            rag.index_messages(messages)
        except Exception as e:
            logger.warning("向量化入库失败：%s", e)
    t = threading.Thread(target=_task, daemon=True)
    t.start()


def _sync_obsidian_async(vault_path):
    """后台线程：增量同步 Obsidian 笔记到向量库（RAG 未就绪时懒加载）。"""
    def _task():
        try:
            rag.sync_obsidian_vault(vault_path)
        except Exception as e:
            logger.warning("Obsidian 笔记导入失败：%s", e)
    threading.Thread(target=_task, daemon=True).start()


# ---------- 启动 ----------
if __name__ == "__main__":
    db.init_db()
    # 后台加载 embedding 模型 / 初始化 ChromaDB（不阻塞启动，就绪后 RAG 自动生效）
    rag.start_background_init()
    # 安全默认：仅本机可访问 + 关闭 debug，避免向局域网暴露 Werkzeug 交互式调试器。
    # 仅当显式设置 FLASK_DEV=1 时才开放 0.0.0.0 与 debug=True（开发环境）。
    dev_mode = os.environ.get("FLASK_DEV") == "1"
    app.run(
        host="0.0.0.0" if dev_mode else "127.0.0.1",
        port=5000,
        debug=dev_mode,
    )