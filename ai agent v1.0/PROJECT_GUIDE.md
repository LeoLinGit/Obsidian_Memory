# MemFlow 项目目录导读指南

> 最近更新：2026-09-02 · 由小月分析生成

## 1. 整体架构概述

这是一个 **AI Agent 长期记忆体系的实验项目（MemFlow）**：核心是 `app/` 下的一个本地 Web 聊天应用（Flask + SQLite + ChromaDB 向量检索 + 前端原生 HTML/JS/CSS），通过 **本地 Ollama（qwen2.5:7b / qwen3.5:9b）** 提供流式对话；根目录下另有几个早期命令行实验脚本（对接云端 DashScope qwen-max 和本地 Ollama）。

整体架构模式：**前后端分离的单体 Flask 应用 + SSE 流式输出 + RAG 长期记忆 + ReAct 工具调用 + Obsidian 双向联动**。后端已模块化（数据库 / 向量检索 / 上下文组装各自独立成文件），前端拆成「HTML 结构 + 外部 CSS + 外部 JS」三件套。依赖仅 flask / requests / chromadb / sentence-transformers / torch 五个第三方库。

**长期记忆的实现思路**：每条对话消息在落库后，后台线程用 sentence-transformers（多语言模型 `paraphrase-multilingual-MiniLM-L12-v2`，中文可用）做 embedding 存入 ChromaDB；模型通过 `memory_search` 工具按需语义检索相关历史 / Obsidian 笔记，作为「相关记忆」注入 prompt，再拼上滑动窗口最近消息 + 早期摘要，实现跨会话记忆。

**工具调用（ReAct）**：模型在回答前可自主调用工具（注册于 `app.py` 顶部的 `TOOL_REGISTRY`）：`get_current_time` / `memory_search` / `file_read` / `file_write` / `web_search`，最多 5 轮推理；工具结果回填后继续生成最终回答。其中 `file_read` 支持只传文件名/关键词——精确路径未命中时会在项目目录与 Obsidian 笔记库内按文件名模糊搜索（如「简历」可命中「简历及面试锻炼.md」），多命中时列出候选路径供选择。

**Obsidian 双向联动**：对话可导出为 Markdown 到 Vault 的 `MemFlow/` 子目录（导出方向）；反向地，Vault 里的笔记可增量导入向量库供 `memory_search` 检索（导入方向）。

## 2. 核心目录解析

| 目录 / 文件 | 作用 |
|---|---|
| `app/` | **正式的 Web 应用主体（MemFlow）**，是项目当前的核心 |
| `app/templates/` | 只有 `index.html`（聊天界面骨架，无内联 CSS/JS） |
| `app/static/` | 前端资源：`style.css`（深色/浅色主题）+ `script.js`（交互逻辑） |
| `app/chroma_db/` | ChromaDB 向量库持久化目录（运行后自动生成） |
| `app/config.json` | Obsidian Vault 路径等运行时配置（由前端设置弹窗写入） |
| 根目录散落的 `.py` | 早期的命令行实验脚本，验证「记忆文件作为提示词」的思路 |

## 3. 关键文件指路

| 文件 | 说明 |
|---|---|
| `app/app.py` | **主入口 + REST 路由 + SSE 转发 + 工具调用**。模型白名单在顶部 `ALLOWED_MODELS`；`TOOL_REGISTRY` 集中注册 5 个工具；主要路由：`/api/models`（白名单）、`/api/models/cleanup`（清理非白名单模型）、`/api/health`（DB/Ollama/RAG 状态）、`/api/conversations`（会话 CRUD）、`/api/chat`（SSE 流式 + ReAct 工具调用）、`/api/sync_chroma`（消息全量同步）、`/api/obsidian/*`（config / sync / export / export-all）。启动时 `rag.start_background_init()` 后台加载 embedding 模型。启动后访问 `http://127.0.0.1:5000` |
| `app/db.py` | **数据库层**：SQLite 两张规范化表（`conversations` + `messages`，外键级联删除），全部走 `get_conn()` 上下文管理器自动 commit/rollback/close |
| `app/rag.py` | **RAG 向量检索 + Obsidian 导入**：ChromaDB PersistentClient + `paraphrase-multilingual-MiniLM-L12-v2`（中文可用）做 embedding；提供 `index_messages`（聊天落库后批量入库）、`search_similar_messages`（top-k 检索）、`sync_all_messages_to_chroma`、`import_obsidian_vault` / `sync_obsidian_vault`（Obsidian 笔记全量/增量导入）、`get_obsidian_stats`；依赖缺失时优雅降级（聊天不受影响） |
| `app/context_manager.py` | **上下文组装**：System Prompt + 早期摘要 + 滑动窗口 + 当前问题，按 token 预算截断（首尾永不截断，优先砍 RAG 记忆）。主入口 `build_prompt_with_summary`（RAG 默认关闭、改由 `memory_search` 工具按需检索）；`build_prompt` 为遗留实现 |
| `app/migrate.py` | **一次性数据迁移脚本**：旧结构（messages 为 JSON 字符串）→ 新结构（messages 独立表）。幂等，已迁移会自动跳过 |
| `app/start.bat` | **一键启动脚本（4 步）**：启动 Ollama → 检查/安装 RAG 依赖 → 跑数据库迁移 → 启动 Flask + 打开浏览器 |
| `app/templates/index.html` | 前端骨架：侧边栏 + 聊天区 + 设置弹窗 + Obsidian 路径弹窗，CDN 引入 marked / DOMPurify / highlight.js |
| `app/static/script.js` | 前端逻辑：会话管理、SSE 流式渲染（含 tool_call 提示）、Markdown 渲染、深色/浅色主题切换、模型选择、Obsidian 导出/同步 |
| `app/static/style.css` | 前端样式：CSS 变量实现深色（默认）/ 浅色主题，含移动端适配 |
| `app/conversations.db` | SQLite 数据库（`conversations` + `messages` 两张表） |
| `ask_memory.py` | 早期脚本：读 `my.memory.md` 作为系统提示词，调云端 DashScope qwen-max，命令行问答 |
| `chat_with_memory.py` | 早期脚本：命令行多轮对话，走本地 Ollama（`ollama` Python 包），历史存 JSON 到 `E:/AI_Memory` |
| `my.memory.md` | 个人记忆文件（自我描述），被 `ask_memory.py` 用作提示词 |

## 4. 新手阅读建议（找回记忆路线）

1. **双击 `app/start.bat`** —— 一键启动：Ollama(11434) → 依赖 → 迁移 → Flask(5000) + 浏览器。
2. **精读 `app/app.py`** —— 先看顶部 `ALLOWED_MODELS` 与 `TOOL_REGISTRY`，再看路由层（REST + `/api/chat` SSE 的 ReAct 循环）。
3. **再看 `app/db.py` → `app/rag.py` → `app/context_manager.py`** —— 理解「消息落库 → 向量化 → 工具检索注入 prompt」的长期记忆闭环。
4. **浏览前端** —— `index.html`（骨架）→ `script.js`（逻辑）→ `style.css`（样式）。
5. 早期实验脚本（`ask_memory.py` / `chat_with_memory.py`）只是思路验证，**不影响主应用**，可最后看或直接忽略。

### 遗留问题 / 待清理

- `app/templates/index.html.bak_20260829_142515` 一处 `.bak` 备份残留，可清理
- `start.bat` 里 Ollama 路径硬编码为 `E:\Ollama\ollama.exe`（已有 PATH fallback，换机器建议确认）
- `.workbuddy/`（memory + trash）是另一工具「workbuddy」的历史产物（含旧 `.bak` 备份），与当前应用无关，确认后可清理
