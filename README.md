# MemFlow — 本地 AI 长期记忆助手

一个跑在本机上的 AI 记忆聊天应用：Flask + 本地 Ollama 大模型 + SQLite + ChromaDB 向量检索，
支持**流式对话、跨会话长期记忆、ReAct 工具调用、Obsidian 笔记双向联动**。

所有数据（聊天记录、向量库、笔记）都留在你自己的电脑里，不依赖任何云端 API。

## 功能特性

- 💬 **SSE 流式对话**：逐字输出，前端实时渲染 Markdown（含代码高亮、XSS 过滤）
- 🧠 **长期记忆（RAG）**：聊天记录与 Obsidian 笔记经 `sentence-transformers` 向量化存入 ChromaDB，模型通过 `memory_search` 工具按需检索相关记忆
- 🔧 **ReAct 工具调用**：模型可自主调用 5 个工具（最多 5 轮推理，带失败熔断）
  - `get_current_time` — 获取当前时间
  - `memory_search` — 语义检索笔记 / 历史对话
  - `file_read` — 读取本地文件（白名单 + 文件名模糊搜索）
  - `file_write` — 写入文件（需前端弹窗确认）
  - `web_search` — 联网搜索（DDG→Bing 兜底）+ 天气直连 wttr.in
- 📓 **Obsidian 双向联动**：对话可导出为 Markdown 到 Vault；Vault 笔记可增量导入为记忆
- 🌗 深色 / 浅色主题、移动端适配

## 技术栈

| 层 | 技术 |
|---|---|
| 后端 | Python 3.14 + Flask |
| 大模型 | 本地 Ollama（`qwen2.5:7b` / `qwen3.5:9b`） |
| 数据库 | SQLite（`conversations` + `messages`） |
| 向量检索 | ChromaDB + `paraphrase-multilingual-MiniLM-L12-v2`（多语言，中文可用） |
| 前端 | 原生 HTML / CSS / JS + CDN（marked / DOMPurify / highlight.js） |

## 目录结构

```
app/
├── app.py              # 主入口：REST 路由 + SSE 转发 + 工具注册与 ReAct 循环
├── db.py               # SQLite 数据层（会话 + 消息，事务管理）
├── rag.py              # RAG 向量检索 + Obsidian 笔记导入
├── context_manager.py  # 上下文组装（滑动窗口 + 早期摘要 + token 截断）
├── migrate.py          # 一次性数据库迁移脚本
├── start.bat           # Windows 一键启动（自动起 Ollama/装依赖/迁移/起服务）
├── requirements.txt    # 依赖清单（版本固定）
├── config.example.json # 配置模板 → 复制为 config.json 填写笔记库路径
├── templates/          # 前端骨架 index.html
├── static/             # style.css + script.js
└── tests/              # pytest 单元测试（工具函数 + 路径白名单）
```

## 快速开始

### 1. 环境准备

- 安装 [Python](https://www.python.org/) 3.12+
- 安装 [Ollama](https://ollama.com/) 并拉取模型：

```bash
ollama pull qwen2.5:7b      # 必需（默认模型）
ollama pull qwen3.5:9b      # 可选（第二个白名单模型）
```

### 2. 安装依赖

```bash
cd app
pip install -r requirements.txt
```

> RAG 依赖 `torch` 体积较大（约 2GB）。若只需聊天、不装 RAG，可只装 `flask`、`requests`、`ddgs`——缺少 chromadb / sentence-transformers 时程序会自动降级（聊天照常，记忆功能关闭）。

### 3. 配置笔记库路径（可选）

Obsidian 联动才需要。二选一：

```bash
# 方式 A：复制模板
cp app/config.example.json app/config.json   # 然后编辑 obsidian_vault_path

# 方式 B：环境变量（优先级更高）
# Windows:  set OBSIDIAN_VAULT_PATH=D:\MyVault
# Linux:    export OBSIDIAN_VAULT_PATH=/home/you/MyVault
```

### 4. 启动

```bash
cd app
python app.py
# 或 Windows 双击 app\start.bat
```

浏览器访问 <http://127.0.0.1:5000>。

首次运行会自动下载 embedding 模型（约 90–470MB）；国内网络已预设 `HF_ENDPOINT=https://hf-mirror.com` 镜像。

## 配置项（环境变量）

| 变量 | 说明 |
|---|---|
| `OBSIDIAN_VAULT_PATH` | Obsidian 笔记库路径（优先于 config.json） |
| `MODEL_NAME` | 强制指定模型（如 `qwen3.5:9b`） |
| `OLLAMA_EXE` | Ollama 可执行文件路径（`start.bat` 用，默认走 PATH） |
| `FLASK_DEV` | 设为 `1` 时开放 `0.0.0.0` + debug（仅限开发） |
| `HF_ENDPOINT` | HuggingFace 下载镜像 |
| `MEMORY_DISTANCE_THRESHOLD` | RAG 相关性阈值 |

## 运行测试

```bash
python -m pytest app/tests/test_tools.py
```

## 隐私说明

- 默认仅监听 `127.0.0.1`，不对局域网开放。
- 不主动开启 CORS，避免第三方网页跨站读取本地数据。
- 对话记录、向量库、Obsidian 笔记均本地存储，已通过 `.gitignore` 排除出版本库。
- 项目不含任何硬编码密钥；笔记库路径通过 `config.json` / 环境变量配置，不上传。

## 许可证

MemFlow 项目采用 CC BY-NC-SA 4.0 协议进行许可。

=======================================================================

CC BY-NC-SA 4.0 (署名-非商业性使用-相同方式共享 4.0 国际版)

您可以自由地：
  - 共享：在任何媒介以任何形式复制、发行本作品
  - 演绎：修改、转换或以本作品为基础进行创作

惟须遵守下列条件：
  - 署名：您必须给出适当的署名，提供指向本许可协议的链接，同时标明是否对原始作品作了修改。
  - 非商业性使用：您不得将本作品用于商业目的（如盈利、售卖等）。
  - 相同方式共享：如果您再混合、转换或者基于本作品进行创作，您必须基于与原先许可协议相同的许可协议分发您贡献的作品。

完整协议内容请见：https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-Hans

© 2026 MemFlow Team. 保留部分权利。
