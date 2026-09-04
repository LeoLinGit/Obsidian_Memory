# -*- coding: utf-8 -*-
"""
rag.py —— RAG 长期记忆（向量存储 + 检索）

用 ChromaDB 做本地持久化向量库（PersistentClient，目录 app/chroma_db/），
embedding 用 sentence-transformers 的 paraphrase-multilingual-MiniLM-L12-v2（多语言含中文，CPU 可跑）。

若 ChromaDB / sentence-transformers 未安装、或初始化失败，会优雅降级：
RAG 功能关闭（is_available() 返回 False），但普通聊天完全不受影响。

对外函数：
    init_chroma_db()                      初始化 ChromaDB 与 embedding 模型
    add_message_to_chroma(...)            单条消息向量化入库
    index_messages(messages)              批量向量化入库（聊天落库后后台调用）
    search_similar_messages(query, k)     检索 top-k 相似历史 / Obsidian 笔记
    sync_all_messages_to_chroma()         全量增量同步（首次运行 / 迁移后）
    import_obsidian_vault(vault_path)     全量导入 Obsidian 笔记
    sync_obsidian_vault(vault_path)       增量同步 Obsidian 笔记
    get_obsidian_stats()                  统计 Obsidian 笔记入库情况
    is_available()                        是否可用
"""

# ---- 强制使用 HuggingFace 国内镜像，解决模型下载超时问题 ----
# 必须在所有第三方库 import 之前设置，这样 sentence-transformers 下载模型时才会走镜像
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import hashlib
import logging
import re
import threading
from datetime import datetime

logger = logging.getLogger("memflow.rag")

# ---- 依赖懒加载 + 优雅降级 ----
try:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    _IMPORT_OK = True
except ImportError as e:
    chromadb = None
    SentenceTransformerEmbeddingFunction = None
    _IMPORT_OK = False
    logger.warning("未安装 chromadb / sentence-transformers，RAG 功能将关闭：%s", e)

# 数据目录：app/chroma_db/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

COLLECTION_NAME = "memflow_messages_v2"   # 换 embedding 模型后改版本号，强制新建 collection（旧库语义空间已失效，避免 embedding_function 不匹配报错）
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Obsidian 笔记导入相关常量
OBS_SOURCE = "obsidian"   # 笔记向量的来源标识（聊天记录无此字段，默认视为 "chat"）
CHUNK_MAX = 500           # 单段最大字数，超过则按句子拆分
# memory_search 相关性门槛：cosine distance 超过该值视为不相关并过滤（0=完全相同，2=完全相反）
# 可用环境变量 MEMORY_DISTANCE_THRESHOLD 覆盖，方便按实际检索效果调参。
SIMILARITY_DISTANCE_THRESHOLD = float(os.environ.get("MEMORY_DISTANCE_THRESHOLD", "1.5"))
# 去重门槛：cosine distance 小于该值（≈相似度 > 0.95）视为重复内容，写入前跳过
DEDUP_DISTANCE_THRESHOLD = 0.05

# 全局单例
_client = None
_collection = None
_available = False

# 初始化锁：防止后台加载线程与请求线程并发重复初始化（避免重复下载/加载模型）
_init_lock = threading.Lock()


def is_available():
    """RAG 是否可用（ChromaDB + embedding 都初始化成功）。"""
    return _available


def init_chroma_db():
    """
    初始化 ChromaDB 和 embedding 模型。线程安全（加锁），失败优雅降级（RAG 关闭），不抛异常。

    注意：构造 embedding 函数会立即下载/加载模型（首次约 470MB，之后走本地缓存），
    耗时较长，因此建议用 start_background_init() 在后台执行，避免阻塞调用方。
    """
    global _client, _collection, _available
    with _init_lock:
        if _available:            # 已初始化成功，直接返回（幂等）
            return True
        if not _IMPORT_OK:
            logger.warning("依赖缺失，跳过 RAG 初始化")
            _available = False
            return False
        try:
            os.makedirs(CHROMA_DIR, exist_ok=True)
            # 构造 embedding 函数即触发模型下载/加载（首次约 90MB，之后走本地缓存）
            ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
            _client = chromadb.PersistentClient(path=CHROMA_DIR)
            _collection = _client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
            _available = True
            logger.info("ChromaDB 初始化成功，collection=%s", COLLECTION_NAME)
            return True
        except Exception as e:
            logger.exception("ChromaDB 初始化失败，RAG 关闭")
            _available = False
            return False


def start_background_init():
    """
    在后台线程初始化 RAG（不阻塞调用方，例如 Flask 启动）。
    加载期间 is_available() 返回 False，聊天照常；就绪后 RAG 自动生效。
    """
    threading.Thread(target=init_chroma_db, daemon=True).start()


def _ensure_collection():
    """确保 collection 已初始化，返回 collection 或 None。"""
    if not _available or _collection is None:
        init_chroma_db()
    return _collection if _available else None


def add_message_to_chroma(message_id, conversation_id, role, content, timestamp):
    """
    单条消息向量化入库（写入前先查重：已有相似度 > 0.95 的重复内容则跳过）。
    失败仅记日志并返回 False，绝不抛异常（避免影响聊天主流程）。
    """
    col = _ensure_collection()
    if col is None or not content:
        return False
    # 去重：检索最相似的一条，若相似度 > 0.95（distance < 0.05）则跳过；
    # 查重失败时降级为「不去重」，仍正常入库，避免因查重异常导致消息漏存。
    try:
        res = col.query(query_texts=[content], n_results=1, include=["distances"])
        dists = (res.get("distances") or [[]])[0] if res else []
        if dists and dists[0] is not None and dists[0] < DEDUP_DISTANCE_THRESHOLD:
            return False
    except Exception as e:
        logger.warning("去重检索失败，跳过查重直接入库：%s", e)
    try:
        col.add(
            ids=[str(message_id)],
            documents=[content],
            metadatas=[{
                "message_id": int(message_id),
                "conversation_id": int(conversation_id),
                "role": role,
                "timestamp": timestamp or "",
                "content": content,
            }],
        )
        return True
    except Exception as e:
        logger.warning("消息 %s 向量化入库失败：%s", message_id, e)
        return False


def index_messages(messages):
    """
    批量向量化入库：把一批消息（含 id / conversation_id / role / content / timestamp）
    写入向量库。供 app.py 在聊天落库后后台调用。失败仅记日志，不抛异常。
    """
    col = _ensure_collection()
    if col is None:
        return 0
    added = 0
    for m in messages:
        if m.get("role") == "tool":
            continue  # 工具执行结果不入向量库，避免污染长期记忆
        if add_message_to_chroma(m["id"], m["conversation_id"], m["role"], m["content"], m["timestamp"]):
            added += 1
    return added


def _parse_timestamp(value):
    """把各种时间戳格式解析为 datetime，失败返回 None。"""
    if not value:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value)
        except (ValueError, OSError):
            return None
    s = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromtimestamp(float(s))   # 可能是 unix 时间戳字符串
    except (ValueError, OSError):
        return None


def _time_decay(dt, now=None):
    """时间衰减系数：1 / (1 + 天数差/30)，越近越高（上限 1.0）。"""
    now = now or datetime.now()
    if dt is None:
        return 1.0
    days = max(0.0, (now - dt).total_seconds() / 86400.0)
    return 1.0 / (1.0 + days / 30.0)


def search_similar_messages(query, top_k=5):
    """
    检索与 query 最相似的 top_k 条历史消息 / Obsidian 笔记。
    超过相关性门槛（SIMILARITY_DISTANCE_THRESHOLD）的结果会被过滤掉。
    按「相似度 × 时间衰减」打分并降序排序（越新越相关权重越高）。
    返回 list[dict]，字段：id / source("chat"|"obsidian") / role / content /
    timestamp / conversation_id / distance / similarity / decay / score；
    source=="obsidian" 时额外含 file_name。
    失败或全部被过滤时返回空列表。
    """
    col = _ensure_collection()
    if col is None or not query:
        return []
    try:
        res = col.query(
            query_texts=[query],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
    except Exception as e:
        logger.warning("向量检索失败：%s", e)
        return []

    ids = (res.get("ids") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    hits = []
    for i, mid in enumerate(ids):
        distance = dists[i] if i < len(dists) else None
        if distance is None:
            continue
        # 相关性门槛：cosine distance 超过阈值视为不相关，直接过滤，避免无关“记忆”带偏模型
        if distance > SIMILARITY_DISTANCE_THRESHOLD:
            continue
        meta = metas[i] if i < len(metas) else {}
        source = meta.get("source", "chat")   # 聊天记录无 source 字段，默认视为 chat
        similarity = max(0.0, 1.0 - distance)
        decay = _time_decay(_parse_timestamp(meta.get("timestamp") or meta.get("file_mtime")))
        hit = {
            "id": mid,
            "source": source,
            "role": meta.get("role", ""),
            "content": (docs[i] if i < len(docs) else "") or meta.get("content", ""),
            "timestamp": meta.get("timestamp", ""),
            "conversation_id": meta.get("conversation_id"),
            "distance": round(distance, 4),
            "similarity": round(similarity, 4),
            "decay": round(decay, 4),
            "score": round(similarity * decay, 4),
        }
        if source == OBS_SOURCE:
            hit["file_name"] = meta.get("file_name", "")
            hit["file_mtime"] = meta.get("file_mtime", "")
        hits.append(hit)
    hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)
    return hits


def sync_all_messages_to_chroma():
    """
    全量增量同步：遍历 messages 表所有消息，跳过已入库的（按 message_id 判重）。
    用于首次运行或迁移后把历史消息一次性灌入向量库。
    返回 {"status": ..., "added": ..., "skipped": ...}
    """
    col = _ensure_collection()
    if col is None:
        return {"status": "disabled", "added": 0, "skipped": 0}

    import db  # 延迟导入，避免循环依赖

    messages = db.list_all_messages()
    # 已入库的 message_id 集合，避免重复向量化
    try:
        existing = set(col.get(include=["metadatas"]).get("ids", []))
    except Exception as e:
        logger.warning("读取 Chroma 已有 ids 失败：%s", e)
        existing = set()

    added = skipped = 0
    for m in messages:
        if str(m["id"]) in existing:
            skipped += 1
            continue
        if m.get("role") == "tool":
            skipped += 1  # 工具执行结果不入向量库
            continue
        if add_message_to_chroma(m["id"], m["conversation_id"], m["role"], m["content"], m["timestamp"]):
            added += 1
        else:
            skipped += 1
    logger.info("全量同步完成：新增 %d 条，跳过 %d 条", added, skipped)
    return {"status": "ok", "added": added, "skipped": skipped}


# ==================== Obsidian 笔记导入 ====================
def _path_hash(rel_path):
    """对相对路径做稳定哈希，用于生成唯一向量 ID。"""
    return hashlib.md5((rel_path or "").encode("utf-8")).hexdigest()[:12]


def _strip_frontmatter(text):
    """去掉 YAML Frontmatter（文件开头 --- 到 ---/... 之间的部分）。"""
    text = (text or "").lstrip("\ufeff")
    if not text.startswith("---"):
        return text
    lines = text.splitlines()
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            return "\n".join(lines[i + 1:])
    return text


def _chunk_text(text):
    """
    把笔记正文拆成若干段：
      1) 按空行拆段落；
      2) 超过 500 字的段落按句子（。！？!?；;）切分后，贪婪合并到接近 500 字；
      3) 极少数无标点的超长句，按 500 字硬切兜底。
    """
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= CHUNK_MAX:
            chunks.append(para)
            continue
        # 长段落：先按句子切分，再贪婪合并到接近 500 字
        sentences = [s for s in re.split(r"(?<=[。！？!?；;])", para) if s.strip()]
        buf = ""
        for s in sentences:
            if len(s) > CHUNK_MAX:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(s[i:i + CHUNK_MAX] for i in range(0, len(s), CHUNK_MAX))
            elif len(buf) + len(s) <= CHUNK_MAX:
                buf += s
            else:
                chunks.append(buf)
                buf = s
        if buf:
            chunks.append(buf)
    return chunks


def _import_one_file(col, vault_path, rel_path):
    """读取并向量化单个笔记文件，返回写入的 chunk 数；失败/无内容返回 0。"""
    full = os.path.join(vault_path, rel_path)
    try:
        with open(full, "r", encoding="utf-8-sig") as f:
            raw = f.read()
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("读取笔记失败 %s：%s", rel_path, e)
        return 0

    body = _strip_frontmatter(raw)
    chunks = _chunk_text(body)
    if not chunks:
        return 0

    ph = _path_hash(rel_path)
    mtime = str(os.path.getmtime(full))
    file_name = os.path.splitext(os.path.basename(rel_path))[0]
    ids, documents, metadatas = [], [], []
    for i, c in enumerate(chunks):
        ids.append("obs_%s_%d" % (ph, i))
        documents.append(c)
        metadatas.append({
            "source": OBS_SOURCE,
            "file_path": rel_path,
            "file_name": file_name,
            "chunk_index": i,
            "file_mtime": mtime,
            "content": c,
        })
    try:
        col.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(chunks)
    except Exception as e:
        logger.warning("笔记 %s 向量化失败：%s", rel_path, e)
        return 0


def _delete_chunks(col, ids):
    """删除指定 id 的向量。"""
    if not ids:
        return
    try:
        col.delete(ids=ids)
    except Exception as e:
        logger.warning("删除向量失败：%s", e)


def _iter_md_files(vault_path):
    """遍历 vault 下所有应导入的 .md 文件，yield 相对路径。"""
    for root, dirs, files in os.walk(vault_path):
        # 跳过隐藏目录（.obsidian / .trash / .git 等）
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in files:
            if not fn.lower().endswith(".md"):
                continue
            # 跳过以 _ 或 . 开头的文件
            if fn.startswith("_") or fn.startswith("."):
                continue
            yield os.path.relpath(os.path.join(root, fn), vault_path)


def import_obsidian_vault(vault_path):
    """
    全量导入 Obsidian Vault 下所有 .md 笔记。
    返回 {"status", "files_scanned", "chunks_added", "skipped"}。
    """
    col = _ensure_collection()
    if col is None:
        return {"status": "disabled", "files_scanned": 0, "chunks_added": 0, "skipped": 0}

    vault_path = (vault_path or "").strip()
    if not vault_path or not os.path.isdir(vault_path):
        logger.warning("Obsidian Vault 路径无效：%s", vault_path)
        return {"status": "error", "detail": "vault 路径无效", "files_scanned": 0, "chunks_added": 0, "skipped": 0}

    files_scanned = chunks_added = skipped = 0
    for rel in _iter_md_files(vault_path):
        files_scanned += 1
        n = _import_one_file(col, vault_path, rel)
        if n:
            chunks_added += n
        else:
            skipped += 1

    logger.info("Obsidian 全量导入完成：扫描 %d 个文件，新增 %d 段，跳过 %d", files_scanned, chunks_added, skipped)
    return {"status": "ok", "files_scanned": files_scanned, "chunks_added": chunks_added, "skipped": skipped}


def sync_obsidian_vault(vault_path):
    """
    增量同步 Obsidian Vault：
      - 按文件 mtime 判断新增 / 更新 / 未变；
      - 删除 vault 中已不存在文件对应的旧向量。
    返回 {"status", "added", "updated", "deleted", "unchanged"}。
    """
    col = _ensure_collection()
    if col is None:
        return {"status": "disabled", "added": 0, "updated": 0, "deleted": 0, "unchanged": 0}

    vault_path = (vault_path or "").strip()
    if not vault_path or not os.path.isdir(vault_path):
        return {"status": "error", "detail": "vault 路径无效", "added": 0, "updated": 0, "deleted": 0, "unchanged": 0}

    # 1) 扫描当前 vault：rel_path -> mtime
    current = {}
    for rel in _iter_md_files(vault_path):
        current[rel] = str(os.path.getmtime(os.path.join(vault_path, rel)))

    # 2) 读取已有 obsidian 向量，按 file_path 分组
    existing = {}   # rel_path -> {"ids": [...], "mtime": str}
    try:
        data = col.get(where={"source": OBS_SOURCE}, include=["metadatas"])
    except Exception as e:
        logger.warning("读取 Obsidian 已有向量失败：%s", e)
        return {"status": "error", "detail": "读取向量库失败", "added": 0, "updated": 0, "deleted": 0, "unchanged": 0}

    metas = data.get("metadatas") or []
    for i, cid in enumerate(data.get("ids") or []):
        meta = metas[i] if i < len(metas) else {}
        rel = meta.get("file_path", "")
        if not rel:
            continue
        existing.setdefault(rel, {"ids": [], "mtime": ""})
        existing[rel]["ids"].append(cid)
        existing[rel]["mtime"] = meta.get("file_mtime", "")

    added = updated = deleted = unchanged = 0

    # 3) 遍历当前文件：新增 / 更新 / 未变
    for rel, mtime in current.items():
        if rel not in existing:
            if _import_one_file(col, vault_path, rel):
                added += 1
        elif existing[rel]["mtime"] != mtime:
            _delete_chunks(col, existing[rel]["ids"])
            if _import_one_file(col, vault_path, rel):
                updated += 1
        else:
            unchanged += 1

    # 4) 删除已不存在文件的旧向量
    for rel, info in existing.items():
        if rel not in current:
            _delete_chunks(col, info["ids"])
            deleted += 1

    logger.info("Obsidian 增量同步完成：新增 %d，更新 %d，删除 %d，未变 %d", added, updated, deleted, unchanged)
    return {"status": "ok", "added": added, "updated": updated, "deleted": deleted, "unchanged": unchanged}


def get_obsidian_stats():
    """统计已入库的 Obsidian 笔记：文件数与 chunk 数。"""
    col = _ensure_collection()
    if col is None:
        return {"files": 0, "chunks": 0}
    try:
        data = col.get(where={"source": OBS_SOURCE}, include=["metadatas"])
    except Exception as e:
        logger.warning("统计 Obsidian 向量失败：%s", e)
        return {"files": 0, "chunks": 0}
    metas = data.get("metadatas") or []
    files = len({m.get("file_path", "") for m in metas if m.get("file_path")})
    chunks = len(data.get("ids") or [])
    return {"files": files, "chunks": chunks}