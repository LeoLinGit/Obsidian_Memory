# -*- coding: utf-8 -*-
"""
db.py —— MemFlow 数据库层

把原来「messages 以 JSON 字符串整坨存在 conversations 表」的结构，
重构为规范化的两张关系表：
    conversations（会话） + messages（消息，带外键与预留 embedding_id）

对外提供：
    init_db / create_conversation / list_conversations
    get_conversation / delete_conversation / append_messages
    migrate_from_old_json（一次性数据迁移）

所有连接都通过 get_conn() 上下文管理器管理，确保异常时自动关闭连接、
自动 commit / rollback。
"""

import json
import logging
import os
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger("memflow.db")

# 数据库文件与 db.py 同目录（app/ 下）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "conversations.db")

DEFAULT_TITLE = "新对话"


def _now():
    """统一时间戳格式（沿用旧版字符串格式，保证前端无需改动）。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def get_conn():
    """
    获取数据库连接的上下文管理器。

    注意：sqlite3 连接自带的 with 只会 commit/rollback，并不会关闭连接；
    因此这里用 contextlib.contextmanager 在 finally 中手动 close，
    保证「异常时自动关闭连接」这一需求真正成立。

    同时开启外键约束（让 ON DELETE CASCADE 生效）、行工厂和 busy_timeout。
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("数据库操作失败，已回滚")
        raise
    finally:
        conn.close()


# ---------------- 建表 ----------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT    NOT NULL,
    created_at TEXT    NOT NULL,
    updated_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role            TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL,
    tool_calls      TEXT,
    tool_name       TEXT,
    embedding_id    TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages (conversation_id, timestamp);
"""


def init_db():
    """初始化数据库：建表 + 建索引（幂等，可重复调用）。"""
    with get_conn() as conn:
        conn.executescript(SCHEMA_SQL)
        _migrate_messages_table(conn)


def _migrate_messages_table(conn):
    """为旧库补齐新增列（tool_calls / tool_name），幂等。"""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(messages)").fetchall()}
    if "tool_calls" not in cols:
        conn.execute("ALTER TABLE messages ADD COLUMN tool_calls TEXT")
    if "tool_name" not in cols:
        conn.execute("ALTER TABLE messages ADD COLUMN tool_name TEXT")


# ---------------- 会话 CRUD ----------------
def create_conversation(title=DEFAULT_TITLE):
    """新建会话：只往 conversations 表插一条记录，返回新会话 id。"""
    now = _now()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, now, now),
        )
        conv_id = cur.lastrowid
    return conv_id


def list_conversations():
    """列出所有会话，按最后更新时间倒序（与旧版返回结构一致）。"""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, updated_at FROM conversations "
            "ORDER BY updated_at DESC, id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_conversation(conv_id):
    """
    按 id 取会话，并通过 conversation_id 关联 messages 表，
    按 timestamp（再按 id）升序返回消息列表。
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conv_id,),
        ).fetchone()
        if row is None:
            return None
        conv = dict(row)
        msg_rows = conn.execute(
            "SELECT role, content, timestamp, tool_calls, tool_name FROM messages "
            "WHERE conversation_id = ? ORDER BY timestamp ASC, id ASC",
            (conv_id,),
        ).fetchall()
        conv["messages"] = [_row_to_message(m) for m in msg_rows]
        return conv


def _row_to_message(row):
    """把 messages 表的一行转成 dict，并把 tool_calls 从 JSON 字符串还原为 list。"""
    m = dict(row)
    tc = m.get("tool_calls")
    if tc:
        try:
            m["tool_calls"] = json.loads(tc)
        except (ValueError, TypeError):
            m["tool_calls"] = None
    return m


def get_recent_messages(conv_id, limit=8):
    """取会话最近 limit 条消息（按时间正序返回，供滑动窗口用）。"""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, conversation_id, role, content, timestamp FROM messages "
            "WHERE conversation_id = ? ORDER BY timestamp ASC, id ASC",
            (conv_id,),
        ).fetchall()
        msgs = [dict(r) for r in rows]
    return msgs[-limit:]


def list_all_messages():
    """列出全部消息（跨会话），供 RAG 全量同步用。"""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, conversation_id, role, content, timestamp FROM messages ORDER BY id ASC"
        ).fetchall()
        return [dict(r) for r in rows]


def delete_conversation(conv_id):
    """删除会话；messages 表的外键 ON DELETE CASCADE 会自动级联删除对应消息。"""
    with get_conn() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))


# ---------------- 消息写入 ----------------
def append_messages(conv_id, messages, new_title=None):
    """
    在一个事务里批量插入消息，并刷新会话 updated_at；
    若传入 new_title 则同时更新标题（用于 AI 首句自动命名）。

    messages: [{"role": "user", "content": "..."}, ...]，可选字段 tool_calls / tool_name 用于落库工具调用轨迹。
    返回：插入的消息列表（含 id / conversation_id / role / content / timestamp），供 RAG 向量化用。
    """
    now = _now()
    inserted = []
    with get_conn() as conn:
        for m in messages:
            tool_calls = m.get("tool_calls")
            tool_calls_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
            tool_name = m.get("tool_name") or None
            cur = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, timestamp, tool_calls, tool_name) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (conv_id, m["role"], m["content"], now, tool_calls_json, tool_name),
            )
            inserted.append({
                "id": cur.lastrowid,
                "conversation_id": conv_id,
                "role": m["role"],
                "content": m["content"],
                "timestamp": now,
                "tool_calls": tool_calls,
                "tool_name": tool_name,
            })
        if new_title is not None:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (new_title, now, conv_id),
            )
        else:
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conv_id),
            )
    return inserted


# ---------------- 数据迁移 ----------------
def migrate_from_old_json(backup_path=None):
    """
    一次性迁移：旧结构（messages 为 JSON 字符串） -> 新结构（messages 表）。

    流程：
      1) 读取旧 conversations 表，逐条解析 messages JSON；
      2) 备份旧库（默认 conversations_old.db，已存在则追加时间戳）；
      3) 重建表结构（conversations 去掉 messages 列 + 新增 messages 表）；
      4) 按原顺序回填会话与消息，保留会话 id 不变。

    返回 (会话数, 消息数)。
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"未找到数据库：{DB_PATH}")

    # 已是新结构则拒绝重复迁移，避免破坏数据
    with get_conn() as conn:
        already = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'"
        ).fetchone()
    if already:
        raise RuntimeError("数据库已是新结构（存在 messages 表），无需重复迁移。")

    # 1) 读取旧数据
    old_data = []
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at, messages FROM conversations"
        ).fetchall()
        for r in rows:
            messages = []
            try:
                raw = json.loads(r["messages"] or "[]")
                if isinstance(raw, list):
                    for m in raw:
                        if isinstance(m, dict) and m.get("role") and m.get("content"):
                            messages.append({"role": m["role"], "content": m["content"]})
            except (ValueError, TypeError) as e:
                logger.warning("会话 %s 的 messages JSON 解析失败，已跳过：%s", r["id"], e)
            old_data.append({
                "id": r["id"],
                "title": r["title"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "messages": messages,
            })

    # 2) 备份旧库
    backup = backup_path or os.path.join(BASE_DIR, "conversations_old.db")
    if os.path.exists(backup):
        base, ext = os.path.splitext(backup)
        backup = f"{base}_{datetime.now().strftime('%Y%m%d%H%M%S')}{ext}"
    shutil.copy2(DB_PATH, backup)
    logger.info("已备份旧数据库到：%s", backup)

    # 3) 重建表结构（先删子表 messages 再删父表，避免外键约束报错）
    with get_conn() as conn:
        conn.executescript(
            "DROP TABLE IF EXISTS messages;\n"
            "DROP TABLE IF EXISTS conversations;\n"
            + SCHEMA_SQL
        )

    # 4) 回填数据（旧 JSON 无逐条时间戳，统一用会话 created_at，靠 id 保持顺序）
    total_msgs = 0
    with get_conn() as conn:
        for conv in old_data:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (conv["id"], conv["title"], conv["created_at"], conv["updated_at"]),
            )
            for m in conv["messages"]:
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, timestamp) "
                    "VALUES (?, ?, ?, ?)",
                    (conv["id"], m["role"], m["content"], conv["created_at"]),
                )
                total_msgs += 1

    logger.info("迁移完成：%d 个会话，%d 条消息", len(old_data), total_msgs)
    return len(old_data), total_msgs
