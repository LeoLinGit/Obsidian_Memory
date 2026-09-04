# -*- coding: utf-8 -*-
"""
migrate.py —— 一次性数据迁移脚本

把旧的 conversations.db（messages 以 JSON 字符串存在 conversations 表）
迁移到新的规范化结构（conversations + messages 两张表）。

用法（在 app/ 目录下）：
    python migrate.py

迁移前会自动把旧库备份为 conversations_old.db（已存在则追加时间戳）。
"""

import logging
import sys

import db


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        n_conv, n_msg = db.migrate_from_old_json()
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"[提示] {e}")
        sys.exit(0)

    print(f"[完成] 已迁移 {n_conv} 个会话、{n_msg} 条消息。")
    print("       旧库已备份为 conversations_old.db（或带时间戳的同名 .db）。")


if __name__ == "__main__":
    main()
