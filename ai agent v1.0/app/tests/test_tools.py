# -*- coding: utf-8 -*-
"""
工具函数单元测试（pytest）。

覆盖 get_current_time / memory_search / file_read / file_write / web_search 的
正常调用与异常场景，以及路径白名单与工具失败判定两个辅助逻辑。
"""
import os
import re
import sys

# 让测试能 import app/ 下的模块（app.py 使用扁平 import：import db / rag / context_manager）
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import pytest

import app as app_module


# ---------- get_current_time ----------
def test_get_current_time_returns_formatted_string():
    result = app_module._tool_get_current_time({})
    assert isinstance(result, str) and result
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result)


# ---------- memory_search ----------
def test_memory_search_missing_query(monkeypatch):
    monkeypatch.setattr(app_module.rag, "is_available", lambda: True)
    assert "未提供检索关键词" in app_module._tool_memory_search({})
    assert "未提供检索关键词" in app_module._tool_memory_search({"query": ""})


def test_memory_search_unavailable(monkeypatch):
    monkeypatch.setattr(app_module.rag, "is_available", lambda: False)
    assert "记忆检索不可用" in app_module._tool_memory_search({"query": "anything"})


# ---------- web_search ----------
def test_web_search_missing_query():
    assert "未提供搜索关键词" in app_module._tool_web_search({})
    assert "未提供搜索关键词" in app_module._tool_web_search({"query": ""})


def test_parse_bing_html():
    html = (
        '<li class="b_algo"><h2><a href="https://example.com/page">'
        'Example Title</a></h2><p>This is a <b>snippet</b> about the result.</p></li>'
        '<li class="b_algo"><h2><a href="https://example.org/other">'
        'Another Title</a></h2><p>Another snippet text.</p></li>'
    )
    results = app_module._parse_bing_html(html, max_results=5)
    assert len(results) == 2
    title, snip, link = results[0]
    assert title == "Example Title"
    assert link == "https://example.com/page"
    assert "snippet" in snip


# ---------- file_read / file_write（路径白名单） ----------
@pytest.fixture
def whitelist(tmp_path, monkeypatch):
    """把白名单限定到 allowed 子目录，outside 子目录则在白名单之外。"""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(
        app_module,
        "_allowed_dirs",
        lambda: [os.path.normcase(os.path.abspath(str(allowed)))],
    )
    return allowed, outside


def test_file_write_and_read_roundtrip(whitelist):
    allowed, _ = whitelist
    p = str(allowed / "note.md")
    res = app_module._tool_file_write({"file_path": p, "content": "hello 世界", "mode": "write"})
    assert "已成功" in res
    assert (allowed / "note.md").read_text(encoding="utf-8") == "hello 世界"
    assert app_module._tool_file_read({"file_path": p}) == "hello 世界"


def test_file_write_append(whitelist):
    allowed, _ = whitelist
    p = str(allowed / "log.txt")
    app_module._tool_file_write({"file_path": p, "content": "a", "mode": "write"})
    app_module._tool_file_write({"file_path": p, "content": "b", "mode": "append"})
    assert (allowed / "log.txt").read_text(encoding="utf-8") == "ab"


def test_file_read_outside_whitelist_rejected(whitelist):
    _, outside = whitelist
    p = str(outside / "secret.txt")
    (outside / "secret.txt").write_text("x", encoding="utf-8")
    assert "拒绝访问" in app_module._tool_file_read({"file_path": p})


def test_file_write_outside_whitelist_rejected(whitelist):
    _, outside = whitelist
    p = str(outside / "evil.txt")
    assert "拒绝访问" in app_module._tool_file_write({"file_path": p, "content": "x"})


def test_file_read_missing_file(whitelist):
    allowed, _ = whitelist
    assert "未找到该文件" in app_module._tool_file_read({"file_path": str(allowed / "nope.txt")})


# ---------- file_read 文件名模糊搜索（支持只传文件名/关键词） ----------
def test_file_read_fuzzy_bare_filename(whitelist):
    """只传裸文件名（无路径），能在白名单目录内模糊匹配并读出内容。"""
    allowed, _ = whitelist
    (allowed / "简历及面试锻炼.md").write_text("面试内容", encoding="utf-8")
    res = app_module._tool_file_read({"file_path": "简历及面试锻炼.md"})
    assert "面试内容" in res


def test_file_read_fuzzy_substring(whitelist):
    """只传关键词「简历」，能命中文件名包含该关键词的文件。"""
    allowed, _ = whitelist
    (allowed / "简历及面试锻炼.md").write_text("面试内容", encoding="utf-8")
    res = app_module._tool_file_read({"file_path": "简历"})
    assert "面试内容" in res


def test_file_read_fuzzy_multiple_matches(whitelist):
    """多个文件匹配时，列出所有候选路径让用户选择。"""
    allowed, _ = whitelist
    (allowed / "简历A.md").write_text("A", encoding="utf-8")
    (allowed / "简历B.md").write_text("B", encoding="utf-8")
    res = app_module._tool_file_read({"file_path": "简历"})
    assert "找到 2 个匹配文件" in res


def test_file_read_directory(whitelist):
    allowed, _ = whitelist
    assert "目标是目录" in app_module._tool_file_read({"file_path": str(allowed)})


def test_file_read_missing_path():
    assert "未提供文件路径" in app_module._tool_file_read({})
    assert "未提供文件路径" in app_module._tool_file_read({"file_path": ""})


def test_file_write_missing_params():
    assert "未提供文件路径" in app_module._tool_file_write({})
    assert "未提供要写入的内容" in app_module._tool_file_write({"file_path": "a.md"})


def test_file_write_invalid_mode(whitelist):
    allowed, _ = whitelist
    res = app_module._tool_file_write(
        {"file_path": str(allowed / "a.md"), "content": "x", "mode": "overwrite"}
    )
    assert "不支持的写入模式" in res


# ---------- 工具失败判定（熔断用） ----------
def test_is_tool_error():
    assert app_module._is_tool_error("未知工具：foo")
    assert app_module._is_tool_error("工具 foo 执行出错：boom")
    assert app_module._is_tool_error("未提供文件路径")
    assert app_module._is_tool_error("拒绝访问：路径不在允许范围内")
    assert not app_module._is_tool_error("未找到相关记忆")
    assert not app_module._is_tool_error("已成功写入文件：a.md")
    assert not app_module._is_tool_error("")
    assert not app_module._is_tool_error(None)
