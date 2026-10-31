"""翻译阶段按时间窗切分与提示词约束测试。"""

from __future__ import annotations

import json

from whisper_stt_service.executors import (
    SrtEntry,
    _build_translate_messages,
    _split_entries_by_time_window,
)


def test_split_entries_by_time_window_uses_30min_boundaries() -> None:
    """同一 30 分钟窗口内的字幕应归并到同一批次。"""

    entries = [
        SrtEntry(index=1, timestamp="00:00:01,000 --> 00:00:02,000", text="a"),
        SrtEntry(index=2, timestamp="00:29:59,000 --> 00:30:01,000", text="b"),
        SrtEntry(index=3, timestamp="00:30:00,000 --> 00:30:02,000", text="c"),
        SrtEntry(index=4, timestamp="01:00:00,000 --> 01:00:03,000", text="d"),
    ]

    batches = _split_entries_by_time_window(entries, window_minutes=30)

    assert [len(batch) for batch in batches] == [2, 1, 1]
    assert [entry.index for entry in batches[0]] == [1, 2]
    assert [entry.index for entry in batches[1]] == [3]
    assert [entry.index for entry in batches[2]] == [4]


def test_build_translate_messages_requests_plot_reasoning_context() -> None:
    """提示词应强调剧情上下文推理与称呼一致性。"""

    messages = _build_translate_messages(
        [SrtEntry(index=7, timestamp="00:11:00,000 --> 00:11:03,000", text="行こう")]
    )

    assert len(messages) == 2
    system_text = messages[0]["content"].lower()
    user_text = messages[1]["content"].lower()
    assert "plot" in system_text
    assert "context" in system_text
    assert "consistent" in system_text
    assert "disfluencies" in system_text
    assert "repetitions" in system_text
    # 输出契约：以 id 为 key 的 JSON 对象（修复"整段后移一位"），不再是
    # [{id, text_zh}] 数组；并显式禁止合并/拆分/新增/丢弃条目。
    assert '"<id>"' in user_text
    assert '"text_zh"' not in user_text
    assert "exactly 1 subtitle items" in user_text
    assert "no more, no fewer" in user_text
    assert "never merge" in user_text
    assert "never split" in user_text
    assert "never add, drop or renumber" in user_text
    assert "denoising" in user_text


def test_build_translate_messages_states_item_count_for_dict_contract() -> None:
    """条数必须写进提示词：key 数量 == 输入条数是可校验的硬约束。"""

    messages = _build_translate_messages(
        [
            SrtEntry(index=84, timestamp="00:11:00,000 --> 00:11:03,000", text="行こう"),
            SrtEntry(index=85, timestamp="00:11:03,000 --> 00:11:05,000", text="待って"),
        ]
    )

    user_text = messages[1]["content"]
    assert "exactly 2 subtitle items" in user_text
    # 输入侧载荷仍是 [{id, text}] 数组，模型逐条对照 id。
    payload = json.loads(user_text.split("\n\n", 1)[1])
    assert [item["id"] for item in payload] == [84, 85]
    assert [item["text"] for item in payload] == ["行こう", "待って"]
