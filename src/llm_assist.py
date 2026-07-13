# -*- coding: utf-8 -*-
"""LLM補助: 回顧メモの自由文を、評価/タグ/狙い度に構造化する。

APIキーは Streamlit secrets（ANTHROPIC_API_KEY）または環境変数から読む。
未設定なら available()=False（＝手動運用にフォールバック）。
"""
import os
import re
import json

MODEL = "claude-haiku-4-5-20251001"   # 安価・高速。メモ整形に十分


def _get_key() -> str:
    key = ""
    try:
        import streamlit as st
        key = st.secrets.get("ANTHROPIC_API_KEY", "") or ""
    except Exception:
        key = ""
    return key or os.environ.get("ANTHROPIC_API_KEY", "") or ""


def available() -> bool:
    return bool(_get_key())


def suggest_from_memo(memo_text: str, eval_options: list, all_tags: list) -> dict:
    """観察メモの自由文 → {'評価','タグ':[...],'狙い度':int,'要約':str}。
    APIキー未設定・失敗時は例外を投げる（呼び出し側で握る）。"""
    key = _get_key()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY が未設定です（secretsに登録してください）。")
    if not str(memo_text).strip():
        raise ValueError("メモが空です。")

    import anthropic
    client = anthropic.Anthropic(api_key=key)

    system = (
        "あなたは競馬のレース回顧メモを構造化するアシスタントです。"
        "入力された日本語の観察メモを、指定された語彙だけを使って分類し、JSONのみで返します。"
        "説明文やコードブロックは書かず、JSONオブジェクト1つだけを出力してください。"
    )
    user = (
        f"# 観察メモ\n{memo_text}\n\n"
        f"# 出力（JSONのみ・キーは日本語）\n"
        f'{{"評価": 次のいずれか1つ {eval_options}, '
        f'"タグ": 次のリストから該当するものだけ複数可 {all_tags}, '
        f'"狙い度": 1〜3の整数（次走の妙味・注目度が高いほど大きい）, '
        f'"要約": 15字程度の短い要約}}\n'
        f"※タグは必ず上記リストの表記と完全一致で選ぶこと。該当が無ければ空配列[]。"
    )
    resp = client.messages.create(
        model=MODEL, max_tokens=400, system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(getattr(b, "text", "") for b in resp.content)
    m = re.search(r"\{.*\}", text, re.S)
    data = json.loads(m.group(0) if m else text)

    ev = str(data.get("評価", "中立"))
    if ev not in eval_options:
        ev = "中立"
    tags = [t for t in (data.get("タグ") or []) if t in all_tags]
    try:
        aim = int(data.get("狙い度", 2))
    except Exception:
        aim = 2
    aim = max(1, min(3, aim))
    return {"評価": ev, "タグ": tags, "狙い度": aim, "要約": str(data.get("要約", "")).strip()}
