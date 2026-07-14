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
        "説明文やコードブロックは書かず、JSONオブジェクト1つだけを出力してください。\n"
        "# 分類ルール（重要）\n"
        "・物理的・進路上の不利（他馬との接触／進路を塞がれた／外を回された／4角など道中での不利 等）は"
        "『不利』系タグ（出遅れ・包まれる・進路なし・前が壁・挟まれる・接触・外々を回された・掛かる・折り合い欠く）から選ぶ。"
        "「4角不利」「道中不利」「不利があった」等も物理系として扱う。種類が判別できればそれを、"
        "判別できなければ最も近いもの（進路が無い旨→進路なし、外を回された旨→外々を回された）を選ぶ。\n"
        "・ペースや位置取りが敗因のときだけ『展開』系タグ（展開不利(前残り)・展開不利(差し届かず)・"
        "ハイペースで脚を使った・スローで後方待機・直線だけの競馬）を使う。物理的な不利がある場合は展開系より不利系を優先。\n"
        "・『展開不利(前残り)』は、後方（差し・追込）の馬が『前が残る（スロー〜前有利）展開』で届かなかった場合のみ。"
        "次のときは使わない：「先行が厳しい」「ハイペース」「前が総崩れ」等ペースが速い記述がある／この馬自身が逃げ・先行だった。"
        "先行・逃げ馬が速い流れで苦しくなったなら『ハイペースで脚を使った』を使う。\n"
        "・「先行厳しい展開で粘った／掲示板に残した」のように厳しい展開でむしろ好走した内容は、"
        "前残りではなく前向きな評価（見どころ十分 等）にする。\n"
        "・「次走」系タグ（次走○○で狙い）は、次走で狙う条件（距離延長/短縮・内枠/外枠・小回り・"
        "ワンターン・直線が長い/広い/平坦なコース 等）がメモに明示されている場合のみ選ぶ。書かれていなければ付けない。\n"
        "・メモに書かれていないことは推測でタグ付けしない。該当しないタグは付けない（過剰付与しない）。確信が持てないタグは省く。\n"
        "# 例\n"
        "メモ『先行厳しい展開で唯一掲示板に粘り込んだ。次走も狙いたい』→ "
        '{"評価":"次走注目","タグ":["見どころ十分"],"狙い度":2,"要約":"厳しい流れを先行で粘る"}'
        "（先行厳しい＝前残りではないので展開不利(前残り)は付けない）\n"
        "メモ『4角で不利があり次走見直したい。先行力が高くコースが向けば』→ "
        '{"評価":"次走注目","タグ":["進路なし"],"狙い度":2,"要約":"4角不利、条件好転待ち"}'
        "（4角不利＝物理的な不利。展開不利(前残り)は付けない）"
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
        model=MODEL, max_tokens=400, temperature=0, system=system,
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
