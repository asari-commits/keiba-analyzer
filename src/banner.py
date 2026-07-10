# -*- coding: utf-8 -*-
"""
SNS投稿用バナー画像（1080x1080 PNG）を生成する。

2枚:
  ① 予測サマリー … 印(◎○▲★△)・馬番(枠色)・馬名・騎手＋根拠タグ・人気/オッズ・AI予測複勝率
  ② コースの好調データ … 出走メンバー限定の 騎手/種牡馬/調教師 勝率TOP3（勝ち数/母数R＋該当馬）

日本語フォントは assets/fonts/NotoSansJP.ttf（OFL, 可変フォント）を同梱。
アプリ画面の表示と同じ値を使う（marks/_fuku_prob/get_reasons/course_top_performers）。
"""
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = Path(__file__).parent.parent / "assets" / "fonts" / "NotoSansJP.ttf"

W = H = 1080

C = {
    "bg": "#0d1117", "panel": "#161b22", "line": "#21262d", "border": "#30363d",
    "t1": "#f0f6fc", "t2": "#8b949e", "t3": "#6e7681", "trow": "#c9d1d9",
    "green": "#2ecc71", "gold": "#f5c518", "odds": "#f1c40f", "blue": "#58a6ff",
    "amber": "#e3b341", "purple": "#c39bd3",
    "pill_bg": "#1f2630", "pill_fg": "#adbac7", "pill_bg_s": "#3a2f10", "pill_fg_s": "#f5d97a",
    "star_on": "#f5c518", "star_off": "#3a3f47", "hero_bg": "#161b22", "star_row_bg": "#1c1526",
}

# 印 → (背景, 文字)
MARK_COLORS = {
    "◎": ("#f5c518", "#3d2c00"), "○": ("#2ecc71", "#08341f"),
    "▲": ("#e67e22", "#3d1f00"), "△": ("#3498db", "#082a45"),
    "★": ("#9b59b6", "#f0e6f7"),
}

# 枠番カラー（JRA公式）
_WAKU_COLORS = {
    1: ("#FFFFFF", "#000000"), 2: ("#2b2b2b", "#FFFFFF"), 3: ("#EE0000", "#FFFFFF"),
    4: ("#0066CC", "#FFFFFF"), 5: ("#FFD700", "#000000"), 6: ("#008000", "#FFFFFF"),
    7: ("#FF8C00", "#000000"), 8: ("#FF69B4", "#000000"),
}


def _umaban_to_waku(ub, n):
    n = max(1, int(n))
    if n <= 8:
        counts = [1] * n + [0] * (8 - n)
    elif n <= 16:
        ones = 16 - n
        counts = [1] * ones + [2] * (8 - ones)
    else:
        threes = n - 16
        counts = [2] * (8 - threes) + [3] * threes
    cum = 0
    for frame in range(1, 9):
        cum += counts[frame - 1]
        if ub <= cum:
            return frame
    return 8


# get_reasons の長いラベル → カード表示と同じ短縮チップ（app.py の _TAG_MAP と一致）
_TAG_SHORT = {
    "馬の平均着順が良い": "馬◎着順", "馬の過去勝率が高い": "馬◎勝率", "馬の複勝率が高い": "馬◎複勝",
    "このコースで好成績": "コース◎", "このコースの勝率が高い": "コース勝率", "このコースの複勝率が高い": "コース複勝",
    "脚質がこのコースに合う": "脚質適性◎", "先行有利コースで先行脚質": "先行◎",
    "騎手の勝率が高い": "騎手◎", "騎手の平均着順が良い": "騎手◎着順", "調教師の勝率が高い": "調教師◎",
    "前走上り3Fが速い": "前走上り◎", "直近3走の上り平均が速い": "上り安定", "前走着差が少ない（接戦）": "前走接戦",
    "直近3走の着差が少ない": "着差安定", "前走タイムが優秀": "前走タイム◎", "前走着順が良い": "前走◎",
    "直近3走の着順が安定": "近走安定", "前走4角で前目につけた": "前走先行", "距離延長・短縮が合う": "距離適性◎",
    "モデル総合スコアが上位": "総合◎",
    "血統が芝/ダに適性": "血統芝ダ", "血統がこの距離に合う": "血統距離", "産駒がこの条件(芝ダ×距離)得意": "産駒◎",
    "血統が道悪巧者": "道悪血統", "母父が芝/ダに適性": "母父芝ダ", "母父がこの距離に合う": "母父距離",
    "母父産駒がこの条件得意": "母父産駒◎", "母父が道悪巧者": "母父道悪",
    "騎手が芝/ダ得意": "騎手芝ダ", "騎手がこの距離得意": "騎手距離",
    "前走は展開負け→今回は先行向き": "展開向く", "前走は展開負け→今回は差し向き": "展開向く",
    "前走大敗を度外視・実績馬（巻き返し妙味）": "実績馬妙味",
}


def _short_tag(label):
    return _TAG_SHORT.get(label, str(label)[:6])


_font_cache = {}


def _font(size, weight="Regular"):
    key = (size, weight)
    if key not in _font_cache:
        f = ImageFont.truetype(str(FONT_PATH), size)
        try:
            f.set_variation_by_name(weight)
        except Exception:
            pass
        _font_cache[key] = f
    return _font_cache[key]


def _tw(dr, text, font):
    return dr.textlength(text, font=font)


def _trunc(dr, text, font, max_w):
    if _tw(dr, text, font) <= max_w:
        return text
    ell = "…"
    while text and _tw(dr, text + ell, font) > max_w:
        text = text[:-1]
    return text + ell


def _pill(dr, x, cy, text, font, bg, fg):
    """左端xに縦中央cyでピルを描画。右端xを返す。"""
    pad = 8
    tw = _tw(dr, text, font)
    h = font.size + 8
    box = (x, cy - h / 2, x + tw + pad * 2, cy + h / 2)
    dr.rounded_rectangle(box, radius=6, fill=bg)
    dr.text((x + pad, cy), text, font=font, fill=fg, anchor="lm")
    return x + tw + pad * 2


def _badge(dr, box, text, bg, fg, font, radius=8, outline=None):
    dr.rounded_rectangle(box, radius=radius, fill=bg, outline=outline, width=1)
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    dr.text((cx, cy), text, font=font, fill=fg, anchor="mm")


def _stars(dr, right_x, cy, n):
    n = int(max(0, min(5, n)))
    f = _font(34, "Regular")
    on = "★" * n
    off = "☆" * (5 - n)
    w_on = _tw(dr, on, f)
    w_off = _tw(dr, off, f)
    x0 = right_x - (w_on + w_off)
    if on:
        dr.text((x0, cy), on, font=f, fill=C["star_on"], anchor="lm")
    if off:
        dr.text((x0 + w_on, cy), off, font=f, fill=C["star_off"], anchor="lm")


def _header(dr, date_label, title, subtitle, confidence):
    dr.rectangle((0, 0, W, 150), fill=C["panel"])
    dr.line((0, 150, W, 150), fill=C["line"], width=2)
    dr.text((44, 30), date_label, font=_font(24, "Regular"), fill=C["t2"], anchor="lm")
    dr.text((44, 70), title, font=_font(42, "Medium"), fill=C["t1"], anchor="lm")
    dr.text((44, 118), subtitle, font=_font(24, "Regular"), fill=C["t2"], anchor="lm")
    dr.text((W - 44, 32), "自信度", font=_font(22, "Regular"), fill=C["t3"], anchor="rm")
    _stars(dr, W - 44, 66, confidence)
    dr.text((W - 44, 116), "AI予測", font=_font(26, "Medium"), fill=C["blue"], anchor="rm")


def _draw_horse_row(dr, y, h, horse, is_hero):
    cy = y + h / 2
    if is_hero:
        dr.rectangle((36, y + 6, W - 36, y + h - 6), fill=C["hero_bg"])
        dr.rectangle((36, y + 6, 42, y + h - 6), fill=C["gold"])
    elif horse.get("mark") == "★":
        dr.rectangle((0, y, W, y + h), fill=C["star_row_bg"])
    dr.line((44, y + h, W - 44, y + h), fill=C["line"], width=1)

    x = 56
    ms = 56 if is_hero else 46
    mk = horse.get("mark", "")
    mbg, mfg = MARK_COLORS.get(mk, ("#30363d", "#f0f6fc"))
    _badge(dr, (x, cy - ms / 2, x + ms, cy + ms / 2), mk, mbg, mfg,
           _font(int(ms * 0.55), "Bold"), radius=10)
    x += ms + 16

    uw, uh = (54, 48) if is_hero else (48, 42)
    wbg, wfg = horse.get("waku_colors", ("#30363d", "#f0f6fc"))
    outline = "#b9c0c7" if wbg.upper() == "#FFFFFF" else ("#555" if wbg == "#2b2b2b" else None)
    _badge(dr, (x, cy - uh / 2, x + uw, cy + uh / 2), str(horse.get("umaban", "")),
           wbg, wfg, _font(int(uh * 0.5), "Bold"), radius=8, outline=outline)
    x += uw + 20

    name_x = x
    fp_right = W - 52
    po_right = fp_right - (150 if is_hero else 118)
    name_max = po_right - name_x - 90

    nf = _font(40 if is_hero else 32, "Medium")
    name = _trunc(dr, horse.get("name", ""), nf, name_max)
    if is_hero:
        dr.text((name_x, cy - 18), name, font=nf, fill=C["t1"], anchor="lm")
        sub_y = cy + 22
    else:
        dr.text((name_x, cy - 16), name, font=nf, fill=C["t1"], anchor="lm")
        sub_y = cy + 20
    if horse.get("anaba"):
        nx = name_x + _tw(dr, name, nf) + 12
        dr.text((nx, cy - (18 if is_hero else 16)), "推奨穴馬",
                font=_font(20, "Regular"), fill=C["purple"], anchor="lm")

    # 騎手 ＋ 根拠タグ
    jf = _font(22 if is_hero else 20, "Regular")
    jx = name_x
    dr.text((jx, sub_y), horse.get("jockey", ""), font=jf, fill=C["t2"], anchor="lm")
    jx += _tw(dr, horse.get("jockey", ""), jf) + 10
    pf = _font(20, "Regular")
    _tag_limit = name_x + name_max + 120   # タグを描ける右端（4個入る余裕）
    _strong_list = horse.get("tag_strong", [])
    for i, tag in enumerate(horse.get("tags", [])[:4]):
        if jx + _tw(dr, tag, pf) + 16 > _tag_limit:
            break
        strong = _strong_list[i] if i < len(_strong_list) else False
        # 寄与率が高い(強)タグは金色で強調・その他はグレー
        bg = C["pill_bg_s"] if strong else C["pill_bg"]
        fg = C["pill_fg_s"] if strong else C["pill_fg"]
        jx = _pill(dr, jx, sub_y, tag, pf, bg, fg) + 7

    # 人気・オッズ
    pop = horse.get("pop", "—")
    odds = horse.get("odds", "—")
    of = _font(22 if is_hero else 20, "Regular")
    dr.text((po_right, cy - 15), pop, font=of, fill=C["trow"], anchor="rm")
    dr.text((po_right, cy + 15), odds, font=of, fill=C["odds"], anchor="rm")

    # AI予測複勝率
    fp = horse.get("fp")
    fp_txt = f"{fp}%" if fp is not None else "—"
    fp_color = C["green"] if (fp is not None and fp >= 30 and horse.get("mark") in ("◎", "○", "▲", "★")) else \
        (C["green"] if (fp is not None and fp >= 40) else C["t2"])
    if is_hero:
        dr.text((fp_right, cy - 20), "AI予測", font=_font(19, "Regular"), fill=C["t3"], anchor="rm")
        dr.text((fp_right, cy + 2), "複勝率", font=_font(19, "Regular"), fill=C["t3"], anchor="rm")
        dr.text((fp_right, cy + 34), fp_txt, font=_font(50, "Bold"), fill=C["green"], anchor="rm")
    else:
        dr.text((fp_right, cy), fp_txt, font=_font(34, "Bold"), fill=fp_color, anchor="rm")


def render_prediction_banner(data) -> bytes:
    img = Image.new("RGB", (W, H), C["bg"])
    dr = ImageDraw.Draw(img)
    _header(dr, data["date"], data["title"], data["subtitle"], data.get("confidence", 0))

    horses = data["horses"]
    top = 170
    foot_h = 66
    bottom = H - foot_h
    avail = bottom - top - 12
    n = len(horses)
    if n <= 0:
        n = 1
    weights = [1.42 if h.get("is_hero") else 1.0 for h in horses]
    tw = sum(weights)
    unit = min(avail / tw, 150)
    used = unit * tw
    y = top + (avail - used) / 2 + 6
    for h, w in zip(horses, weights):
        rh = unit * w
        _draw_horse_row(dr, y, rh, h, h.get("is_hero", False))
        y += rh

    # footer
    dr.rectangle((0, bottom, W, H), fill=C["panel"])
    dr.line((0, bottom, W, bottom), fill=C["line"], width=2)
    dr.text((44, H - foot_h / 2), "◎本命 ○対抗 ▲単穴 ★妙味(穴) △連下　／　AI予測複勝率＝モデル推定の参考値",
            font=_font(20, "Regular"), fill=C["t3"], anchor="lm")

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# カテゴリ別の色（ヘッダ帯の背景・文字・バー色）
_CAT_STYLE = {
    "騎手":  ("#14273d", "#7cb7f5", "#378add"),
    "種牡馬": ("#3a2f10", "#f0c96a", "#d9a441"),
    "調教師": ("#291a3d", "#c9a6e0", "#9b6fc7"),
}
_RANK_BADGE = {1: "#f5c518", 2: "#b8c0c9", 3: "#cd8a4a"}  # 金銀銅


def _rounded_panel(dr, box, fill, radius=18, outline=None, ow=1):
    dr.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=ow)


def render_course_banner(data) -> bytes:
    img = Image.new("RGB", (W, H), C["bg"])
    dr = ImageDraw.Draw(img)

    # ── ヘッダ ──
    dr.rectangle((0, 0, W, 150), fill=C["panel"])
    dr.line((0, 150, W, 150), fill=C["line"], width=2)
    dr.rectangle((0, 0, 8, 150), fill=C["green"])
    dr.text((40, 52), data["header"], font=_font(38, "Medium"), fill=C["t1"], anchor="lm")
    dr.text((40, 104), "出走メンバー限定・勝率トップ3　（数字=勝ち数/母数R ＋ 該当出走馬）",
            font=_font(22, "Regular"), fill=C["t2"], anchor="lm")

    cols = [("騎手", data.get("jockey", [])),
            ("種牡馬", data.get("sire", [])),
            ("調教師", data.get("trainer", []))]
    # バー基準（全体の最大勝率＝フルバー）
    _allp = [e["pct"] for _, es in cols for e in es] or [1]
    _pmax = max(_allp) or 1

    top = 176
    foot_h = 62
    bottom = H - foot_h - 12
    col_w = W / 3
    gap = 12
    for ci, (label, entries) in enumerate(cols):
        hbg, hfg, barc = _CAT_STYLE.get(label, (C["panel"], C["t1"], C["green"]))
        x0 = ci * col_w + gap
        x1 = (ci + 1) * col_w - gap
        # 列パネル（背景で余白を埋める）
        _rounded_panel(dr, (x0, top, x1, bottom), fill="#12181f", outline="#242c36", ow=1)
        pad = 24
        ix0, ix1 = x0 + pad, x1 - pad
        # カテゴリ見出し帯
        _rounded_panel(dr, (ix0, top + 18, ix1, top + 64), fill=hbg, radius=10)
        dr.text(((ix0 + ix1) / 2, top + 41), label, font=_font(26, "Medium"), fill=hfg, anchor="mm")

        ey = top + 84
        eh = (bottom - ey - 8) / 3
        if not entries:
            dr.text(((ix0 + ix1) / 2, ey + eh), "該当データなし",
                    font=_font(20, "Regular"), fill=C["t3"], anchor="mm")
        for ri, e in enumerate(entries[:3], 1):
            cy = ey + eh * (ri - 1)
            # エントリカード（スロットを埋めて余白を無くす）
            _rounded_panel(dr, (ix0, cy + 4, ix1, cy + eh - 12), fill="#182029", radius=14)
            cpad = 16
            cx0, cx1 = ix0 + cpad, ix1 - cpad
            mid = (cy + 4 + cy + eh - 12) / 2
            # 順位バッジ
            _bc = _RANK_BADGE.get(ri, "#5f6b78")
            dr.ellipse((cx0, mid - 42, cx0 + 32, mid - 10), fill=_bc)
            dr.text((cx0 + 16, mid - 26), str(ri), font=_font(18, "Bold"), fill="#1a1a1a", anchor="mm")
            # 馬名（バッジ右）＋ 勝率（右寄せ・大）
            nm = _trunc(dr, e["name"], _font(26, "Medium"), (cx1 - (cx0 + 44)) - 84)
            dr.text((cx0 + 44, mid - 26), nm, font=_font(26, "Medium"), fill=C["t1"], anchor="lm")
            dr.text((cx1, mid - 27), f"{e['pct']}%", font=_font(32, "Bold"), fill=C["green"], anchor="rm")
            # サブ（勝ち数/母数R ＋ 該当馬）
            sub = f"{e['wins']}/{e['n']}R"
            if e.get("horse"):
                sub += f"　{e['horse']}"
            sub = _trunc(dr, sub, _font(19, "Regular"), cx1 - cx0)
            dr.text((cx0, mid + 6), sub, font=_font(19, "Regular"), fill=C["t2"], anchor="lm")
            # 勝率バー（全体最大＝フル）
            bw = cx1 - cx0
            _rounded_panel(dr, (cx0, mid + 30, cx1, mid + 44), fill="#0f151c", radius=7)
            _fillw = max(10, int(bw * e["pct"] / _pmax))
            _rounded_panel(dr, (cx0, mid + 30, cx0 + _fillw, mid + 44), fill=barc, radius=7)

    dr.rectangle((0, H - foot_h, W, H), fill=C["panel"])
    dr.line((0, H - foot_h, W, H - foot_h), fill=C["line"], width=2)
    dr.text((40, H - foot_h / 2), "AI競馬予測ツール　／　過去データに基づく参考情報",
            font=_font(20, "Regular"), fill=C["t3"], anchor="lm")

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
def _fmt_pop_odds(row):
    pop_v = None
    for c in ("人気_live", "人気"):
        v = pd.to_numeric(row.get(c), errors="coerce")
        if pd.notna(v) and v > 0:
            pop_v = int(v)
            break
    odds_v = pd.to_numeric(row.get("単勝オッズ_live"), errors="coerce")
    pop = f"{pop_v}人気" if pop_v else "—"
    odds = f"{odds_v:.1f}倍" if pd.notna(odds_v) and odds_v > 0 else "—"
    return pop, odds


def build_banner_data(show_df, *, venue_abbr, venue_full, is_turf, dist,
                      date_label, title, subtitle, confidence, n_horses):
    """show_df（marks/_fuku_prob等を持つ）とレースメタから2枚分のデータを組む。"""
    from pred_utils import get_reasons
    try:
        from pred_utils import reason_strength
    except Exception:
        reason_strength = None
    from course_stats import course_top_performers

    df = show_df.sort_values("pred_rank").copy()
    marked = df[df.get("_mark", pd.Series("", index=df.index)).astype(str) != ""]
    if marked.empty:
        marked = df.head(6)

    horses = []
    for _, row in marked.iterrows():
        mark = str(row.get("_mark", "") or "")
        try:
            ub = int(pd.to_numeric(row.get("馬番"), errors="coerce"))
        except Exception:
            ub = 0
        waku = _umaban_to_waku(ub, n_horses) if ub else 0
        reasons = get_reasons(row, df, top_n=4) or []
        strong = []
        for r in reasons:
            strong.append(bool(reason_strength(r) >= 3) if reason_strength else False)
        pop, odds = _fmt_pop_odds(row)
        fp_v = pd.to_numeric(row.get("_fuku_prob"), errors="coerce")
        horses.append({
            "mark": mark, "umaban": ub,
            "waku_colors": _WAKU_COLORS.get(waku, ("#30363d", "#f0f6fc")),
            "name": str(row.get("馬名", "")), "jockey": str(row.get("騎手", "") or ""),
            "tags": [_short_tag(r) for r in reasons], "tag_strong": strong,
            "pop": pop, "odds": odds,
            "fp": int(round(fp_v * 100)) if pd.notna(fp_v) else None,
            "anaba": mark == "★",
            "is_hero": mark == "◎",
        })
    # ◎が無い場合、先頭をheroに
    if horses and not any(h["is_hero"] for h in horses):
        horses[0]["is_hero"] = True

    banner1 = {"date": date_label, "title": title, "subtitle": subtitle,
               "confidence": confidence, "horses": horses}

    # コースTOP3
    surf = "芝" if is_turf else "ダ"
    course = course_top_performers(df, venue_abbr, is_turf, int(dist) if dist else 0,
                                   top_n=3, min_n=5)

    def _conv(rows):
        out = []
        for r in rows or []:
            out.append({"name": str(r["name"]), "pct": round(r["rate"] * 100),
                        "wins": round(r["rate"] * r["n"]), "n": int(r["n"]),
                        "horse": (r["horses"][0] if r.get("horses") else "")})
        return out

    banner2 = {
        "header": f"{venue_full} {surf}{int(dist) if dist else ''}m　このコースの好調データ",
        "jockey": _conv(course.get("騎手", [])),
        "sire": _conv(course.get("種牡馬", [])),
        "trainer": _conv(course.get("調教師", [])),
    }
    return banner1, banner2


def make_banners(show_df, **meta):
    b1, b2 = build_banner_data(show_df, **meta)
    return render_prediction_banner(b1), render_course_banner(b2)
