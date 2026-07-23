# -*- coding: utf-8 -*-
"""
出走表用「各馬の得意ペース（ラップ適性）」ビュー。

data/processed/lap_master_joined.parquet（ラップCSVをmasterに指紋キー結合したもの）から、
各馬の過去レースを実現ペースz（後半3F差を芝ダ×距離帯で標準化）で並べ、
好走（3着内）レースがどのペース帯に集まるかで得意ペースを推定して可視化する。

  pace_z < 0 = 後傾/スロー/瞬発戦（末脚型）が得意
  pace_z > 0 = 前傾/ハイペース/消耗戦（押し切り型）が得意

想定ペース（そのコース・距離の平均的な相対ペース）を軸上に重ね、各馬の得意ペースとの
相性を示す。lap_master_joined.parquet が無い環境では空文字を返し、呼び出し側で非表示にする。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

_JOIN = Path(__file__).resolve().parent.parent / "data" / "processed" / "lap_master_joined.parquet"
_COURSE_PROFILE = Path(__file__).resolve().parent.parent / "data" / "processed" / "lap_course_profile.parquet"
_STATE: dict = {"loaded": False, "j": None, "course": None}
_CSTATE: dict = {"loaded": False, "df": None}


def _load_course_profile():
    """コース別ラップ傾向テーブル(lap_course_profile.parquet)をロード。"""
    if _CSTATE["loaded"]:
        return _CSTATE["df"]
    _CSTATE["loaded"] = True
    try:
        if _COURSE_PROFILE.exists():
            _CSTATE["df"] = pd.read_parquet(_COURSE_PROFILE)
    except Exception:
        _CSTATE["df"] = None
    return _CSTATE["df"]


def course_profile_summary(venue_token: str, is_turf: bool, dist):
    """コースの平均ラップ・ペース傾向。venue×芝ダ×距離で lap_course_profile を引く。"""
    df = _load_course_profile()
    if df is None or df.empty:
        return None
    try:
        dist = int(float(dist))
    except Exception:
        return None
    surf = "芝" if is_turf else "ダ"
    sub = df[(df["venue"] == venue_token) & (df["TD"] == surf)]
    if sub.empty:
        return None
    exact = sub[sub["距離"] == dist]
    if not exact.empty:
        r = exact.iloc[0]
    else:  # 同コースで最も近い距離にフォールバック
        r = sub.iloc[(sub["距離"] - dist).abs().values.argmin()]
    return {
        "n": int(r["n"]), "f3": float(r["前3F平均"]), "l3": float(r["後3F平均"]),
        "diff": float(r["後半3F差平均"]), "rpci": float(r["RPCI平均"]),
        "ptype": str(r["pace_type"]), "dist": int(r["距離"]),
        "exact": (not exact.empty),
    }


def _load():
    if _STATE["loaded"]:
        return _STATE["j"], _STATE["course"]
    _STATE["loaded"] = True
    try:
        if not _JOIN.exists():
            return None, None
        cols = ["馬名", "着順_num", "距離", "芝・ダ", "lap_venue", "lap_後半3F差", "日付_dt", "人気"]
        j = pd.read_parquet(_JOIN, columns=cols)
        j["着順_num"] = pd.to_numeric(j["着順_num"], errors="coerce")
        j["lap_後半3F差"] = pd.to_numeric(j["lap_後半3F差"], errors="coerce")
        j = j[j["着順_num"].notna() & j["lap_後半3F差"].notna()].copy()
        j["good"] = (j["着順_num"] <= 3).astype(int)
        j["distbin"] = (pd.to_numeric(j["距離"], errors="coerce") // 200 * 200)
        g = j.groupby(["芝・ダ", "distbin"])["lap_後半3F差"]
        j["pace_z"] = (j["lap_後半3F差"] - g.transform("mean")) / g.transform("std")
        j = j.dropna(subset=["pace_z"])
        course = j.groupby(["lap_venue", "芝・ダ", "distbin"])["pace_z"].mean()
        _STATE["j"], _STATE["course"] = j, course
        return j, course
    except Exception:
        return None, None


def _clampx(z: float) -> float:
    return max(0.0, min(100.0, (max(-2.5, min(2.5, z)) + 2.5) / 5.0 * 100.0))


def _classify(pref, ngood, pstd):
    if ngood < 2 or (pstd is not None and pstd > 1.2):
        return "unknown", "傾向不明"
    if pref >= 0.4:
        return "high", "ハイペース型"
    if pref <= -0.4:
        return "slow", "瞬発型"
    return "flex", "自在"


def _profile(j: pd.DataFrame, name: str):
    h = j[j["馬名"] == name]
    if len(h) == 0:
        return None
    h = h.sort_values("日付_dt")
    gd = h.loc[h["good"] == 1, "pace_z"]
    pref = float(gd.mean()) if len(gd) else float("nan")
    pstd = float(gd.std()) if len(gd) >= 2 else None
    vc, vt = _classify(pref, len(gd), pstd)
    return {
        "name": name, "n": int(len(h)), "ngood": int(h["good"].sum()),
        "pref": None if np.isnan(pref) else round(pref, 2),
        "gsd": None if pstd is None else round(pstd, 2),
        "gz": [round(float(z), 2) for z in gd],   # 好走レースのpace_zのみ
        "vc": vc, "vt": vt,
    }


# 明暗どちらのテーマでも読める中トーンの色（Streamlitのlight/darkに追随）
_BADGE = {
    "high": ("#cf6f33", "rgba(207,111,51,.15)"),
    "slow": ("#3f86c9", "rgba(63,134,201,.15)"),
    "flex": ("#6f8378", "rgba(111,131,120,.15)"),
    "unknown": ("#808a84", "rgba(128,138,132,.12)"),
}
_GOOD = "#2f9e6f"      # 好走ドット
_C_COOL = "#3f86c9"    # 瞬発側
_C_WARM = "#cf6f33"    # ハイペース側

_CSS = """
<style>
.lpv-wrap{font-size:13px;}                       /* 文字色はStreamlitテーマを継承 */
.lpv-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:2px 0 12px;
  padding:9px 12px;background:rgba(128,140,132,.10);border:1px solid rgba(128,140,132,.35);border-radius:9px;}
.lpv-head b{font-weight:700}
.lpv-axis{display:flex;justify-content:space-between;font-size:11px;color:#8b968e;margin:0 0 6px;}
.lpv-axis .l{color:#3f86c9}.lpv-axis .r{color:#cf6f33}
.lpv-row{display:grid;grid-template-columns:150px 1fr 96px;gap:10px;align-items:center;
  padding:6px 0;border-top:1px solid rgba(128,140,132,.25);}
.lpv-row:first-of-type{border-top:0}
.lpv-nm{font-weight:600;line-height:1.25;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lpv-num{display:inline-block;min-width:18px;text-align:center;font-size:11px;font-weight:700;
  padding:0 4px;margin-right:5px;border-radius:4px;border:1px solid rgba(128,140,132,.5);opacity:.85}
.lpv-bd{display:inline-block;font-size:10px;font-weight:700;padding:1px 6px;border-radius:999px;margin-top:2px}
.lpv-strip{position:relative;height:30px;border-radius:7px;border:1px solid rgba(128,140,132,.4);overflow:hidden;
  background:linear-gradient(90deg,rgba(63,134,201,.20),transparent 40%,transparent 60%,rgba(207,111,51,.20));}
.lpv-band{position:absolute;top:4px;bottom:4px;border-radius:5px;border:1px solid;opacity:.55}
.lpv-mid{position:absolute;left:50%;top:0;bottom:0;width:1px;background:currentColor;opacity:.16}
.lpv-race{position:absolute;top:0;bottom:0;width:0;border-left:2px dashed currentColor;opacity:.6}
.lpv-race::before{content:"想定";position:absolute;top:1px;left:3px;font-size:9px;opacity:.7;white-space:nowrap}
.lpv-dot{position:absolute;transform:translate(-50%,-50%);border-radius:50%}
.lpv-dot.g{width:8px;height:8px;background:#2f9e6f;box-shadow:0 0 0 1.5px rgba(255,255,255,.35)}
.lpv-pf{position:absolute;top:0;bottom:0;width:2px;transform:translateX(-50%)}
.lpv-pf::after{content:"";position:absolute;top:-1px;left:50%;transform:translateX(-50%);
  border-left:5px solid transparent;border-right:5px solid transparent;border-top:6px solid currentColor}
.lpv-fit{font-size:12px;font-weight:700;text-align:right}
.lpv-keys{font-size:11px;color:#8b968e;margin-top:10px;display:flex;gap:16px;flex-wrap:wrap}
.lpv-keys i{display:inline-block;width:9px;height:9px;border-radius:50%;vertical-align:middle;margin-right:5px}
.lpv-course{margin:2px 0 12px;padding:10px 12px;background:rgba(128,140,132,.10);
  border:1px solid rgba(128,140,132,.35);border-radius:9px}
.lpv-course-h{font-weight:700;font-size:13px;margin-bottom:6px}
.lpv-course-n{font-weight:400;color:#8b968e;font-size:11px;margin-left:4px}
.lpv-course-b{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12.5px;align-items:center}
.lpv-course-b b{font-variant-numeric:tabular-nums}
.lpv-course-t{font-weight:700;padding:1px 9px;border:1px solid;border-radius:999px;font-size:11.5px}
/* スマホ: 表示幅を詰めて横はみ出しを防ぐ */
@media (max-width:640px){
  .lpv-wrap{font-size:12px}
  .lpv-row{grid-template-columns:88px 1fr 52px;gap:6px}
  .lpv-nm{font-size:11.5px}
  .lpv-num{min-width:15px;font-size:10px;margin-right:3px;padding:0 2px}
  .lpv-bd{font-size:9px}
  .lpv-strip{height:26px}
  .lpv-fit{font-size:11px}
  .lpv-flbl{display:none}
  .lpv-axis{font-size:9.5px}
  .lpv-course-b{font-size:11px;gap:4px 10px}
  .lpv-course-h{font-size:12px}
  .lpv-keys{font-size:9.5px;gap:8px}
}
</style>
"""


def _course_box_html(venue_full: str, venue_token: str, is_turf: bool, dist) -> str:
    """コース平均ラップ・ペース傾向の見出しボックス。データが無ければ ''。"""
    s = course_profile_summary(venue_token, is_turf, dist)
    if not s:
        return ""
    surf = "芝" if is_turf else "ダート"
    col = ("#cf6f33" if "前傾" in s["ptype"]
           else ("#3f86c9" if "後傾" in s["ptype"] else "#8b968e"))
    approx = "" if s["exact"] else " (近似距離)"
    vlabel = venue_full or venue_token
    return ('<div class="lpv-course">'
            f'<div class="lpv-course-h">▷ {_esc(vlabel)}・{surf}{s["dist"]}m{approx} '
            f'平均ラップ・ペース傾向<span class="lpv-course-n">（過去{s["n"]}戦）</span></div>'
            '<div class="lpv-course-b">'
            f'<span>前半3F <b>{s["f3"]:.1f}</b></span>'
            f'<span>後半3F <b>{s["l3"]:.1f}</b></span>'
            f'<span>後半3F差 <b>{s["diff"]:+.1f}</b></span>'
            f'<span>RPCI <b>{s["rpci"]:.1f}</b></span>'
            f'<span class="lpv-course-t" style="color:{col};border-color:{col}">{s["ptype"]}</span>'
            '</div></div>')


def _pace_word(z):
    if z is None:
        return None
    if z <= -0.3:
        return ("後傾・スロー（瞬発戦になりやすい）", "#5aa2e0")
    if z >= 0.3:
        return ("前傾・ハイペース（消耗戦になりやすい）", "#e08a52")
    return ("平均的（どちらにも振れる）", "#9aa39c")


def render_race_pace_html(names, venue_token: str, is_turf: bool, dist, numbers=None, venue_full="") -> str:
    """出走各馬のペース適性セクションのHTMLを返す。データが無ければ ''。
    numbers: {馬名: 馬番} を渡すと馬番チップを表示。
    並び順は「今回の想定ペースへの適合度が高い順」（◎合う→中間→▲逆ペース）。
    """
    j, course = _load()
    if j is None:
        return ""
    try:
        dist = int(float(dist))
    except Exception:
        return ""
    surf = "芝" if is_turf else "ダ"
    distbin = dist // 200 * 200
    numbers = numbers or {}

    # 想定ペース（コース平均の相対z）
    race_z = None
    if course is not None:
        try:
            race_z = float(course.loc[(venue_token, surf, float(distbin))])
        except Exception:
            race_z = None

    profs = []
    for nm in names:
        p = _profile(j, str(nm))
        if p:
            profs.append(p)
    if not profs:
        return ""

    # 適合度で降順ソート（◎合う=上, ▲逆ペース=下, 中間/不明=真ん中）
    def _fit(p):
        if race_z is None or p["pref"] is None or p["vc"] == "unknown":
            return 0.0
        return p["pref"] * (1.0 if race_z > 0 else -1.0)
    profs.sort(key=_fit, reverse=True)

    # ヘッダ: コース平均ラップ・ペース傾向（無ければ想定ペースの語のみ）
    head = _course_box_html(venue_full, venue_token, is_turf, dist)
    if not head:
        pw = _pace_word(race_z)
        if pw:
            head = (f'<div class="lpv-head">🏇 <b>各馬の得意ペース</b>'
                    f'<span>｜このコース({venue_token}・{surf}{dist}m)の想定ペース: '
                    f'<b style="color:{pw[1]}">{pw[0]}</b></span></div>')
        else:
            head = ('<div class="lpv-head">🏇 <b>各馬の得意ペース</b>'
                    '<span>｜このコースの平均ラップは標本不足のため表示できません</span></div>')

    rows = []
    for p in profs:
        col, bg = _BADGE[p["vc"]]
        # 得意ゾーン帯（好走の平均±SD）: 傾向を面で示す
        band = ""
        if p["pref"] is not None and p["gsd"] is not None and p["vc"] in ("high", "slow", "flex"):
            lo = _clampx(p["pref"] - p["gsd"])
            hi = _clampx(p["pref"] + p["gsd"])
            band = (f'<div class="lpv-band" style="left:{lo:.1f}%;width:{max(hi - lo, 2):.1f}%;'
                    f'background:{bg};border-color:{col}"></div>')
        # 好走レースのみ小ドット（凡走ドットは廃止してノイズ削減）
        dots = []
        gz = p["gz"]
        for i, z in enumerate(gz):
            y = 50 + ((i % 3) - 1) * 15
            dots.append(f'<span class="lpv-dot g" style="left:{_clampx(z):.1f}%;top:{y}%"></span>')
        race_mark = (f'<div class="lpv-race" style="left:{_clampx(race_z):.1f}%"></div>'
                     if race_z is not None else "")
        pref_mark = ""
        if p["pref"] is not None and p["vc"] in ("high", "slow", "flex"):
            pref_mark = (f'<div class="lpv-pf" style="left:{_clampx(p["pref"]):.1f}%;'
                         f'color:{col};background:{col}"></div>')  # colored marker

        # 相性判定（明確なタイプ × 想定ペース）。スマホは記号のみ表示(ラベルはCSSで非表示)
        fit_html = '<span style="color:#8b968e">—</span>'
        if race_z is not None and p["pref"] is not None and p["vc"] in ("high", "slow"):
            same = (np.sign(p["pref"]) == np.sign(race_z)) and abs(race_z) >= 0.12
            if abs(race_z) < 0.12:
                fit_html = '<span style="color:#8b968e">中間</span>'
            elif same:
                fit_html = ('<span style="color:#2f9e6f"><span class="lpv-fsym">◎</span>'
                            '<span class="lpv-flbl"> 合う</span></span>')
            else:
                fit_html = ('<span style="color:#cf6f33"><span class="lpv-fsym">▲</span>'
                            '<span class="lpv-flbl"> 逆ペース</span></span>')

        num = numbers.get(p["name"])
        num_chip = ""
        if num is not None and str(num) not in ("", "nan"):
            try:
                num_chip = f'<span class="lpv-num">{int(float(num))}</span>'
            except Exception:
                num_chip = ""
        rows.append(
            f'<div class="lpv-row">'
            f'<div><div class="lpv-nm">{num_chip}{_esc(p["name"])}</div>'
            f'<span class="lpv-bd" style="color:{col};background:{bg}">{p["vt"]}'
            f'（{p["ngood"]}/{p["n"]}好走）</span></div>'
            f'<div class="lpv-strip">{band}<div class="lpv-mid"></div>{race_mark}{pref_mark}{"".join(dots)}</div>'
            f'<div class="lpv-fit">{fit_html}</div>'
            f'</div>'
        )

    axis = ('<div class="lpv-axis"><span class="l">◀ スロー・瞬発（末脚）</span>'
            '<span>中間</span><span class="r">ハイペース・消耗（押し切り）▶</span></div>')
    keys = ('<div class="lpv-keys">'
            '<span><i style="background:#2f9e6f"></i>好走(3着内)レース</span>'
            '<span>▨ 得意ゾーン(好走の集中域)</span>'
            '<span>▽ 得意ペース　┊ 想定ペース(破線)</span>'
            '<span>相性は明確なタイプのみ判定</span></div>')

    note = ('<div style="font-size:11px;color:#8b968e;margin-top:8px">'
            '※好走レースのラップ形状からの傾向で、着順予測ではありません。凡走レースは表示省略。'
            '好走が少ない/ばらつく馬は「傾向不明」。</div>')

    return _CSS + '<div class="lpv-wrap">' + head + axis + "".join(rows) + keys + note + '</div>'


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))
