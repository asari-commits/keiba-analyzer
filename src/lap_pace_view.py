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
import re
import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path

_JOIN = Path(__file__).resolve().parent.parent / "data" / "processed" / "lap_master_joined.parquet"
# 条件別ペース傾向（ラップCSV全期間=10年で集計）。masterの5年制限を受けないので
# 年1回の重賞でも n≈10 を確保できる。無ければ join から5年分で代替生成する。
_TPROF = Path(__file__).resolve().parent.parent / "data" / "processed" / "lap_tendency_profile.parquet"
_STATE: dict = {"loaded": False, "j": None, "prof": None}

# 縮小推定の強さ（大きいほど上位階層＝コース平均に寄る）。
# 傾向は10年分で集計するため、年1回の重賞でも n≈10 となりレース別が十分効く。
# n が小さいレースは自動的に控えめになる（ハードな足切りはしない）。
_K_CLASS = 10.0
_K_RACE = 6.0
# 「採用条件」としてボックスに表示する最小n（推定自体はnに関わらず常に反映）
_MIN_SHOW_CLASS = 5
_MIN_SHOW_RACE = 3
_CLS_NAME = {0: "新馬", 1: "未勝利", 2: "1勝", 3: "2勝", 4: "3勝",
             5: "オープン", 6: "OP(L)", 7: "G3", 8: "G2", 9: "G1"}
# レース別プロファイルから除外する汎用クラス名（＝クラス条件付けで既にカバー済み）
_GENERIC_RN = {"新馬", "未勝利", "1勝", "2勝", "3勝", "1勝クラス", "2勝クラス",
               "3勝クラス", "オープン", "ｵｰﾌﾟﾝ", "", "nan", "None"}


def _norm_rn(s):
    """レース名を正規化してmaster/pred_df間の表記差を吸収。
    例: '関屋記念ＨG3'/'関屋記念(GIII)' → '関屋記念' 。"""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"第\d+回", "", s)
    s = re.sub(r"[（(][^）)]*[）)]$", "", s)          # 末尾の(...)（グレード等）
    s = re.sub(r"(HG[123]|JG[123]|G[123]|GI{1,3}|OP|L|H)+$", "", s)
    s = re.sub(r"[\s　]+", "", s)
    return s.strip()


def _row_prof(r):
    """傾向プロファイルの1行 → 統計dict。"""
    return {"n": int(r["n"]), "z": float(r["z"]), "f3": float(r["f3"]),
            "l3": float(r["l3"]), "diff": float(r["diff"]), "rpci": float(r["rpci"])}


def _agg_prof(grp):
    return {
        "n": int(len(grp)),
        "z": float(grp["pace_z"].mean()),
        "f3": float(pd.to_numeric(grp["lap_前3F"], errors="coerce").mean()),
        "l3": float(pd.to_numeric(grp["lap_後3F"], errors="coerce").mean()),
        "diff": float(pd.to_numeric(grp["lap_後半3F差"], errors="coerce").mean()),
        "rpci": float(pd.to_numeric(grp["lap_RPCI"], errors="coerce").mean()),
    }


def _load():
    if _STATE["loaded"]:
        return _STATE["j"], _STATE["prof"]
    _STATE["loaded"] = True
    try:
        if not _JOIN.exists():
            return None, None
        cols = ["馬名", "着順_num", "距離", "芝・ダ", "lap_venue", "lap_後半3F差",
                "日付_dt", "人気", "日付", "開催", "Ｒ", "クラス_num", "レース名",
                "lap_前3F", "lap_後3F", "lap_RPCI"]
        avail = None
        try:
            import pyarrow.parquet as _pq
            avail = set(_pq.ParquetFile(_JOIN).schema.names)
        except Exception:
            avail = None
        use = [c for c in cols if avail is None or c in avail]
        j = pd.read_parquet(_JOIN, columns=use)
        j["着順_num"] = pd.to_numeric(j["着順_num"], errors="coerce")
        j["lap_後半3F差"] = pd.to_numeric(j["lap_後半3F差"], errors="coerce")
        j = j[j["着順_num"].notna() & j["lap_後半3F差"].notna()].copy()
        j["good"] = (j["着順_num"] <= 3).astype(int)
        j["distbin"] = (pd.to_numeric(j["距離"], errors="coerce") // 200 * 200)

        # 10年分の傾向プロファイル（あれば優先）
        tprof = None
        if _TPROF.exists():
            try:
                tprof = pd.read_parquet(_TPROF)
            except Exception:
                tprof = None

        # 各馬の履歴z: 傾向プロファイルと同じ標準化基準を使う（無ければ自前で標準化）
        norm = None
        if tprof is not None and "norm" in set(tprof["level"]):
            nm = tprof[tprof["level"] == "norm"]
            norm = {(r["TD"], r["distbin"]): (r["mu"], r["sd"]) for _, r in nm.iterrows()}
        if norm:
            _mu = j.set_index(["芝・ダ", "distbin"]).index.map(
                lambda k: norm.get(k, (np.nan, np.nan))[0])
            _sd = j.set_index(["芝・ダ", "distbin"]).index.map(
                lambda k: norm.get(k, (np.nan, np.nan))[1])
            j["pace_z"] = (j["lap_後半3F差"].values - np.asarray(_mu, dtype=float)) / \
                          np.asarray(_sd, dtype=float)
        else:
            g = j.groupby(["芝・ダ", "distbin"])["lap_後半3F差"]
            j["pace_z"] = (j["lap_後半3F差"] - g.transform("mean")) / g.transform("std")
        j = j.dropna(subset=["pace_z"])

        # 傾向プロファイルが使えるならそれを採用して以降の自前集計をスキップ
        if tprof is not None and {"course", "class", "race"} & set(tprof["level"]):
            course, klass, race = {}, {}, {}
            for _, r in tprof.iterrows():
                lv = r["level"]
                if lv == "course":
                    course[(r["venue"], r["TD"], r["distbin"])] = _row_prof(r)
                elif lv == "class" and pd.notna(r["class_num"]):
                    klass[(r["venue"], r["TD"], r["distbin"], int(r["class_num"]))] = _row_prof(r)
                elif lv == "race" and r["race"]:
                    d = _row_prof(r)
                    d["venue"], d["surf"], d["distbin"] = r["venue"], r["TD"], r["distbin"]
                    race[str(r["race"])] = d
            prof = {"course": course, "klass": klass, "race": race}
            _STATE["j"], _STATE["prof"] = j, prof
            return j, prof

        # レース単位に一意化してプロファイルを集計（頭数で重み付かないように）
        has_class = "クラス_num" in j.columns
        has_rn = "レース名" in j.columns
        rk = [c for c in ["日付", "開催", "Ｒ"] if c in j.columns]
        races = j.drop_duplicates(rk).copy() if rk else j.copy()

        course, klass, race = {}, {}, {}
        for key, grp in races.groupby(["lap_venue", "芝・ダ", "distbin"]):
            course[key] = _agg_prof(grp)
        if has_class:
            races["クラス_num"] = pd.to_numeric(races["クラス_num"], errors="coerce")
            for key, grp in races.dropna(subset=["クラス_num"]).groupby(
                    ["lap_venue", "芝・ダ", "distbin", "クラス_num"]):
                klass[(key[0], key[1], key[2], int(key[3]))] = _agg_prof(grp)
        if has_rn:
            races["_rn_norm"] = races["レース名"].map(_norm_rn)
            for rn, grp in races.groupby("_rn_norm"):
                if rn in _GENERIC_RN or len(grp) < 5:
                    continue
                _vc = grp.groupby(["lap_venue", "芝・ダ", "distbin"]).size().idxmax()
                d = _agg_prof(grp)
                d["venue"], d["surf"], d["distbin"] = _vc
                race[rn] = d

        prof = {"course": course, "klass": klass, "race": race}
        _STATE["j"], _STATE["prof"] = j, prof
        return j, prof
    except Exception:
        return None, None


def _resolve_pace(prof, venue, surf, distbin, class_num, race_name):
    """条件別ペース傾向を縮小推定で解決。
    戻り: (race_z=推定pace_z, disp=(level, stats, label)) / データ無ければ (None, None)。
    レース名別→クラス×コース別→コース別 の順に、標本が十分なら具体的にする。
    """
    course = prof["course"].get((venue, surf, distbin))
    if not course:
        return None, None
    est = course["z"]
    disp = ("course", course, "全クラス平均")
    if class_num is not None and not pd.isna(class_num):
        c = prof["klass"].get((venue, surf, distbin, int(class_num)))
        if c:
            est = (c["n"] * c["z"] + _K_CLASS * est) / (c["n"] + _K_CLASS)
            if c["n"] >= _MIN_SHOW_CLASS:
                disp = ("class", c, _CLS_NAME.get(int(class_num), f"クラス{int(class_num)}"))
    if race_name:
        key = _norm_rn(race_name)
        r = prof["race"].get(key)
        # フォールバック: 冗長なレース名(例 '関屋記念(GⅢ)サラ系３歳以上') に対し、
        # 同コースのプロファイル名が含まれていれば最長一致を採用。
        if not (r and (r["venue"], r["surf"], r["distbin"]) == (venue, surf, distbin)):
            tgt = unicodedata.normalize("NFKC", str(race_name))
            cands = [(k, v) for k, v in prof["race"].items()
                     if len(k) >= 3 and k in tgt
                     and (v["venue"], v["surf"], v["distbin"]) == (venue, surf, distbin)]
            if cands:
                key, r = max(cands, key=lambda kv: len(kv[0]))
        if r and (r["venue"], r["surf"], r["distbin"]) == (venue, surf, distbin):
            est = (r["n"] * r["z"] + _K_RACE * est) / (r["n"] + _K_RACE)
            if r["n"] >= _MIN_SHOW_RACE:
                disp = ("race", r, key)
    return est, disp


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
.lpv-course-cls{display:inline-block;font-weight:700;font-size:11.5px;padding:1px 8px;margin:0 2px;
  border-radius:999px;background:rgba(128,140,132,.18);border:1px solid rgba(128,140,132,.45)}
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


_LVL_TAG = {"race": "レース別", "class": "クラス別", "course": "全クラス平均"}


def _course_box_html(venue_full: str, is_turf: bool, dist, disp) -> str:
    """条件別（レース/クラス/コース）の平均ラップ・ペース傾向ボックス。無ければ ''。"""
    if not disp:
        return ""
    level, s, label = disp
    surf = "芝" if is_turf else "ダート"
    diff = s["diff"]
    ptype = ("後傾(瞬発寄り)" if diff < -0.3 else ("前傾(消耗寄り)" if diff > 0.3 else "中間"))
    col = "#cf6f33" if "前傾" in ptype else ("#3f86c9" if "後傾" in ptype else "#8b968e")
    try:
        dist_i = int(float(dist))
    except Exception:
        dist_i = dist
    cls_chip = (f'<span class="lpv-course-cls">{_esc(label)}</span>'
                if level != "course" else '')
    return ('<div class="lpv-course">'
            f'<div class="lpv-course-h">▷ {_esc(venue_full)}・{surf}{dist_i}m {cls_chip}'
            f'<span class="lpv-course-n">（{_LVL_TAG[level]}・過去{s["n"]}戦）</span></div>'
            '<div class="lpv-course-b">'
            f'<span>前半3F <b>{s["f3"]:.1f}</b></span>'
            f'<span>後半3F <b>{s["l3"]:.1f}</b></span>'
            f'<span>後半3F差 <b>{diff:+.1f}</b></span>'
            f'<span>RPCI <b>{s["rpci"]:.1f}</b></span>'
            f'<span class="lpv-course-t" style="color:{col};border-color:{col}">{ptype}</span>'
            '</div></div>')


def _pace_word(z):
    if z is None:
        return None
    if z <= -0.3:
        return ("後傾・スロー（瞬発戦になりやすい）", "#5aa2e0")
    if z >= 0.3:
        return ("前傾・ハイペース（消耗戦になりやすい）", "#e08a52")
    return ("平均的（どちらにも振れる）", "#9aa39c")


def render_race_pace_html(names, venue_token: str, is_turf: bool, dist, numbers=None,
                          venue_full="", class_num=None, race_name=None) -> str:
    """出走各馬のペース適性セクションのHTMLを返す。データが無ければ ''。
    numbers: {馬名: 馬番} を渡すと馬番チップを表示。
    class_num/race_name: 今回レースのクラス(0-9)/レース名。想定ペースを
      レース別→クラス×コース別→コース別 の縮小推定で条件付けする。
    並び順は「今回の想定ペースへの適合度が高い順」（◎合う→中間→▲逆ペース）。
    """
    j, prof = _load()
    if j is None:
        return ""
    try:
        dist = int(float(dist))
    except Exception:
        return ""
    surf = "芝" if is_turf else "ダ"
    distbin = dist // 200 * 200
    numbers = numbers or {}

    # 想定ペース（条件別・縮小推定）
    race_z, disp = (None, None)
    if prof is not None:
        race_z, disp = _resolve_pace(prof, venue_token, surf, distbin, class_num, race_name)

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

    # ヘッダ: 条件別ラップ・ペース傾向（無ければ想定ペースの語のみ）
    head = _course_box_html(venue_full, is_turf, dist, disp)
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
