# -*- coding: utf-8 -*-
"""
日付パースの共通ヘルパ。

このプロジェクトには日付の慣習が2系統ある:
  - ライブ出馬表 : 8桁 'YYYYMMDD'（例 20260906）
  - master       : 6桁 'YYMMDD' （例 260906）

各所で `format='%Y%m%d'` / `format='%y%m%d'` をハードコードしていたため、
想定外の桁数が流れ込むと pd.to_datetime が黙って NaT を返し、5年フィルタや
特徴量結合から行が脱落する潜在バグがあった（過去の「0026年」誤読と同型）。

to_dt() は桁数を自動判別して吸収する。混在した Series でも行ごとに正しく処理する。
"""
import pandas as pd


def to_dt(s) -> pd.Series:
    """6桁YYMMDD / 8桁YYYYMMDD が混在していても正しく datetime へ変換する。

    - 数字以外を除去した上で、8桁は %Y%m%d、6桁は %y%m%d として解釈。
    - それ以外（曖昧・無効）は NaT。あえて汎用パーサに頼らない
      （'2026. 9.6' 等を dateutil が“それらしい誤った日付”に化かすのを防ぐ＝
        静かな誤りより見える失敗(NaT)の方が安全）。区切り付き表記は呼び出し側で
      正規化してから渡すこと（weekly_update.ensure_date_col など）。
    """
    s = pd.Series(s)
    digits = s.astype(str).str.strip().str.replace(r"\D", "", regex=True)

    out = pd.to_datetime(digits.where(digits.str.len() == 8),
                         format="%Y%m%d", errors="coerce")
    six = digits.str.len() == 6
    if six.any():
        out = out.fillna(pd.to_datetime(digits.where(six),
                                        format="%y%m%d", errors="coerce"))
    return out


def to_yymmdd(s) -> pd.Series:
    """任意の日付表記を6桁 'YYMMDD' 文字列へ正規化（master形式）。変換不能は空文字。"""
    dt = to_dt(s)
    return dt.dt.strftime("%y%m%d").fillna("")
