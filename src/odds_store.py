"""
過去オッズ・払戻データ（Targetエクスポート43列CSV）を正規化して永続ストア化する。

用途:
  - 実配当ベースのROIバックテスト（roi_backtest.py）
  - 結果回顧・回収率トラッキングを実オッズ/配当で補完（netkeibaスクレイプの代替）

入力CSV（cp932, 43列）の要点:
  - 日付は 'yyyy. m. d'（空白入り）→ 6桁 YYMMDD に正規化
  - 単勝配当は勝ち馬のみ実配当。非勝ち馬は '(オッズ)' 括弧付き参照値 → NaN 扱い
  - 複勝配当は3着以内の各馬。券種払戻(馬連/馬単/三連複/三連単)は該当行に格納
  - (日付,開催,馬番) は非ユニーク（同一開催日12R×馬番で重複）。
    レースID(新)[:16] の下2桁=R, [8:10]=場コード。→ (日付,開催,R,馬番) で一意化
"""
from pathlib import Path
import numpy as np
import pandas as pd

ODDS_STORE_PATH = Path(__file__).parent.parent / "data" / "processed" / "odds_store.parquet"
RACE_STORE_PATH = Path(__file__).parent.parent / "data" / "processed" / "odds_race_store.parquet"

_ZEN = str.maketrans('０１２３４５６７８９', '0123456789')


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(',', '', regex=False).str.strip(),
                         errors='coerce')


def _paynum(s: pd.Series) -> pd.Series:
    """括弧付き(=参照オッズ、配当でない)は NaN にして数値化。"""
    x = s.astype(str).str.strip()
    return pd.to_numeric(x.where(~x.str.startswith('('), np.nan)
                          .str.replace(',', '', regex=False), errors='coerce')


def build_odds_store(csv_path, encoding: str = 'cp932'):
    """Targetオッズ払戻CSVを正規化し、馬単位ストアとレース単位払戻ストアを保存する。

    返り値: (horse_df, race_df)
    """
    o = pd.read_csv(csv_path, encoding=encoding, dtype=str)

    # 日付 'yyyy. m. d' → 6桁 YYMMDD int
    d = o['日付(yyyy.mm.dd)'].str.replace(' ', '', regex=False).str.split('.', expand=True)
    o['日付'] = (d[0].str[2:].str.zfill(2) + d[1].str.zfill(2) + d[2].str.zfill(2)).astype(int)

    o['馬番'] = pd.to_numeric(o['馬番'], errors='coerce').astype('Int64')
    o['rkey'] = o['レースID(新)'].str[:16]
    o['R'] = pd.to_numeric(o['rkey'].str[14:16], errors='coerce').astype('Int64')

    o['tan_odds'] = _num(o['単勝オッズ'])
    o['fuku_lo'] = _num(o['複勝オッズ下限'])
    o['fuku_hi'] = _num(o['複勝オッズ上限'])
    o['tan_pay'] = _paynum(o['単勝配当'])
    o['fuku_pay'] = _paynum(o['複勝配当'])
    for col, nm in [('馬連', 'umaren'), ('馬単', 'umatan'),
                    ('３連複', 'sanpuku'), ('３連単', 'santan')]:
        o[nm] = _num(o[col])

    horse = o[['日付', '開催', 'R', '馬番', '馬名', 'rkey', '人気', '着順',
               'tan_odds', 'fuku_lo', 'fuku_hi', 'tan_pay', 'fuku_pay']].copy()
    race = (o.groupby('rkey')
             .agg(umaren=('umaren', 'max'), umatan=('umatan', 'max'),
                  sanpuku=('sanpuku', 'max'), santan=('santan', 'max'))
             .reset_index())

    ODDS_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    horse.to_parquet(ODDS_STORE_PATH)
    race.to_parquet(RACE_STORE_PATH)
    return horse, race


def load_odds_store():
    """(horse_df, race_df) を返す。無ければ (None, None)。"""
    if not ODDS_STORE_PATH.exists():
        return None, None
    horse = pd.read_parquet(ODDS_STORE_PATH)
    race = pd.read_parquet(RACE_STORE_PATH) if RACE_STORE_PATH.exists() else None
    return horse, race


def attach_odds(df: pd.DataFrame) -> pd.DataFrame:
    """予測/master DataFrame に (日付,開催,Ｒ,馬番) で単複オッズ・配当・券種払戻を結合する。

    df 側は 日付(6桁int)・開催・Ｒ・馬番 を持つ前提（master/build_features 出力）。
    """
    horse, race = load_odds_store()
    if horse is None:
        return df
    out = df.copy()
    out['_R'] = pd.to_numeric(out.get('Ｒ', out.get('R')), errors='coerce').astype('Int64')
    out['_umaban'] = pd.to_numeric(out['馬番'], errors='coerce').astype('Int64')
    h = horse.rename(columns={'R': '_R', '馬番': '_umaban'})
    keys = ['日付', '開催', '_R', '_umaban']
    out = out.merge(h[keys + ['rkey', 'tan_odds', 'fuku_lo', 'fuku_hi', 'tan_pay', 'fuku_pay']],
                    on=keys, how='left')
    if race is not None:
        out = out.merge(race, on='rkey', how='left')
    return out.drop(columns=['_R', '_umaban'])


if __name__ == '__main__':
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else \
        str(Path.home() / 'Downloads' / '過去5年オッズ払い戻しデータ.csv')
    h, r = build_odds_store(src)
    print(f'オッズストア保存: {ODDS_STORE_PATH}')
    print(f'  馬単位 {len(h):,}行 / レース単位 {len(r):,}レース')
    print(f'  期間 {h["日付"].min()}〜{h["日付"].max()}')
