"""検証：前半期間で見つけた条件が後半期間でも再現するか（アウトオブサンプル）"""
import numpy as np
import pandas as pd
import cond_core as C

pd.set_option('display.width', 300)
pd.set_option('display.max_rows', 100)

d = C.load()
d = d[~d['jump']].copy()
d['dist_grp'] = pd.cut(d['距離'], [0, 1400, 1800, 2200, 9999],
                       labels=['短距離~1400', 'マイル1401-1800', '中距離1801-2200', '長距離2201~'])
mid = pd.Timestamp('2024-02-01')     # 前半 21/7-24/1, 後半 24/2-26/7
A, B = d[d['日付_dt'] < mid], d[d['日付_dt'] >= mid]
print(f'前半 {len(A):,}行 ({A["日付_dt"].min().date()}~{A["日付_dt"].max().date()}) / '
      f'後半 {len(B):,}行 ({B["日付_dt"].min().date()}~{B["日付_dt"].max().date()})')


def oos(by, min_n_a, min_n_b, label, metric='単回収率%', top=15):
    a = C.agg(A, by, min_n=min_n_a)
    b = C.agg(B, by, min_n=min_n_b)
    j = a[['n', metric]].join(b[['n', metric]], lsuffix='_前半', rsuffix='_後半', how='inner')
    j = j.dropna()
    corr = j[f'{metric}_前半'].corr(j[f'{metric}_後半'])
    # 前半トップ群が後半でも基準超えか
    t = j.sort_values(f'{metric}_前半', ascending=False).head(top)
    print(f'\n--- {label} ---')
    print(f'条件数={len(j)}  前半-後半 相関 r={corr:.3f}')
    print(f'前半トップ{top}の後半平均{metric} = {t[f"{metric}_後半"].mean():.1f} '
          f'(全体平均 {j[f"{metric}_後半"].mean():.1f})')
    print(t.round(1).to_string())
    return corr


print('\n' + '=' * 70)
print('【1】枠バイアス（コース × 枠グループ）')
oos(['venue', 'turf', 'dist_grp', 'waku_grp'], 200, 200, 'コース×枠', metric='複回収率%')

print('\n' + '=' * 70)
print('【2】脚質バイアス（コース × 脚質）')
oos(['venue', 'turf', 'dist_grp', 'kyakushitsu'], 150, 150, 'コース×脚質', metric='複勝率%')

print('\n' + '=' * 70)
print('【3】種牡馬')
oos(['種牡馬', 'turf'], 100, 100, '種牡馬×芝ダ')

print('\n' + '=' * 70)
print('【4】騎手')
oos(['騎手', 'turf'], 100, 100, '騎手×芝ダ')

print('\n' + '=' * 70)
print('【5】調教師')
oos(['調教師', 'turf'], 100, 100, '調教師×芝ダ')

print('\n' + '=' * 70)
print('【6】人気帯×脚質')
oos(['ninki_grp', 'kyakushitsu'], 300, 300, '人気×脚質')

# --- 新潟芝直線1000mの外枠有利を単独検証 ---
print('\n' + '=' * 70)
print('【7】新潟芝1000m（直線）の枠別 — 期間別')
n1000 = d[(d['venue'] == '新潟') & (d['turf'] == '芝') & (d['距離'] == 1000)]
for lbl, sub in [('前半', n1000[n1000['日付_dt'] < mid]), ('後半', n1000[n1000['日付_dt'] >= mid])]:
    print(f'\n[{lbl}] n={len(sub)}')
    print(C.agg(sub, 'waku_grp', min_n=50)[['n', '勝率%', '複勝率%', '単回収率%', '複回収率%']].round(1).to_string())

# --- 逃げ馬の単勝回収率が100%超えるのは「結果論」か検証 ---
print('\n' + '=' * 70)
print('【8】逃げ馬の回収率は事前に使えない（4角通過は結果）。')
print('事前情報である「前走4角1番手」で代替検証：')
d['prev_nige'] = (pd.to_numeric(d['前4角'], errors='coerce') == 1)
print(C.agg(d[~d['jump']], ['prev_nige', 'turf'], min_n=500)[
    ['n', '勝率%', '複勝率%', '単回収率%', '複回収率%', 'avg_ninki']].round(1).to_string())
print('\n前走逃げ × 今回の人気帯:')
print(C.agg(d[d['prev_nige']], 'ninki_grp', min_n=200)[
    ['n', '勝率%', '複勝率%', '単回収率%', '複回収率%']].round(1).to_string())
