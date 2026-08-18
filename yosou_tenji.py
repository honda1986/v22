#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yosou_tenji.py -- 展示タイムを足すと1着予想はどれだけ良くなるか

■ 何を測るか
  今のモデルの対数損失は 1.2009、市場(オッズ)は約1.146。
  この差 0.055 のうち、展示タイムでどれだけ説明できるかを測る。
  = 「当日データを使わない」という縛りのコストを数字にする。

■ 作り方の方針
  展示タイムは場と体重で水準が変わるが、場も体重も既に特徴量に入っている。
  なので補正を作り込むより、素の値を足して交互作用は木に任せる方が
  筋が良い可能性がある。決めつけずに全部測って比べる。

    素の展示        そのまま。場と体重はモデルが既に持っている
    レース内偏差    その艇 − レース6艇の平均。場の水準差が消える
    レース内順位    1〜6位。外れ値に強い
    体重残差        展示を体重で回帰した残り。「体重のわりに速いか」

  ついでに進入コースと風・波も測る。どちらも当日データ。

■ 比較のしかた
  ベースラインもこのスクリプトの中で学習する。
  元の1.2009とは学習期間もパラメータも違うので絶対値は一致しないが、
  同じ条件どうしの比較なので差は読める。

  学習 2024-03-01〜2025-03-13 / 検証 2025-03-14以降
  (README の「学習に使っていない期間で検証する」に合わせる)

使い方 (Colab)
  !pip -q install lightgbm
  !git clone --depth 1 https://github.com/honda1986/v22.git
  !cp -r v22/yosou_model .
  %run v22/yosou_tenji.py
"""

import glob
import gzip
import json
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

import yosou_train as YT
import yosou_train2 as YT2

KFILE = "v22/kfile" if os.path.isdir("v22/kfile") else "kfile"
SPLIT = "20250314"       # ここ以降を検証に使う
DFROM = "20240301"       # kfile がある範囲
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1,
              verbose=-1, num_threads=os.cpu_count() or 4)
ROUNDS = 600


# ================================================================
# 1. 既存のパイプラインで特徴量を作る
# ================================================================
print("=" * 66)
print("1. 読み込み")
print("=" * 66)
t0 = time.time()

F3 = json.load(open("yosou_model/features3.json", encoding="utf-8"))
feats = F3["p1"]
print(f"  既存の1着特徴量 {len(feats)}個")

df = YT2.load2()
df = df[df["date"].astype(str) >= DFROM].copy()
df, _ = YT.add_features(df)
cube, Y1, Y2, Y3, date, n = YT2.to_cube(df, feats)
keys = df.sort_values(["race", "lane"])["race"].values[::6]
print(f"  レース {n:,} ({date.min()}〜{date.max()})  {time.time()-t0:.0f}秒")

# 1着の艇(0〜5)。形が違っても拾えるようにする
Y1 = np.asarray(Y1)
win = Y1.argmax(1) if Y1.ndim == 2 else Y1.astype(int).ravel()
print(f"  1着の分布(枠番): "
      f"{dict(zip(*np.unique(win + 1, return_counts=True)))}")


# ================================================================
# 2. kfile を結合する
# ================================================================
print("\n" + "=" * 66)
print("2. kfile の結合")
print("=" * 66)

K = {}
for p in sorted(glob.glob(f"{KFILE}/*.json.gz")):
    d = os.path.basename(p)[:8]
    if d < DFROM:
        continue
    with gzip.open(p, "rt", encoding="utf-8") as f:
        kd = json.load(f)
    for r in kd["races"]:
        K[f"{d}-{r['jcd']:02d}-{r['rno']}"] = r
print(f"  kfile レース {len(K):,}")

tenji = np.full((n, 6), np.nan)
course = np.full((n, 6), np.nan)
wind = np.full(n, np.nan)
wave = np.full(n, np.nan)
hitn = 0
for i, k in enumerate(keys):
    r = K.get(k)
    if r is None:
        continue
    hitn += 1
    wind[i] = r.get("wind") if r.get("wind") is not None else np.nan
    wave[i] = r.get("wave") if r.get("wave") is not None else np.nan
    for e in r["entries"]:
        j = e["lane"] - 1
        if 0 <= j < 6:
            if e.get("tenji") is not None:
                tenji[i, j] = e["tenji"]
            if e.get("course") is not None:
                course[i, j] = e["course"]

full = ~np.isnan(tenji).any(1)
print(f"  結合できたレース {hitn:,}/{n:,} ({hitn/n:.1%})")
print(f"  6艇すべて展示あり {full.sum():,} ({full.mean():.1%})")
print(f"  展示タイム 平均{np.nanmean(tenji):.2f} "
      f"範囲{np.nanmin(tenji):.2f}〜{np.nanmax(tenji):.2f}")

# 展示が揃ったレースだけを対象にする
use = full & ~np.isnan(wind)
cube = {c: v[use] for c, v in cube.items()}
tenji, course = tenji[use], course[use]
wind, wave = wind[use], wave[use]
win, date, keys = win[use], np.asarray(date)[use], keys[use]
n = int(use.sum())
print(f"  → 対象 {n:,}レース")


# ================================================================
# 3. 展示タイムの加工を4通り作る
# ================================================================
print("\n" + "=" * 66)
print("3. 展示タイムの加工")
print("=" * 66)

tenji_dev = tenji - tenji.mean(1, keepdims=True)
tenji_rank = tenji.argsort(1).argsort(1) + 1.0

# 体重で回帰した残差。係数は学習期間だけで求める(検証への漏れを防ぐ)
wcol = next((c for c in cube if c.lower() in
             ("weight", "wt", "w_kg")), None)
if wcol is None:
    print("  体重の列が見つからないので、体重残差は素の値で代用します")
    tenji_res = tenji.copy()
else:
    tr = date < SPLIT
    x = cube[wcol][tr].ravel()
    y = tenji[tr].ravel()
    ok = ~(np.isnan(x) | np.isnan(y))
    a, b = np.polyfit(x[ok], y[ok], 1)
    tenji_res = tenji - (a * cube[wcol] + b)
    print(f"  展示 = {a:.4f} × 体重 + {b:.3f}  "
          f"(体重1kgあたり {a*1000:.1f}ミリ秒)")
    print(f"  残差の標準偏差 {np.nanstd(tenji_res):.3f} "
          f"(素の標準偏差 {np.nanstd(tenji):.3f})")

W = np.repeat(wind[:, None], 6, 1)
V = np.repeat(wave[:, None], 6, 1)


# ================================================================
# 4. 学習して比べる
# ================================================================
print("\n" + "=" * 66)
print("4. 学習と検証")
print("=" * 66)

tr = date < SPLIT
te = ~tr
print(f"  学習 {tr.sum():,}レース / 検証 {te.sum():,}レース")

rr = np.repeat(np.arange(n), 6)
ybin = np.zeros((n, 6))
ybin[np.arange(n), win] = 1


def run(label, extra):
    """extra: {名前: (n,6)の配列}"""
    cols = list(feats)
    mats = [cube[c] for c in feats]
    for name, arr in extra.items():
        cols.append(name)
        mats.append(arr)
    X = np.column_stack([m.ravel() for m in mats]).astype(np.float32)
    y = ybin.ravel()
    trm = np.repeat(tr, 6)
    tem = np.repeat(te, 6)

    ds = lgb.Dataset(X[trm], y[trm], feature_name=cols)
    m = lgb.train(PARAMS, ds, num_boost_round=ROUNDS)

    p = m.predict(X[tem]).astype(np.float64)
    p = YT2.norm_by(p, np.repeat(np.arange(int(te.sum())), 6),
                    int(te.sum())).reshape(-1, 6)
    p = np.clip(p, 1e-9, 1)
    p /= p.sum(1, keepdims=True)
    wte = win[te]
    ll = -np.log(p[np.arange(len(wte)), wte]).mean()
    acc = (p.argmax(1) == wte).mean()
    return ll, acc, m, cols


VARIANTS = [
    ("既存のみ(基準)", {}),
    ("＋展示 素の値", {"tenji": tenji}),
    ("＋展示 レース内偏差", {"tenji_dev": tenji_dev}),
    ("＋展示 レース内順位", {"tenji_rank": tenji_rank}),
    ("＋展示 体重残差", {"tenji_res": tenji_res}),
    ("＋進入コース", {"course": course}),
    ("＋風と波", {"wind": W, "wave": V}),
    ("＋展示(素+偏差)", {"tenji": tenji, "tenji_dev": tenji_dev}),
    ("＋展示+進入", {"tenji": tenji, "tenji_dev": tenji_dev,
                   "course": course}),
    ("＋当日データ全部", {"tenji": tenji, "tenji_dev": tenji_dev,
                     "tenji_rank": tenji_rank, "tenji_res": tenji_res,
                     "course": course, "wind": W, "wave": V}),
]

res = []
base = None
last_model = last_cols = None
for label, extra in VARIANTS:
    t1 = time.time()
    ll, acc, m, cols = run(label, extra)
    if base is None:
        base = ll
    res.append((label, ll, acc, ll - base))
    print(f"  {label:<22} logloss={ll:.4f}  的中={acc:.1%}  "
          f"差{ll-base:+.4f}  ({time.time()-t1:.0f}秒)", flush=True)
    last_model, last_cols = m, cols

print("\n" + "=" * 66)
print("まとめ")
print("=" * 66)
print(f"  {'買い方':<22}{'logloss':>10}{'1着的中':>9}{'基準との差':>11}")
for label, ll, acc, d in res:
    mark = "  ←" if d < -0.005 else ""
    print(f"  {label:<22}{ll:>10.4f}{acc:>9.1%}{d:>+11.4f}{mark}")

print(f"""
  参考: 元のモデル 1.2009 / 市場(オッズ) 約1.146 / 差 0.055
        この基準は学習期間が短いので絶対値は元と一致しない。
        読むのは「基準との差」の列。
""")

# 効いている特徴量を見る
imp = pd.DataFrame({"f": last_cols,
                    "gain": last_model.feature_importance("gain")})
imp = imp.sort_values("gain", ascending=False)
new = {"tenji", "tenji_dev", "tenji_rank", "tenji_res",
       "course", "wind", "wave"}
print("  当日データの重要度(全特徴量中の順位)")
for i, r in enumerate(imp.itertuples(), 1):
    if r.f in new:
        print(f"    {i:>3}位 {r.f:<12} gain={r.gain:,.0f}")

print("""
============================================================
読み方

  差が -0.005 以上   展示は効いている。実装を検討する価値あり
  差が -0.002 未満   誤差。既存特徴量に埋もれている

  効いていた場合、実運用には締切15〜20分前の取得が必要になる。
  Kファイルはレース後の配布なので当日は使えない。
============================================================
""")
