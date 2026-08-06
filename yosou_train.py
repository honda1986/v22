#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yosou_train.py -- 前日でも分かるデータだけで各艇の1着確率を出すモデルを作る
                  (Google Colab で実行)

■ 使う情報(前日に確定しているもの)
  出走表: 枠番・級別・年齢・体重・F数・平均ST・全国勝率/2連率・当地勝率/2連率
          モーター2連率・ボート2連率
  今節:   得点率・節内順位・節平均ST・走数
  コース別(直近6ヶ月): 1着率・3連率・ST
  開催:   場・R番号・日目・節の長さ

■ 使わない情報(レース当日にならないと分からない)
  オッズ / 展示タイム / 進入コース
  ★特に進入コースは raw に本番進入が入っており、使うとカンニングになる

■ 正直な前提
  このモデルはオッズより精度が低い。v23の関門テストで確認済み。
    ファンダのみ 1.1904  /  市場のみ 1.1457
  勝つための道具ではなく、オッズが出る前に各艇の実力を眺めるための道具。

■ 使い方 (Colab)
  !pip -q install lightgbm
  !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
  %run v22/yosou_train.py
"""

import glob
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("pip install lightgbm を先に実行してください")

RAW = "v22/raw" if os.path.isdir("v22/raw") else "raw"
TOK = "v22/tokuten" if os.path.isdir("v22/tokuten") else "tokuten"
OUT = "yosou_model"

# 出走表から(前日に分かる)
CARD = ["lane", "cls_val", "age", "weight", "f_count", "avg_st",
        "n_win", "n_2ren", "l_win", "l_2ren", "m_2ren", "b_2ren"]
# 今節・コース別から(前日に分かる)
SETSU = ["tok", "srank", "genten", "nruns", "st_setsu",
         "c_win", "c_ren3", "c_st"]


def load():
    tokf = {os.path.basename(p)[:8]: p for p in glob.glob(os.path.join(TOK, "*.json.gz"))}
    rawf = sorted(glob.glob(os.path.join(RAW, "*.json.gz")))
    if not tokf:
        sys.exit(f"{TOK}/ がありません")
    print(f"raw {len(rawf)}日 / tokuten {len(tokf)}日")

    rows = []
    t0 = time.time()
    for k, p in enumerate(rawf):
        d = os.path.basename(p)[:8]
        with gzip.open(p, "rt", encoding="utf-8") as f:
            rd = json.load(f)
        tk, meta = {}, {}
        if d in tokf:
            with gzip.open(tokf[d], "rt", encoding="utf-8") as f:
                td = json.load(f)
            for js, v in td.get("venues", {}).items():
                j = int(js)
                for r in v.get("races", []):
                    meta[(j, r["rno"])] = (r.get("name", ""), v.get("day_no"),
                                           v.get("n_days"))
                    for x in r["lanes"]:
                        tk[(j, r["rno"], x["lane"])] = x

        for r in rd["races"]:
            if "error" in r or not r.get("hit"):
                continue
            hit1 = int(r["hit"].split("-")[0])
            name, day_no, n_days = meta.get((r["jcd"], r["rno"]), ("", None, None))
            for e in r["entries"]:
                x = tk.get((r["jcd"], r["rno"], e["lane"]), {})
                rows.append({
                    "date": int(d), "jcd": r["jcd"], "rno": r["rno"],
                    "race": f"{d}-{r['jcd']:02d}-{r['rno']}",
                    **{c: e.get(c) for c in CARD},
                    "tok": x.get("tokuten"), "srank": x.get("rank"),
                    "genten": x.get("genten"), "nruns": x.get("n_runs"),
                    "st_setsu": x.get("st_setsu"), "c_win": x.get("c_win"),
                    "c_ren3": x.get("c_ren3"), "c_st": x.get("c_st"),
                    "day_no": day_no, "n_days": n_days,
                    "is_final": 1 if any(w in (name or "")
                                         for w in ("準優", "優勝", "選抜")) else 0,
                    "y": 1 if e["lane"] == hit1 else 0,
                })
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(rawf)}日  {len(rows):,}行  "
                  f"{time.time()-t0:.0f}秒", flush=True)
    df = pd.DataFrame(rows)
    print(f"読み込み完了 {len(df):,}行 / {df['race'].nunique():,}レース  "
          f"{time.time()-t0:.0f}秒")
    return df


def add_features(df):
    df = df.copy()
    for c in ("tenji", "st_setsu", "c_st"):
        if c in df:
            df.loc[df[c] <= 0, c] = np.nan
    g = df.groupby("race")
    # レース内での相対化(強さは絶対値より相対で効く)
    for c in ("n_win", "l_win", "m_2ren", "c_win", "avg_st", "tok", "st_setsu"):
        df[f"{c}_dev"] = df[c] - g[c].transform("mean")
        df[f"{c}_rk"] = g[c].rank(ascending=(c in ("avg_st", "st_setsu")),
                                  method="min")
    df["cls_max"] = g["cls_val"].transform("max")
    df["cls_gap"] = df["cls_val"] - df["cls_max"]
    df["lane1_win"] = g["c_win"].transform("max")
    feats = (CARD + SETSU + ["jcd", "rno", "day_no", "n_days", "is_final",
                             "cls_max", "cls_gap"] +
             [f"{c}_dev" for c in ("n_win", "l_win", "m_2ren", "c_win",
                                   "avg_st", "tok", "st_setsu")] +
             [f"{c}_rk" for c in ("n_win", "l_win", "m_2ren", "c_win",
                                  "avg_st", "tok", "st_setsu")])
    return df, [f for f in feats if f in df.columns]


def norm(raw, race):
    s = pd.Series(raw).groupby(race).transform("sum").values
    return raw / np.maximum(s, 1e-12)


def main():
    df = load()
    df, feats = add_features(df)
    dates = np.sort(df["date"].unique())
    cut = dates[int(len(dates) * 0.75)]
    tr = df["date"] < cut
    te = df["date"] >= cut
    va_cut = dates[int(len(dates) * 0.65)]
    print(f"\n学習 {int(tr.sum())//6:,}レース (〜{cut})")
    print(f"検証 {int(te.sum())//6:,}レース ({cut}〜)")
    print(f"特徴量 {len(feats)}個  (オッズ・展示タイム・進入コースは不使用)")

    p = dict(objective="binary", learning_rate=0.04, num_leaves=63,
             min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, verbose=-1, seed=42)
    m = lgb.train(p,
                  lgb.Dataset(df.loc[tr & (df["date"] < va_cut), feats],
                              df.loc[tr & (df["date"] < va_cut), "y"]),
                  num_boost_round=3000,
                  valid_sets=[lgb.Dataset(df.loc[tr & (df["date"] >= va_cut), feats],
                                          df.loc[tr & (df["date"] >= va_cut), "y"])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"木の数 {m.best_iteration}")

    pr = norm(m.predict(df.loc[te, feats]), df.loc[te, "race"].values)
    y = df.loc[te, "y"].values
    race = df.loc[te, "race"].values
    lane = df.loc[te, "lane"].values

    print("\n" + "=" * 54)
    print("[1] 精度")
    ll = -np.log(np.clip(pr[y == 1], 1e-12, None)).mean()
    base = -np.log(np.clip(norm(np.where(lane == 1, .55, .09), race)[y == 1],
                           1e-12, None)).mean()
    print(f"  このモデル      対数損失 {ll:.4f}")
    print(f"  枠番だけの目安  対数損失 {base:.4f}")
    print(f"  参考: 市場(オッズ)は 約1.146。オッズには勝てません")

    d = pd.DataFrame({"race": race, "p": pr, "y": y, "lane": lane})
    top = d.loc[d.groupby("race")["p"].idxmax()]
    print(f"  1番手に選んだ艇の的中率 {top['y'].mean()*100:.1f}%")

    print("\n[2] 較正 (出した確率どおりに当たっているか)")
    b = pd.cut(d["p"], [0, .05, .1, .15, .2, .3, .4, .5, .7, 1])
    t = d.groupby(b, observed=True).agg(n=("y", "size"), 予想=("p", "mean"),
                                        実測=("y", "mean"))
    for k, r in t.iterrows():
        if r["n"] < 200:
            continue
        print(f"  {str(k):<12} {int(r['n']):>7,}件  "
              f"予想{r['予想']*100:5.1f}%  実測{r['実測']*100:5.1f}%")

    print("\n[3] 効いた特徴量 上位15")
    imp = pd.Series(m.feature_importance("gain"), index=feats)
    for k, v in imp.sort_values(ascending=False).head(15).items():
        print(f"  {k:<16} {v:,.0f}")

    os.makedirs(OUT, exist_ok=True)
    m.save_model(f"{OUT}/lgb_yosou.txt")
    json.dump(feats, open(f"{OUT}/features.json", "w"))
    print(f"\n{OUT}/ に保存しました。ダウンロードしてリポジトリに置いてください。")


if __name__ == "__main__":
    main()
