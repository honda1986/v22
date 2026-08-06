#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yosou_check.py -- 「本命が1号艇でないとき」に何かあるかを測る

■ 仮説(はむさん)
  モデルの本命が1号艇でないレースは、1着の的中率が高いように見える。

■ 測り方
  学習に使っていない期間だけを使う(--from 20250314)。
  モデルの本命について、実測1着率と市場想定(3連単オッズから逆算)を比べる。
  想定回収率 = 75% × 実測 ÷ 市場。100%に届くには 実測÷市場 が 1.33 以上必要。
  実際に3連単を買った場合の回収率も出す。

  使い方 (Colab)
    !pip -q install lightgbm
    !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
    %run v22/yosou_check.py
"""

import argparse
import gzip
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import yosou_train as YT

RAW = YT.RAW
MODEL = "v22/yosou_model" if os.path.isdir("v22/yosou_model") else "yosou_model"


def pc(x):
    return f"{x*100:.1f}%"


def tbl(header, rows):
    if not rows:
        print("  (該当なし)")
        return
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    print("  " + "  ".join(str(h).rjust(w[i]) for i, h in enumerate(header)))
    for r in rows:
        print("  " + "  ".join(str(v).rjust(w[i]) for i, v in enumerate(r)))


def load_odds(dfrom):
    """レースごとの3連単オッズと的中組"""
    combos = None
    out = {}
    for p in sorted(glob.glob(os.path.join(RAW, "*.json.gz"))):
        d = os.path.basename(p)[:8]
        if d < dfrom:
            continue
        with gzip.open(p, "rt", encoding="utf-8") as f:
            rd = json.load(f)
        if combos is None:
            combos = rd["combos"]
            cix = {c: i for i, c in enumerate(combos)}
            first = np.array([int(c.split("-")[0]) for c in combos])
        for r in rd["races"]:
            if "error" in r or r.get("n_odds") != 120 or not r.get("hit"):
                continue
            if not r.get("pay_3t"):
                continue
            o = np.array(r["odds"], dtype=np.float64)
            hi = cix[r["hit"]]
            if abs(o[hi] * 100 - r["pay_3t"]) > 10:
                continue
            out[f"{d}-{r['jcd']:02d}-{r['rno']}"] = (o, hi, r["pay_3t"])
    return out, combos, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="dfrom", default="20250314",
                    help="学習に使っていない期間の開始")
    args = ap.parse_args()

    import lightgbm as lgb
    feats = json.load(open(f"{MODEL}/features.json", encoding="utf-8"))
    model = lgb.Booster(model_file=f"{MODEL}/lgb_yosou.txt")

    df = YT.load()
    df = df[df["date"].astype(str) >= args.dfrom].copy()
    df, _ = YT.add_features(df)
    print(f"\n検証対象 {df['race'].nunique():,}レース ({args.dfrom}〜)")

    raw = model.predict(df[feats])
    df["p"] = YT.norm(raw, df["race"].values)

    OD, combos, first = load_odds(args.dfrom)
    print(f"オッズが揃ったレース {len(OD):,}")

    rows = []
    for rid, g in df.groupby("race", sort=False):
        if rid not in OD:
            continue
        o, hi, pay = OD[rid]
        inv = 1.0 / o
        q = inv / inv.sum()
        q1 = np.array([q[first == l].sum() for l in range(1, 7)])
        g = g.sort_values("lane")
        p = g["p"].values
        top = int(np.argmax(p))
        rows.append({
            "race": rid, "date": int(rid[:8]),
            "top_lane": top + 1, "top_p": p[top], "top_q": q1[top],
            "won": 1 if g["y"].values[top] == 1 else 0,
            "o": o, "hi": hi, "pay": pay,
        })
    R = pd.DataFrame(rows)
    print(f"突き合わせ {len(R):,}レース\n")

    def stat(m):
        k = int(m.sum())
        if k < 200:
            return None
        a = R.loc[m, "won"].mean()
        q = R.loc[m, "top_q"].mean()
        se = np.sqrt(max(a * (1 - a), 1e-9) / k)
        return {"n": k, "act": a, "mkt": q, "model": R.loc[m, "top_p"].mean(),
                "z": (a - q) / se, "roi": 0.75 * a / max(q, 1e-9)}

    print("=" * 58)
    print("[1] モデルの本命の実力")
    print("  想定回収率 = 75% × 実測 ÷ 市場。100%には 実測÷市場 が1.33必要")
    g = [("本命が1号艇", R["top_lane"] == 1),
         ("本命が1号艇でない", R["top_lane"] != 1),
         ("　うち2号艇", R["top_lane"] == 2),
         ("　うち3号艇", R["top_lane"] == 3),
         ("　うち4号艇以降", R["top_lane"] >= 4)]
    out = []
    for lab, m in g:
        s = stat(m)
        if not s:
            continue
        out.append([lab, f"{s['n']:,}", pc(s["model"]), pc(s["mkt"]),
                    pc(s["act"]), f"{s['z']:+.1f}", pc(s["roi"])])
    tbl(["区分", "レース", "モデル", "市場", "実測", "z値", "想定回収率"], out)

    print("\n[2] モデルの自信の強さ別 (本命が1号艇でないレースのみ)")
    sub = R[R["top_lane"] != 1]
    qs = np.quantile(sub["top_p"], [0.25, 0.5, 0.75])
    g2 = [(f"〜{qs[0]:.0%}", sub["top_p"] < qs[0]),
          (f"{qs[0]:.0%}〜{qs[1]:.0%}", (sub["top_p"] >= qs[0]) & (sub["top_p"] < qs[1])),
          (f"{qs[1]:.0%}〜{qs[2]:.0%}", (sub["top_p"] >= qs[1]) & (sub["top_p"] < qs[2])),
          (f"{qs[2]:.0%}〜", sub["top_p"] >= qs[2])]
    out = []
    for lab, m in g2:
        idx = sub.index[m]
        s = stat(R.index.isin(idx))
        if s:
            out.append([lab, f"{s['n']:,}", pc(s["model"]), pc(s["mkt"]),
                        pc(s["act"]), f"{s['z']:+.1f}", pc(s["roi"])])
    tbl(["モデル確率", "レース", "モデル", "市場", "実測", "z値", "想定回収率"], out)

    print("\n" + "=" * 58)
    print("[3] 実際に3連単を買った場合  (本命を1着に固定し、人気順にN点)")
    out = []
    for lab, m in g:
        idx = R.index[m]
        if len(idx) < 200:
            continue
        for N in (4, 8, 12):
            cost = ret = 0.0
            hits = 0
            for i in idx:
                o = R.at[i, "o"]
                tl = R.at[i, "top_lane"]
                cand = np.where(first == tl)[0]
                sel = cand[np.argsort(o[cand])[:N]]
                cost += N * 100
                if R.at[i, "hi"] in sel:
                    ret += R.at[i, "pay"]
                    hits += 1
            r = ret / cost
            se = np.sqrt(max(hits / len(idx) * (1 - hits / len(idx)), 1e-9) / len(idx))
            out.append([lab, N, f"{len(idx):,}", f"{hits:,}",
                        pc(hits / len(idx)), pc(r), f"{ret-cost:+,.0f}"])
    tbl(["区分", "点数", "レース", "的中", "的中率", "回収率", "収支"], out)

    print("\n" + "=" * 58)
    print("[4] 期間で割って再現するか (本命が1号艇でないレース)")
    sub = R[R["top_lane"] != 1]
    half = sub["date"].median()
    out = []
    for lab, m in (("前半", sub["date"] <= half), ("後半", sub["date"] > half)):
        s = stat(R.index.isin(sub.index[m]))
        if s:
            out.append([lab, f"{s['n']:,}", pc(s["mkt"]), pc(s["act"]),
                        f"{s['z']:+.1f}", pc(s["roi"])])
    tbl(["期間", "レース", "市場", "実測", "z値", "想定回収率"], out)

    print("\n" + "=" * 58)
    print("判断の目安")
    print("  ・[1]の想定回収率が100%未満なら、的中率が高くても金にはならない")
    print("  ・[3]が実際の回収率。100%を超えなければ買えない")
    print("  ・前半と後半のどちらかが割れるなら偶然")
    print("  ・締切時オッズでの数字。実運用はこれより落ちる")


if __name__ == "__main__":
    main()
