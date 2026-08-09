#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yosou_bunpai.py -- 勝負レースで1万円をどう配分するかを比べる (Colab)

■ 比べる3つの配分
  A 均等買い      … 8点に1,250円ずつ
  B 払戻を均等に  … オッズに反比例して配分(どれが当たっても同じ払戻)
  C 確率に比例    … モデルの確率が高い組ほど多く買う

■ 先に理屈
  買う組の集合が同じなら、期待払戻は配分によらない。
    期待払戻 = Σ(投資額_i × p_i × o_i)
  p_i × o_i はどの組でも0.75付近(控除率25%)なので、
  どう配分しても投資額×0.75あたりに落ち着く。
  配分が変えるのは「ばらつき」であって「回収率」ではない。

  それを34,018レースの実測で確かめる。

■ 対象
  勝負レース(上位8組の確率合計 >= 0.553)、上位8組を購入、1レース1万円

  使い方 (Colab)
    !pip -q install lightgbm
    !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
    !rm -rf app && git clone --depth 1 https://github.com/honda1986/boatrace-app.git app
    !mkdir -p yosou_model && cp app/yosou_model/* yosou_model/
    %run v22/yosou_bunpai.py
"""

import argparse
import glob
import gzip
import json
import os

import numpy as np
import pandas as pd

import yosou_train as YT
import yosou_train2 as YT2

MODEL = "yosou_model"
BUDGET = 10000.0
NPTS = 8
SHOBU_TH = 0.553


def pc(x):
    return f"{x*100:.1f}%"


def tbl(header, rows):
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    print("  " + "  ".join(str(h).rjust(w[i]) for i, h in enumerate(header)))
    for r in rows:
        print("  " + "  ".join(str(v).rjust(w[i]) for i, v in enumerate(r)))


def load_odds(dfrom):
    combos = None
    out = {}
    for p in sorted(glob.glob(os.path.join(YT.RAW, "*.json.gz"))):
        d = os.path.basename(p)[:8]
        if d < dfrom:
            continue
        with gzip.open(p, "rt", encoding="utf-8") as f:
            rd = json.load(f)
        if combos is None:
            combos = rd["combos"]
        for r in rd["races"]:
            if "error" in r or r.get("n_odds") != 120 or not r.get("hit"):
                continue
            if not r.get("pay_3t"):
                continue
            out[f"{d}-{r['jcd']:02d}-{r['rno']}"] = (
                np.array(r["odds"], dtype=np.float64), r["hit"])
    return out, combos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="dfrom", default="20250314")
    ap.add_argument("--th", type=float, default=SHOBU_TH)
    ap.add_argument("--pts", type=int, default=NPTS)
    args = ap.parse_args()

    import lightgbm as lgb
    F3 = json.load(open(f"{MODEL}/features3.json", encoding="utf-8"))
    feats = F3["p1"]
    m1 = lgb.Booster(model_file=f"{MODEL}/lgb_p1.txt")
    m2 = lgb.Booster(model_file=f"{MODEL}/lgb_p2.txt")
    m3 = lgb.Booster(model_file=f"{MODEL}/lgb_p3.txt")

    df = YT2.load2()
    df = df[df["date"].astype(str) >= args.dfrom].copy()
    df, _ = YT.add_features(df)
    cube, Y1, Y2, Y3, date, n = YT2.to_cube(df, feats)
    keys = df.sort_values(["race", "lane"])["race"].values[::6]
    print(f"\n検証 {n:,}レース")

    Xf = np.column_stack([cube[c].ravel() for c in feats]).astype(np.float32)
    rr = np.repeat(np.arange(n), 6)
    p1 = YT2.norm_by(m1.predict(Xf).astype(np.float32), rr, n).reshape(n, 6)
    cube["p1"] = p1

    pairs = [(a, b) for a in range(6) for b in range(6) if b != a]
    tri = [(a, b, c) for a, b in pairs for c in range(6) if c not in (a, b)]
    P2 = np.zeros((n, 6, 6))
    for a in range(6):
        X, _, _, R, L, _ = YT2.build(cube, feats, n, [np.full(n, a)])
        P2[R, a, L] = np.maximum(m2.predict(X).astype(np.float32), 1e-9)
    P2 /= P2.sum(2, keepdims=True)
    P3 = np.zeros((n, 6, 6, 6))
    for a, b in pairs:
        X, _, _, R, L, _ = YT2.build(cube, feats, n,
                                     [np.full(n, a), np.full(n, b)])
        P3[R, a, b, L] = np.maximum(m3.predict(X).astype(np.float32), 1e-9)
    for a, b in pairs:
        P3[:, a, b] /= P3[:, a, b].sum(1, keepdims=True)
    CP = np.stack([p1[:, a] * P2[:, a, b] * P3[:, a, b, c] for a, b, c in tri], 1)
    names = [f"{a+1}-{b+1}-{c+1}" for a, b, c in tri]
    print("確率を計算しました")

    OD, combos = load_odds(args.dfrom)
    sum8 = np.sort(CP, 1)[:, -8:].sum(1)
    sel = np.array([(keys[i] in OD) and (sum8[i] >= args.th) for i in range(n)])
    print(f"勝負レース(閾値{args.th}) {int(sel.sum()):,}件 "
          f"/ 全体の{sel.mean()*100:.1f}%")

    N = args.pts
    res = {k: [] for k in ("A", "B", "C")}
    hits = 0
    for i in np.where(sel)[0]:
        o_all, hit = OD[keys[i]]
        ordi = np.argsort(-CP[i])[:N]
        cn = [names[j] for j in ordi]
        pr = CP[i][ordi]
        od = np.array([o_all[combos.index(c)] for c in cn])
        won = hit in cn
        j = cn.index(hit) if won else -1
        hits += won

        stakes = {
            "A": np.full(N, BUDGET / N),                    # 均等
            "B": BUDGET * (1 / od) / (1 / od).sum(),        # 払戻を均等に
            "C": BUDGET * pr / pr.sum(),                    # 確率に比例
        }
        for k, st in stakes.items():
            ret = st[j] * od[j] if won else 0.0
            res[k].append(ret)

    m = int(sel.sum())
    print(f"的中 {hits:,} / {m:,}  ({hits/m*100:.1f}%)\n")

    print("=" * 58)
    print(f"[1] 1レース{BUDGET:,.0f}円で上位{N}組を買った場合")
    rows = []
    labels = {"A": "均等買い", "B": "払戻を均等に", "C": "確率に比例"}
    for k in ("A", "B", "C"):
        v = np.array(res[k])
        cost = m * BUDGET
        roi = v.sum() / cost
        se = np.sqrt(((v - roi * BUDGET) ** 2).sum()) / cost
        rows.append([labels[k], pc(roi), pc(se), f"{v.sum()-cost:+,.0f}",
                     f"{v[v>0].mean():,.0f}" if (v > 0).any() else "—",
                     f"{v[v>0].min():,.0f}" if (v > 0).any() else "—",
                     f"{v[v>0].max():,.0f}" if (v > 0).any() else "—"])
    tbl(["配分", "回収率", "誤差±", "収支", "平均払戻", "最小", "最大"], rows)

    print("\n[2] 当たったときの払戻のばらつき")
    rows = []
    for k in ("A", "B", "C"):
        v = np.array(res[k])
        w = v[v > 0]
        rows.append([labels[k], f"{w.std():,.0f}",
                     f"{w.std()/w.mean():.2f}",
                     f"{np.percentile(w,25):,.0f}",
                     f"{np.percentile(w,50):,.0f}",
                     f"{np.percentile(w,75):,.0f}"])
    tbl(["配分", "標準偏差", "ばらつき率", "下位25%", "中央", "上位25%"], rows)
    print("  ばらつき率が小さいほど『どれが当たっても同じくらい戻る』")

    print("\n[3] 資金の減り方 (1レースずつ買い続けた場合)")
    rows = []
    for k in ("A", "B", "C"):
        v = np.array(res[k])
        bal = np.cumsum(v - BUDGET)
        rows.append([labels[k], f"{bal[-1]:+,.0f}",
                     f"{bal.min():+,.0f}",
                     f"{int(np.max(np.maximum.accumulate(bal)-bal)):,}"])
    tbl(["配分", "最終収支", "最悪時", "最大の落ち込み"], rows)

    print("\n" + "=" * 58)
    print("結論の読み方")
    print("  ・回収率が3つともほぼ同じなら、配分では期待値を動かせない")
    print("  ・払戻を均等にする配分は『ばらつき率』が小さくなるはず")
    print("  ・つまり配分が変えるのは体験であって、勝ち負けではない")
    print("  ・締切時オッズでの数字。実運用はこれより落ちる")


if __name__ == "__main__":
    main()
