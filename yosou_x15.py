#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yosou_x15.py -- 「払戻が15,000〜20,000円の層だけ買う」を追う (Colab)

■ 前回出た数字
  勝負レース × 確率5%以上 × 払戻均等
    払戻15,000〜20,000円  1,333レース  的中率53.5%  回収率89.6% ± 2.3%
  他の層は81%台。ここだけ飛び抜けている。

■ ただし前科がある
  以前「払戻15,000円以上」で226件・90.2%が出たが、
  母数を10倍にしたら消えた(全レースで見ると払戻が大きいほど回収率は下がる)。

■ 今回やること
  条件を緩めて母数を増やし、同じ数字が保たれるかを見る。
    勝負レースの閾値を 0.553 → 0.50 → 0.45 → 0.40 → 条件なし
    確率の閾値を 5% → 4% → 3%
  母数が増えても89%前後なら本物。81%に戻るなら上振れ。

  使い方 (Colab)
    !pip -q install lightgbm
    !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
    !cp -r v22/yosou_model .
    %run v22/yosou_x15.py
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="dfrom", default="20250314")
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
        X, _, _, R_, L_, _ = YT2.build(cube, feats, n, [np.full(n, a)])
        P2[R_, a, L_] = np.maximum(m2.predict(X).astype(np.float32), 1e-9)
    P2 /= P2.sum(2, keepdims=True)
    P3 = np.zeros((n, 6, 6, 6))
    for a, b in pairs:
        X, _, _, R_, L_, _ = YT2.build(cube, feats, n,
                                       [np.full(n, a), np.full(n, b)])
        P3[R_, a, b, L_] = np.maximum(m3.predict(X).astype(np.float32), 1e-9)
    for a, b in pairs:
        P3[:, a, b] /= P3[:, a, b].sum(1, keepdims=True)
    CP = np.stack([p1[:, a] * P2[:, a, b] * P3[:, a, b, c] for a, b, c in tri], 1)
    names = [f"{a+1}-{b+1}-{c+1}" for a, b, c in tri]
    print("確率を計算しました")

    combos = None
    OD = {}
    for p in sorted(glob.glob(os.path.join(YT.RAW, "*.json.gz"))):
        d = os.path.basename(p)[:8]
        if d < args.dfrom:
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
            OD[f"{d}-{r['jcd']:02d}-{r['rno']}"] = (
                r["hit"], np.array(r["odds"], dtype=np.float64))
    cidx = {c: i for i, c in enumerate(combos)}
    reo = np.array([cidx[c] for c in names])

    idx = np.array([i for i in range(n) if keys[i] in OD])
    HI = np.array([names.index(OD[keys[i]][0]) for i in idx])
    ODS = np.stack([OD[keys[i]][1][reo] for i in idx])
    P = CP[idx]
    DT = date[idx]
    S8 = np.sort(P, 1)[:, -8:].sum(1)
    m = len(idx)
    print(f"突き合わせ {m:,}レース\n")

    def make(th_p):
        """確率th_p以上の組を買ったときの X(払戻) と 的中 を返す"""
        X = np.full(m, np.nan)
        won = np.zeros(m, dtype=bool)
        cnt = np.zeros(m, dtype=int)
        for i in range(m):
            sel = np.where(P[i] >= th_p)[0]
            cnt[i] = len(sel)
            if len(sel) == 0:
                continue
            X[i] = BUDGET / (1.0 / ODS[i][sel]).sum()
            won[i] = HI[i] in sel
        return X, won, cnt

    CACHE = {t: make(t) for t in (0.03, 0.04, 0.05)}

    def stat(mask, th_p, minn=200):
        X, won, cnt = CACHE[th_p]
        mm = mask & ~np.isnan(X)
        k = int(mm.sum())
        if k < minn:
            return None
        v = X[mm] * won[mm]
        cost = k * BUDGET
        roi = v.sum() / cost
        se = np.sqrt(((v - roi * BUDGET) ** 2).sum()) / cost
        return {"n": k, "hit": won[mm].mean(), "roi": roi, "se": se,
                "X": X[mm].mean(), "pl": v.sum() - cost}

    print("=" * 62)
    print("[1] 前回の数字を再現  (勝負レース 0.553 × 確率5%以上)")
    base = S8 >= 0.553
    X5, _, _ = CACHE[0.05]
    out = []
    for lo, hi in ((0, 12000), (12000, 15000), (15000, 20000), (20000, 10**9)):
        s = stat(base & (X5 >= lo) & (X5 < hi), 0.05)
        if s:
            out.append([f"{lo:,}〜{hi:,}円" if hi < 10**8 else "20,000円〜",
                        f"{s['n']:,}", f"{s['X']:,.0f}円", pc(s["hit"]),
                        pc(s["roi"]), pc(s["se"])])
    tbl(["払戻の水準", "レース", "平均払戻", "的中率", "回収率", "誤差±"], out)

    print("\n" + "=" * 62)
    print("[2] 勝負レースの条件を緩めて母数を増やす")
    print("  払戻15,000〜20,000円・確率5%以上。89%が保たれるか")
    out = []
    for th in (0.553, 0.52, 0.50, 0.45, 0.40, 0.0):
        mm = (S8 >= th) & (X5 >= 15000) & (X5 < 20000)
        s = stat(mm, 0.05, 150)
        if s:
            out.append([f"{th:.3f}以上" if th else "条件なし", f"{s['n']:,}",
                        pc(s["hit"]), pc(s["roi"]), pc(s["se"]),
                        f"{s['pl']:+,.0f}"])
    tbl(["確率合計", "レース", "的中率", "回収率", "誤差±", "収支"], out)

    print("\n[3] 確率の閾値も緩める (勝負レース条件なし・払戻15,000〜20,000円)")
    out = []
    for tp in (0.03, 0.04, 0.05):
        Xt, _, _ = CACHE[tp]
        mm = (Xt >= 15000) & (Xt < 20000)
        s = stat(mm, tp, 150)
        if s:
            out.append([f"{tp*100:.0f}%以上", f"{s['n']:,}", pc(s["hit"]),
                        pc(s["roi"]), pc(s["se"]), f"{s['pl']:+,.0f}"])
    tbl(["確率の閾値", "レース", "的中率", "回収率", "誤差±", "収支"], out)

    print("\n" + "=" * 62)
    print("[4] 払戻の水準を細かく切る (勝負レース条件なし・確率5%以上)")
    out = []
    for lo, hi in ((10000, 13000), (13000, 15000), (15000, 17000),
                   (17000, 20000), (20000, 25000), (25000, 10**9)):
        s = stat((X5 >= lo) & (X5 < hi), 0.05)
        if s:
            out.append([f"{lo//1000}〜{hi//1000}千円" if hi < 10**8 else "25千円〜",
                        f"{s['n']:,}", pc(s["hit"]), pc(s["roi"]), pc(s["se"])])
    tbl(["払戻", "レース", "的中率", "回収率", "誤差±"], out)

    print("\n[5] 年度で割る (払戻15,000〜20,000円・確率5%以上・勝負条件なし)")
    yr = (DT // 10000 - ((DT // 100) % 100 < 5)).astype(int)
    out = []
    for y in sorted(set(yr.tolist())):
        s = stat((X5 >= 15000) & (X5 < 20000) & (yr == y), 0.05, 100)
        if s:
            out.append([f"{y}年度", f"{s['n']:,}", pc(s["hit"]), pc(s["roi"]),
                        pc(s["se"])])
    tbl(["年度", "レース", "的中率", "回収率", "誤差±"], out)

    print("\n[6] 勝負レース × 払戻15,000〜20,000円 を年度で割る")
    out = []
    for y in sorted(set(yr.tolist())):
        s = stat(base & (X5 >= 15000) & (X5 < 20000) & (yr == y), 0.05, 100)
        if s:
            out.append([f"{y}年度", f"{s['n']:,}", pc(s["hit"]), pc(s["roi"]),
                        pc(s["se"])])
    tbl(["年度", "レース", "的中率", "回収率", "誤差±"], out)

    print("\n" + "=" * 62)
    print("判断")
    print("  ・[2]で母数を増やしても89%前後なら本物")
    print("  ・81%台に戻るなら、1,333件の上振れだった")
    print("  ・[4]が単調でなければノイズ")
    print("  ・年度がそろわない条件は使わない")


if __name__ == "__main__":
    main()
