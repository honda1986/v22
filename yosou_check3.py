#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yosou_check3.py -- 3段カスケードの予想が、どんなレースで当たるかを測る (Colab)

■ 測ること
  学習に使っていない期間(20250314〜)で、3連単の上位N組を買った場合の
  的中率と回収率を、レースの性質ごとに出す。

  的中の定義 = 3連単の上位N組に入る(N = 3 / 6 / 8 / 12)

■ 絞る軸(勝負レースの候補)
  本命の1着確率 / 上位8組の確率合計 / 本命が1号艇か
  1着と2着の差 / 開催日目 / A1の人数 / 場

■ 正直な前提
  「絞れば回収率が上がる」は11回すべて否定されている。
  当たりやすいレースはオッズも安いので回収率は平らになるはず。
  ただし『的中率で絞る』のは意味がある。外れ続けずに遊べる。

  使い方 (Colab)
    !pip -q install lightgbm
    !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
    %run v22/yosou_check3.py
"""

import argparse
import glob
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import yosou_train as YT
import yosou_train2 as YT2

MODEL = "v22/yosou_model" if os.path.isdir("v22/yosou_model") else "yosou_model"
NPTS = [3, 6, 8, 12]


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
                np.array(r["odds"], dtype=np.float64), r["hit"], r["pay_3t"])
    return out, combos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="dfrom", default="20250314")
    args = ap.parse_args()

    import lightgbm as lgb
    F3 = json.load(open(f"{MODEL}/features3.json", encoding="utf-8"))
    feats, n2, n3 = F3["p1"], F3["p2"], F3["p3"]
    m1 = lgb.Booster(model_file=f"{MODEL}/lgb_p1.txt")
    m2 = lgb.Booster(model_file=f"{MODEL}/lgb_p2.txt")
    m3 = lgb.Booster(model_file=f"{MODEL}/lgb_p3.txt")

    df = YT2.load2()
    df = df[df["date"].astype(str) >= args.dfrom].copy()
    df, _ = YT.add_features(df)
    cube, Y1, Y2, Y3, date, n = YT2.to_cube(df, feats)
    keys = df.sort_values(["race", "lane"])["race"].values[::6]
    print(f"\n検証 {n:,}レース ({args.dfrom}〜)")

    Xf = np.column_stack([cube[c].ravel() for c in feats]).astype(np.float32)
    rr = np.repeat(np.arange(n), 6)
    p1 = YT2.norm_by(m1.predict(Xf).astype(np.float32), rr, n).reshape(n, 6)
    cube["p1"] = p1
    print("1着確率を計算しました")

    # --- 2着・3着を全通り(候補固定なし)で回す ---
    pairs = [(a, b) for a in range(6) for b in range(6) if b != a]
    tri = [(a, b, c) for a, b in pairs for c in range(6) if c not in (a, b)]
    t0 = time.time()
    P2 = np.zeros((n, 6, 6))
    for a in range(6):
        w = np.full(n, a)
        X, _, _, R, L, _ = YT2.build(cube, feats, n, [w])
        v = m2.predict(X).astype(np.float32)
        for k in range(len(R)):
            P2[R[k], a, L[k]] = max(v[k], 1e-9)
    P2 /= P2.sum(2, keepdims=True)
    print(f"  2着 {time.time()-t0:.0f}秒")

    P3 = np.zeros((n, 6, 6, 6))
    for a, b in pairs:
        wa, wb = np.full(n, a), np.full(n, b)
        X, _, _, R, L, _ = YT2.build(cube, feats, n, [wa, wb])
        v = m3.predict(X).astype(np.float32)
        for k in range(len(R)):
            P3[R[k], a, b, L[k]] = max(v[k], 1e-9)
    for a, b in pairs:
        P3[:, a, b] /= P3[:, a, b].sum(1, keepdims=True)
    print(f"  3着 {time.time()-t0:.0f}秒")

    # --- 120通りの確率 ---
    names = [f"{a+1}-{b+1}-{c+1}" for a, b, c in tri]
    CP = np.stack([p1[:, a] * P2[:, a, b] * P3[:, a, b, c] for a, b, c in tri], 1)
    order = np.argsort(-CP, axis=1)

    OD, combos = load_odds(args.dfrom)
    cix = {c: i for i, c in enumerate(combos)}
    ok = np.array([k in OD for k in keys])
    print(f"オッズが揃ったレース {int(ok.sum()):,}")

    hit_rank = np.full(n, -1)
    pay = np.zeros(n)
    cost = np.zeros((n, len(NPTS)))
    ret = np.zeros((n, len(NPTS)))
    for i in np.where(ok)[0]:
        o, hit, pv = OD[keys[i]]
        pay[i] = pv
        top = [names[j] for j in order[i]]
        if hit in top:
            hit_rank[i] = top.index(hit) + 1
        for k, N in enumerate(NPTS):
            cost[i, k] = N * 100
            if 0 < hit_rank[i] <= N:
                ret[i, k] = pv

    sel = ok
    R = pd.DataFrame({
        "date": date[sel], "key": keys[sel], "rank": hit_rank[sel],
        "pay": pay[sel],
        "top_p": p1[sel].max(1), "top_lane": p1[sel].argmax(1) + 1,
        "sum8": np.sort(CP[sel], 1)[:, -8:].sum(1),
        "gap": np.sort(p1[sel], 1)[:, -1] - np.sort(p1[sel], 1)[:, -2],
        "day_no": cube["day_no"][sel][:, 0],
        "a1": (cube["cls_val"][sel] == 4).sum(1),
        "jcd": cube["jcd"][sel][:, 0],
    })
    for k, N in enumerate(NPTS):
        R[f"ret{N}"] = ret[sel, k]
        R[f"cost{N}"] = cost[sel, k]

    def stat(m, N):
        k = int(m.sum())
        if k < 300:
            return None
        c = R.loc[m, f"cost{N}"].sum()
        v = R.loc[m, f"ret{N}"].values
        r = v.sum() / c
        se = np.sqrt(((v - r * N * 100) ** 2).sum()) / c
        hit = ((R.loc[m, "rank"] > 0) & (R.loc[m, "rank"] <= N)).mean()
        return {"n": k, "hit": hit, "roi": r, "se": se, "pl": v.sum() - c}

    print("\n" + "=" * 58)
    print("[1] 全レースで上位N組を買った場合")
    out = []
    for N in NPTS:
        s = stat(pd.Series(True, index=R.index), N)
        out.append([f"上位{N}組", f"{s['n']:,}", pc(s["hit"]), pc(s["roi"]),
                    pc(s["se"]), f"{s['pl']:+,.0f}"])
    tbl(["買い方", "レース", "的中率", "回収率", "誤差±", "収支"], out)

    print("\n  的中したとき、上位何組目だったか")
    rk = R.loc[R["rank"] > 0, "rank"]
    out = []
    for lo, hi in ((1, 1), (2, 3), (4, 6), (7, 8), (9, 12), (13, 120)):
        c = int(((rk >= lo) & (rk <= hi)).sum())
        out.append([f"{lo}〜{hi}位" if lo != hi else "1位", f"{c:,}",
                    pc(c / len(R))])
    out.append(["120組に無し", f"{int((R['rank']<0).sum()):,}",
                pc((R["rank"] < 0).mean())])
    tbl(["順位", "レース", "全体に占める割合"], out)

    print("\n" + "=" * 58)
    print("[2] レースの性質で絞る (上位8組を買った場合)")

    def show(title, col, q=5, labels=None):
        print(f"\n  ◇ {title}")
        if labels:
            gs = [(lab, m) for lab, m in labels]
        else:
            qs = np.unique(np.quantile(R[col], np.linspace(0, 1, q + 1)))
            gs = [(f"{qs[i]:.3g}〜{qs[i+1]:.3g}",
                   (R[col] >= qs[i]) & (R[col] < qs[i + 1] if i < len(qs) - 2
                                        else R[col] <= qs[i + 1]))
                  for i in range(len(qs) - 1)]
        out = []
        for lab, m in gs:
            s = stat(m, 8)
            if s:
                out.append([lab, f"{s['n']:,}", pc(s["hit"]), pc(s["roi"]),
                            pc(s["se"]), f"{s['pl']:+,.0f}"])
        tbl(["区分", "レース", "的中率", "回収率", "誤差±", "収支"], out)

    show("本命の1着確率", "top_p")
    show("上位8組の確率合計", "sum8")
    show("1着と2着の差", "gap")
    show("本命の枠", None, labels=[
        (f"{l}号艇", R["top_lane"] == l) for l in range(1, 7)])
    show("A1の人数", None, labels=[
        (f"{k}人", R["a1"] == k) for k in range(0, 7)])
    show("開催日目", None, labels=[
        (f"{k}日目", R["day_no"] == k) for k in range(1, 8)])

    print("\n" + "=" * 58)
    print("[3] 的中率で絞った場合 (上位8組・確率合計の高い順)")
    out = []
    for frac in (0.05, 0.10, 0.20, 0.30, 0.50):
        th = np.quantile(R["sum8"], 1 - frac)
        s = stat(R["sum8"] >= th, 8)
        if s:
            out.append([f"上位{frac:.0%}", f"{s['n']:,}", pc(s["hit"]),
                        pc(s["roi"]), pc(s["se"]), f"{s['pl']:+,.0f}"])
    tbl(["範囲", "レース", "的中率", "回収率", "誤差±", "収支"], out)

    print("\n  同じ絞り方で買う点数を変える")
    th = np.quantile(R["sum8"], 0.8)
    out = []
    for N in NPTS:
        s = stat(R["sum8"] >= th, N)
        if s:
            out.append([f"上位{N}組", f"{s['n']:,}", pc(s["hit"]), pc(s["roi"]),
                        pc(s["se"]), f"{s['pl']:+,.0f}"])
    tbl(["買い方", "レース", "的中率", "回収率", "誤差±", "収支"], out)

    print("\n" + "=" * 58)
    print("[4] 年度で割って再現するか (確率合計 上位20%・8組)")
    yr = (R["date"] // 10000 - ((R["date"] // 100) % 100 < 5)).astype(int)
    out = []
    for y in sorted(set(yr.tolist())):
        s = stat((R["sum8"] >= th) & (yr == y), 8)
        if s:
            out.append([f"{y}年度", f"{s['n']:,}", pc(s["hit"]), pc(s["roi"]),
                        pc(s["se"])])
    tbl(["年度", "レース", "的中率", "回収率", "誤差±"], out)

    print("\n" + "=" * 58)
    print("判断の目安")
    print("  ・回収率は100%を超えない見込み。絞っても平らになるはず")
    print("  ・的中率で絞るのは意味がある。外れ続けずに遊べる")
    print("  ・年度がそろわない条件は使わない")
    print("  ・締切時オッズでの数字。実運用はこれより落ちる")


if __name__ == "__main__":
    main()
