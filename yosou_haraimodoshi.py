#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yosou_haraimodoshi.py -- 「当たれば◯円」の水準でレースを選んだらどうなるか (Colab)

■ 考え方
  払戻を均等にする配分では、当たったときの払戻が
      X = 10000 / Σ(1/o_i)
  で決まる。X が大きいレースだけ買えば、当たったときの見返りは増える。
  ただしオッズが高い = 当たりにくい ので、回収率は変わらないはず。

  それを実測で確かめる。X の水準ごとに 的中率・回収率・収支 を出す。

■ 対象
  勝負レース(上位8組の確率合計 >= 0.553)、上位8組、1レース1万円

  使い方 (Colab)
    !pip -q install lightgbm
    !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
    !cp -r v22/yosou_model .
    %run v22/yosou_haraimodoshi.py
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
    ap.add_argument("--th", type=float, default=0.0,
                    help="確率合計の下限。0なら全レース")
    ap.add_argument("--pts", type=int, default=8)
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
    cidx = {c: i for i, c in enumerate(combos)}
    sum8 = np.sort(CP, 1)[:, -8:].sum(1)
    N = args.pts

    rows = []
    for i in range(n):
        if keys[i] not in OD or sum8[i] < args.th:
            continue
        o_all, hit = OD[keys[i]]
        ordi = np.argsort(-CP[i])[:N]
        cn = [names[j] for j in ordi]
        od = np.array([o_all[cidx[c]] for c in cn])
        X = BUDGET / (1.0 / od).sum()          # 当たったときの払戻(均等配分)
        won = hit in cn
        rows.append({"date": date[i], "X": X, "won": won,
                     "ret": X if won else 0.0,
                     "sum8": sum8[i], "minodds": od.min(), "maxodds": od.max()})
    R = pd.DataFrame(rows)
    print(f"対象 {len(R):,}件 (確率合計の下限 {args.th})\n")

    def stat(m, minn=200):
        k = int(m.sum())
        if k < minn:
            return None
        v = R.loc[m, "ret"].values
        cost = k * BUDGET
        roi = v.sum() / cost
        se = np.sqrt(((v - roi * BUDGET) ** 2).sum()) / cost
        return {"n": k, "hit": R.loc[m, "won"].mean(), "roi": roi, "se": se,
                "X": R.loc[m, "X"].mean(), "pl": v.sum() - cost}

    print("=" * 58)
    print(f"[1] 当たったときの払戻(X)の水準で分ける  上位{N}組・払戻均等・1万円")
    out = []
    for lo, hi in ((0, 11000), (11000, 13000), (13000, 15000), (15000, 18000),
                   (18000, 22000), (22000, 30000), (30000, 10**9)):
        m = (R["X"] >= lo) & (R["X"] < hi)
        s = stat(m)
        if s:
            out.append([f"{lo:,}〜{hi:,}円" if hi < 10**8 else "30,000円〜",
                        f"{s['n']:,}", f"{s['X']:,.0f}", pc(s["hit"]),
                        pc(s["roi"]), pc(s["se"]), f"{s['pl']:+,.0f}"])
    tbl(["払戻の水準", "レース", "平均払戻", "的中率", "回収率", "誤差±", "収支"], out)

    print(f"\n[2] 『X以上のレースだけ買う』とどうなるか")
    out = []
    for th in (0, 12000, 15000, 18000, 20000, 25000, 30000):
        s = stat(R["X"] >= th)
        if s:
            out.append([f"{th:,}円以上" if th else "全部", f"{s['n']:,}",
                        f"{s['X']:,.0f}", pc(s["hit"]), pc(s["roi"]),
                        pc(s["se"]), f"{s['pl']:+,.0f}"])
    tbl(["条件", "レース", "平均払戻", "的中率", "回収率", "誤差±", "収支"], out)

    print("\n[3] 払戻15,000円以上を、確率合計の条件別に分ける")
    print("  条件を緩めるほど件数が増える。回収率が保たれるかを見る")
    out = []
    for th in (0.0, 0.40, 0.45, 0.50, 0.553):
        m = (R["X"] >= 15000) & (R["sum8"] >= th)
        s = stat(m, 150)
        if s:
            out.append([f"{th:.2f}以上" if th else "条件なし", f"{s['n']:,}",
                        f"{s['X']:,.0f}", pc(s["hit"]), pc(s["roi"]),
                        pc(s["se"]), f"{s['pl']:+,.0f}"])
    tbl(["確率合計", "レース", "平均払戻", "的中率", "回収率", "誤差±", "収支"], out)

    print("\n[4] 払戻の水準 × 確率合計  (回収率のみ)")
    ths = [0.0, 0.40, 0.45, 0.50, 0.553]
    head = ["払戻"] + [f"{t:.2f}〜" if t else "全部" for t in ths]
    out = []
    for lo, hi in ((12000, 15000), (15000, 18000), (18000, 25000),
                   (25000, 10**9)):
        line = [f"{lo//1000}〜{hi//1000}千円" if hi < 10**8 else "25千円〜"]
        for t in ths:
            s = stat((R["X"] >= lo) & (R["X"] < hi) & (R["sum8"] >= t), 150)
            line.append(f"{s['roi']*100:.0f}±{s['se']*100:.0f} ({s['n']})"
                        if s else "—")
        out.append(line)
    tbl(head, out)

    print("\n[5] 払戻15,000円以上(条件なし)を年度で割る")
    yr = (R["date"] // 10000 - ((R["date"] // 100) % 100 < 5)).astype(int)
    out = []
    for y in sorted(set(yr.tolist())):
        s = stat((R["X"] >= 15000) & (yr == y), 100)
        if s:
            out.append([f"{y}年度", f"{s['n']:,}", pc(s["hit"]), pc(s["roi"]),
                        pc(s["se"])])
    tbl(["年度", "レース", "的中率", "回収率", "誤差±"], out)

    print("\n[6] 参考: 8組の最安オッズ別")
    out = []
    qs = np.quantile(R["minodds"], [0, .2, .4, .6, .8, 1])
    for i in range(len(qs) - 1):
        m = (R["minodds"] >= qs[i]) & (R["minodds"] < qs[i + 1]
                                       if i < len(qs) - 2 else R["minodds"] <= qs[i + 1])
        s = stat(m)
        if s:
            out.append([f"{qs[i]:.1f}〜{qs[i+1]:.1f}倍", f"{s['n']:,}",
                        f"{s['X']:,.0f}", pc(s["hit"]), pc(s["roi"]), pc(s["se"])])
    tbl(["最安オッズ", "レース", "平均払戻", "的中率", "回収率", "誤差±"], out)

    print("\n" + "=" * 58)
    print("判断の目安")
    print("  ・回収率がどの水準も同じなら、払戻の大きさでは選べない")
    print("  ・100%を誤差2つ分こえ、年度もそろって初めて意味がある")
    print("  ・締切時オッズでの数字。実運用はこれより落ちる")


if __name__ == "__main__":
    main()
