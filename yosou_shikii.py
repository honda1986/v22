#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yosou_shikii.py -- 「予想確率◯%以上の組だけ買う」を検証する (Colab)

■ 発想(はむさん)
  上位8点と点数を固定すると、レースによって質が違う。
  堅いレースなら8点目でも確率5%あるが、荒れそうなレースの8点目は1%以下。
  その1%の組を買うのが回収率を下げているのではないか。

  確率で切れば買う点数が自然に変わる。
    堅いレース   確率5%以上が12点 → 1,200円
    荒れるレース 確率5%以上が3点  →   300円
  荒れそうなレースでは自然に少額になる。資金配分としても筋が通っている。

■ 比べるもの
  A 上位N点固定(3/6/8/12点)
  B 確率◯%以上(3/5/8/10/12%)
  それぞれ「全レース」と「勝負レース(確率合計>=0.553)」の両方で測る。

  使い方 (Colab)
    !pip -q install lightgbm
    !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
    !cp -r v22/yosou_model .
    %run v22/yosou_shikii.py
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
SHOBU_TH = 0.553


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
    ap.add_argument("--th", type=float, default=SHOBU_TH)
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
            OD[f"{d}-{r['jcd']:02d}-{r['rno']}"] = (r["hit"], r["pay_3t"])

    ok = np.array([k in OD for k in keys])
    idx = np.where(ok)[0]
    HI = np.array([names.index(OD[keys[i]][0]) for i in idx])
    PAY = np.array([OD[keys[i]][1] for i in idx], dtype=float)
    P = CP[idx]
    DT = date[idx]
    sum8 = np.sort(P, 1)[:, -8:].sum(1)
    shobu = sum8 >= args.th
    m = len(idx)
    print(f"突き合わせ {m:,}レース  "
          f"勝負レース {int(shobu.sum()):,}件 ({shobu.mean()*100:.1f}%)\n")

    order = np.argsort(-P, axis=1)
    rank_of_hit = np.array([int(np.where(order[i] == HI[i])[0][0]) + 1
                            for i in range(m)])

    def by_points(N, mask):
        """上位N点を買う"""
        sel = mask
        k = int(sel.sum())
        won = (rank_of_hit <= N) & sel
        ret = PAY[won].sum()
        cost = k * N * 100
        return k, N * 1.0, won.sum() / k, ret, cost, PAY[won]

    def by_prob(th, mask):
        """確率th以上の組を買う。点数はレースごとに変わる"""
        cnt = (P >= th).sum(1)
        sel = mask & (cnt > 0)
        k = int(sel.sum())
        if k < 300:
            return None
        hit_p = P[np.arange(m), HI]
        won = sel & (hit_p >= th)
        ret = PAY[won].sum()
        cost = cnt[sel].sum() * 100
        return k, cnt[sel].mean(), won.sum() / k, ret, cost, PAY[won]

    def row(lab, res):
        """1レースあたりの払戻を並べて、回収率の誤差を出す"""
        if res is None:
            return None
        k, pts, hit, ret, cost, w = res
        roi = ret / cost
        per = np.zeros(k)                 # 1レースごとの払戻(外れは0)
        per[:len(w)] = w
        avg_cost = cost / k
        se = np.sqrt(((per - roi * avg_cost) ** 2).sum()) / cost
        return [lab, f"{k:,}", f"{pts:.1f}", pc(hit), pc(roi), pc(se),
                f"{avg_cost:,.0f}円", f"{ret-cost:+,.0f}"]

    H = ["買い方", "レース", "平均点数", "的中率", "回収率", "誤差±",
         "1R投資", "収支"]

    for title, mask in (("全レース", np.ones(m, dtype=bool)),
                        (f"勝負レース(確率合計>={args.th})", shobu)):
        print("=" * 64)
        print(f"[{title}]  {int(mask.sum()):,}レース")
        print("\n  ◇ 上位N点を固定して買う")
        out = []
        for N in (3, 6, 8, 12):
            r = row(f"上位{N}点", by_points(N, mask))
            if r:
                out.append(r)
        tbl(H, out)

        print("\n  ◇ 予想確率が◯%以上の組を買う (点数はレースごとに変わる)")
        out = []
        for th in (0.03, 0.05, 0.08, 0.10, 0.12, 0.15):
            r = row(f"{th*100:.0f}%以上", by_prob(th, mask))
            if r:
                out.append(r)
        tbl(H, out)
        print()

    print("=" * 64)
    print("[確率5%以上] を年度で割る")
    yr = (DT // 10000 - ((DT // 100) % 100 < 5)).astype(int)
    out = []
    for lab, mask in (("全レース", np.ones(m, dtype=bool)), ("勝負レース", shobu)):
        for y in sorted(set(yr.tolist())):
            r = row(f"{lab} {y}年度", by_prob(0.05, mask & (yr == y)))
            if r:
                out.append(r)
    tbl(H, out)

    print("\n[参考] 買った点数の分布 (確率5%以上・全レース)")
    cnt = (P >= 0.05).sum(1)
    out = []
    for lo, hi in ((0, 0), (1, 2), (3, 4), (5, 6), (7, 8), (9, 12), (13, 120)):
        c = int(((cnt >= lo) & (cnt <= hi)).sum())
        if c:
            out.append([f"{lo}〜{hi}点" if lo != hi else "0点(買えない)",
                        f"{c:,}", pc(c / m)])
    tbl(["点数", "レース", "割合"], out)

    print("\n" + "=" * 64)
    print("判断の目安")
    print("  ・上位N点固定と、確率◯%以上で、回収率に差が出るか")
    print("  ・100%を誤差2つ分こえ、年度もそろって初めて意味がある")
    print("  ・締切時オッズでの数字。実運用はこれより落ちる")


if __name__ == "__main__":
    main()
