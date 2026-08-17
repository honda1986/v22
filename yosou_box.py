#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yosou_box.py -- 予想順位の層から1艇ずつ選んで3連単BOXを買う (Colab)

■ 発想(はむさん)
  予想1位を軸に、2〜3位から1艇、4〜6位から1艇を選んで3連単BOX(6点)。
  上位3艇BOXは「本命が飛ぶと全滅」だが、下位を1艇混ぜれば中穴を拾える。

  混戦(本命38%未満)での上位3艇BOXが80.3%だったので、その改良版になるか。

■ 比べる組み方(すべて3艇BOX = 6点)
  1+2+3   上位3艇(既存)
  1+2+4 / 1+3+4 / 1+2+5 / 1+3+5 / 1+2+6 / 1+3+6
  2+3+4   軸を外す(本命が飛ぶ前提)
  1+4+5   下位2艇を混ぜる
  さらに 2×3=6通りの3艇組を全部買う36点も対照として測る

  使い方 (Colab)
    !pip -q install lightgbm
    !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
    !cp -r v22/yosou_model .
    %run v22/yosou_box.py
"""

import argparse
import glob
import gzip
import json
import os
from itertools import permutations

import numpy as np

import yosou_train as YT
import yosou_train2 as YT2

MODEL = "yosou_model"

# (表示名, 使う予想順位) 順位は1始まり
PLANS = [
    ("1+2+3 上位3艇", (1, 2, 3)),
    ("1+2+4", (1, 2, 4)),
    ("1+3+4", (1, 3, 4)),
    ("1+2+5", (1, 2, 5)),
    ("1+3+5", (1, 3, 5)),
    ("1+2+6", (1, 2, 6)),
    ("1+3+6", (1, 3, 6)),
    ("1+4+5", (1, 4, 5)),
    ("2+3+4 軸を外す", (2, 3, 4)),
    ("1+2+3+4 の4艇BOX24点", (1, 2, 3, 4)),
]


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
    NIX = {c: i for i, c in enumerate(names)}
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

    idx = [i for i in range(n) if keys[i] in OD]
    K = len(PLANS)
    cost = np.zeros((len(idx), K))
    ret = np.zeros((len(idx), K))
    hitf = np.zeros((len(idx), K), dtype=bool)
    S8 = np.zeros(len(idx))
    TOPP = np.zeros(len(idx))
    DT = np.zeros(len(idx), dtype=np.int64)

    for r_i, i in enumerate(idx):
        hit, o_all = OD[keys[i]]
        o = o_all[reo]
        hi = NIX[hit]
        S8[r_i] = np.sort(CP[i])[-8:].sum()
        TOPP[r_i] = p1[i].max()
        DT[r_i] = date[i]
        rank = np.argsort(-p1[i]) + 1        # 予想順位1位の艇番 = rank[0]

        for k, (_lab, ranks) in enumerate(PLANS):
            boats = [int(rank[q - 1]) for q in ranks]
            sel = [NIX[f"{a}-{b}-{c}"] for a, b, c in permutations(boats, 3)]
            cost[r_i, k] = len(sel) * 100
            if hi in sel:
                ret[r_i, k] = o[hi] * 100
                hitf[r_i, k] = True

    print(f"突き合わせ {len(idx):,}レース\n")

    def stat(k, m):
        c = cost[m, k]
        v = ret[m, k]
        if len(c) < 300:
            return None
        roi = v.sum() / c.sum()
        se = np.sqrt(((v - roi * c) ** 2).sum()) / c.sum()
        return {"n": len(c), "hit": hitf[m, k].mean(), "roi": roi, "se": se,
                "cost": c.mean(), "pl": v.sum() - c.sum()}

    def show(title, m):
        print(f"\n  ◇ {title}　{int(m.sum()):,}レース")
        out = []
        for k in range(K):
            s = stat(k, m)
            if s:
                out.append([PLANS[k][0], f"{s['cost']:.0f}円", pc(s["hit"]),
                            pc(s["roi"]), pc(s["se"]), f"{s['pl']:+,.0f}"])
        out.sort(key=lambda r: -float(r[3].rstrip("%")))
        tbl(["組み方", "1R投資", "的中率", "回収率", "誤差±", "収支"], out)

    print("=" * 64)
    print("[1] 本命の1着確率で切る  (回収率の高い順に並べています)")
    show("混戦 本命が38%未満", TOPP < 0.38)
    show("中位 本命が38〜55%", (TOPP >= 0.38) & (TOPP < 0.55))
    show("堅い 本命が55%以上", TOPP >= 0.55)
    show("全レース", np.ones(len(idx), dtype=bool))

    print("\n" + "=" * 64)
    print("[2] 確率合計で切る")
    q = np.quantile(S8, [0.2, 0.8])
    show("確率合計 下位20%", S8 <= q[0])
    show("確率合計 上位20%", S8 >= q[1])

    print("\n" + "=" * 64)
    print("[3] 混戦での上位3つを年度で割る")
    yr = (DT // 10000 - ((DT // 100) % 100 < 5)).astype(int)
    mk = TOPP < 0.38
    best = sorted(range(K), key=lambda k: -(stat(k, mk) or {"roi": 0})["roi"])[:3]
    out = []
    for k in best:
        for y in sorted(set(yr.tolist())):
            s = stat(k, mk & (yr == y))
            if s:
                out.append([PLANS[k][0], f"{y}年度", f"{s['n']:,}",
                            pc(s["hit"]), pc(s["roi"]), pc(s["se"])])
    tbl(["組み方", "年度", "レース", "的中率", "回収率", "誤差±"], out)

    print("\n" + "=" * 64)
    print("判断の目安")
    print("  ・混戦で100%を誤差2つ分こえる組み方があれば本物の候補")
    print("  ・10通り×5区分=50通り見ているので、1つ高く出るのは当たり前")
    print("  ・年度がそろわない条件は使わない")
    print("  ・締切時オッズでの数字。実運用はこれより落ちる")


if __name__ == "__main__":
    main()
