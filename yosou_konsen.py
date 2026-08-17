#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yosou_konsen.py -- 混戦レースは別の買い方をした方がいいか (Colab)

■ いまの状況
  勝負レース(確率が上位8組に集中)は 上位8組で的中率67.6%。
  混戦レース(確率が散っている)は 上位8組で的中率28.5%。当たらない。
  混戦は「買わない」で済ませてきたが、買い方を変える余地が一番大きいのもここ。

■ 先に厳しい事実
  19回の検証で、穴を買うほど回収率は下がることが分かっている。
    1〜15番人気 75〜78% / 16〜30番 71.3% / 31〜60番 66.3% / 61番〜 53.5%
  ただしそれは全レースを混ぜた数字。
  「混戦レースに限れば穴が有利」はまだ否定できていない。

■ 比べる買い方
  3連単 上位8点 / 上位16点 / 上位3艇ボックス6点
  3連複 上位1点 / 上位3点          ← 20通りしかないので混戦でも絞れる
  2連単 上位4点
  市場9〜20番人気のうちモデル上位   ← 穴狙い
  本命を2着3着に置く(1着は他)       ← 混戦なら本命が飛ぶ前提

  ★3連複のオッズは持っていないので、3連単6通りの和から合成する
    (控除率が同じなら近似として使える)

  使い方 (Colab)
    !pip -q install lightgbm
    !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
    !cp -r v22/yosou_model .
    %run v22/yosou_konsen.py
"""

import argparse
import glob
import gzip
import json
import os
from itertools import combinations, permutations

import numpy as np

import yosou_train as YT
import yosou_train2 as YT2

MODEL = "yosou_model"


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

    # 3連複(20通り) と 2連単(30通り) の索引を作る
    TRIO = [tuple(sorted(z)) for z in combinations(range(1, 7), 3)]
    TRIO_IX = {t: i for i, t in enumerate(TRIO)}
    trio_members = [[NIX[f"{a}-{b}-{c}"] for a, b, c in permutations(t)]
                    for t in TRIO]
    NIREN = [(a, b) for a in range(1, 7) for b in range(1, 7) if b != a]
    NIREN_IX = {t: i for i, t in enumerate(NIREN)}
    niren_members = [[NIX[f"{a}-{b}-{c}"] for c in range(1, 7)
                      if c not in (a, b)] for a, b in NIREN]

    idx = [i for i in range(n) if keys[i] in OD]
    print(f"突き合わせ {len(idx):,}レース\n")

    # 各買い方の (投資, 払戻, 的中) を集める
    LABEL = ["3連単 上位8点", "3連単 上位16点", "3連単 上位3艇BOX6点",
             "3連複 上位1点", "3連複 上位3点", "2連単 上位4点",
             "穴 市場9〜20番人気×モデル上位6点", "本命を2〜3着に置く8点"]
    K = len(LABEL)
    cost = np.zeros((len(idx), K))
    ret = np.zeros((len(idx), K))
    hitf = np.zeros((len(idx), K), dtype=bool)
    S8 = np.zeros(len(idx))
    TOPP = np.zeros(len(idx))
    DT = np.zeros(len(idx), dtype=np.int64)

    for r_i, i in enumerate(idx):
        hit, o_all = OD[keys[i]]
        o = o_all[reo]
        p = CP[i]
        hi = NIX[hit]
        S8[r_i] = np.sort(p)[-8:].sum()
        TOPP[r_i] = p1[i].max()
        DT[r_i] = date[i]
        order = np.argsort(-p)
        nk = np.empty(120, dtype=np.int32)
        nk[np.argsort(o)] = np.arange(1, 121)

        def put(k, sel):
            sel = list(sel)
            if not sel:
                return
            cost[r_i, k] = len(sel) * 100
            if hi in sel:
                ret[r_i, k] = o[hi] * 100
                hitf[r_i, k] = True

        put(0, order[:8])
        put(1, order[:16])
        # 上位3艇ボックス
        top3 = np.argsort(-p1[i])[:3] + 1
        put(2, [NIX[f"{a}-{b}-{c}"] for a, b, c in permutations(top3)])
        # 3連複(合成オッズ)
        tp = np.array([p[m].sum() for m in trio_members])
        to = np.array([1.0 / (1.0 / o[m]).sum() for m in trio_members])
        thit = TRIO_IX[tuple(sorted(int(x) for x in hit.split("-")))]
        for k, N in ((3, 1), (4, 3)):
            sel = np.argsort(-tp)[:N]
            cost[r_i, k] = N * 100
            if thit in sel:
                ret[r_i, k] = to[thit] * 100
                hitf[r_i, k] = True
        # 2連単(合成オッズ)
        np_ = np.array([p[m].sum() for m in niren_members])
        no = np.array([1.0 / (1.0 / o[m]).sum() for m in niren_members])
        a, b, _c = (int(x) for x in hit.split("-"))
        nhit = NIREN_IX[(a, b)]
        sel = np.argsort(-np_)[:4]
        cost[r_i, 5] = 4 * 100
        if nhit in sel:
            ret[r_i, 5] = no[nhit] * 100
            hitf[r_i, 5] = True
        # 穴: 市場9〜20番人気のうち、モデルの確率が高い6点
        cand = np.where((nk >= 9) & (nk <= 20))[0]
        put(6, cand[np.argsort(-p[cand])[:6]])
        # 本命を2〜3着に置く(1着は本命以外)
        tl = int(np.argmax(p1[i])) + 1
        cand2 = [j for j, nm in enumerate(names)
                 if not nm.startswith(f"{tl}-") and f"{tl}" in nm.split("-")[1:]]
        put(7, sorted(cand2, key=lambda j: -p[j])[:8])

    def stat(k, m):
        c = cost[m, k]
        v = ret[m, k]
        use = c > 0
        if use.sum() < 300:
            return None
        roi = v[use].sum() / c[use].sum()
        se = np.sqrt(((v[use] - roi * c[use]) ** 2).sum()) / c[use].sum()
        return {"n": int(use.sum()), "hit": hitf[m, k][use].mean(),
                "roi": roi, "se": se, "cost": c[use].mean(),
                "pl": v[use].sum() - c[use].sum()}

    def show(title, m):
        print(f"\n  ◇ {title}　{int(m.sum()):,}レース")
        out = []
        for k in range(K):
            s = stat(k, m)
            if s:
                out.append([LABEL[k], f"{s['n']:,}", f"{s['cost']:.0f}円",
                            pc(s["hit"]), pc(s["roi"]), pc(s["se"]),
                            f"{s['pl']:+,.0f}"])
        tbl(["買い方", "レース", "1R投資", "的中率", "回収率", "誤差±", "収支"],
            out)

    print("=" * 66)
    print("[1] レースの性質ごとに、買い方を比べる")
    q = np.quantile(S8, [0.2, 0.5, 0.8])
    show("混戦(確率合計 下位20%)", S8 <= q[0])
    show("中位(確率合計 20〜80%)", (S8 > q[0]) & (S8 < q[2]))
    show("堅い(確率合計 上位20%)", S8 >= q[2])
    show("全レース", np.ones(len(idx), dtype=bool))

    print("\n" + "=" * 66)
    print("[2] 本命の1着確率で切る")
    show("本命が38%未満(混戦)", TOPP < 0.38)
    show("本命が38〜55%(中位)", (TOPP >= 0.38) & (TOPP < 0.55))
    show("本命が55%以上(堅い)", TOPP >= 0.55)

    print("\n" + "=" * 66)
    print("[3] 混戦での上位3つを年度で割る")
    yr = (DT // 10000 - ((DT // 100) % 100 < 5)).astype(int)
    mk = S8 <= q[0]
    best = sorted(range(K), key=lambda k: -(stat(k, mk) or {"roi": 0})["roi"])[:3]
    out = []
    for k in best:
        for y in sorted(set(yr.tolist())):
            s = stat(k, mk & (yr == y))
            if s:
                out.append([LABEL[k], f"{y}年度", f"{s['n']:,}",
                            pc(s["hit"]), pc(s["roi"]), pc(s["se"])])
    tbl(["買い方", "年度", "レース", "的中率", "回収率", "誤差±"], out)

    print("\n" + "=" * 66)
    print("判断の目安")
    print("  ・混戦で100%を誤差2つ分こえる買い方があれば本物の候補")
    print("  ・8通り×3区分=24通り見ているので、1つ高く出るのは当たり前")
    print("  ・年度がそろわない条件は使わない")
    print("  ・3連複と2連単は3連単から合成したオッズ。実際とは少しずれます")
    print("  ・締切時オッズでの数字。実運用はこれより落ちる")


if __name__ == "__main__":
    main()
