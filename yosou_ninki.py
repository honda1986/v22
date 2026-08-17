#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yosou_ninki.py -- 5〜8番人気で決着する買い目を切り出せるか (Colab)

■ 発想(はむさん)
  決着した組の人気の分布を見ると
    1番人気    10.8%  平均  715円
    2〜3番人気  15.6%  平均  994円
    4〜8番人気  22.6%  平均1,713円   ← いちばん多く、配当も悪くない
    9〜20番人気 23.3%  平均3,772円
  4〜8番人気を安定して当てられれば、8点で800円投資して1,713円になる。

  そして「払戻15,000〜20,000円が効いた」理由も、たぶんここ。
  あの条件は結果的に中位人気を狙う形になっている。

■ これまでとの違い
  EVで絞る検証は何度もやって全部だめだった。
  今回は「モデルの順位で下位を切り出す」。
  上位1〜2組(=市場の1〜3番人気)を捨てて、3位以下だけを買う形。
  これは一度も試していない。

■ 測ること
  ・モデルの順位帯ごとの回収率(1〜2位 / 3〜5位 / 6〜10位 / 11〜15位)
  ・市場人気の帯ごとの回収率
  ・両方を掛け合わせた表 …「モデルは上位、市場は中位」のマスを探す

  使い方 (Colab)
    !pip -q install lightgbm
    !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
    !cp -r v22/yosou_model .
    %run v22/yosou_ninki.py
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
    reo = np.array([cidx[c] for c in names])   # tri順に並べ替える索引

    # 全点を1行ずつに展開する
    MR, NK, OO, HH, DT, S8 = [], [], [], [], [], []
    for i in range(n):
        if keys[i] not in OD:
            continue
        hit, o_all = OD[keys[i]]
        o = o_all[reo]
        mr = np.empty(120, dtype=np.int16)
        mr[np.argsort(-CP[i])] = np.arange(1, 121)      # モデルの順位
        nk = np.empty(120, dtype=np.int16)
        nk[np.argsort(o)] = np.arange(1, 121)           # 市場の人気順
        h = np.zeros(120, dtype=bool)
        h[names.index(hit)] = True
        MR.append(mr); NK.append(nk); OO.append(o); HH.append(h)
        DT.append(np.full(120, date[i]))
        S8.append(np.full(120, np.sort(CP[i])[-8:].sum()))
    MR = np.concatenate(MR); NK = np.concatenate(NK)
    OO = np.concatenate(OO); HH = np.concatenate(HH)
    DT = np.concatenate(DT); S8 = np.concatenate(S8)
    nr = len(MR) // 120
    print(f"突き合わせ {nr:,}レース / {len(MR):,}点\n")

    def stat(m, minn=500):
        k = int(m.sum())
        if k < minn:
            return None
        ret = (OO[m] * HH[m] * 100).sum()
        cost = k * 100
        roi = ret / cost
        v = OO[m] * HH[m] * 100
        se = np.sqrt(((v - roi * 100) ** 2).sum()) / cost
        return {"n": k, "hit": HH[m].mean(), "roi": roi, "se": se,
                "pl": ret - cost, "o": OO[m].mean()}

    print("=" * 62)
    print("[1] モデルの順位で切る  (その順位の1点だけを買い続けた場合)")
    out = []
    for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 12),
                   (13, 20), (21, 40), (41, 120)):
        s = stat((MR >= lo) & (MR <= hi))
        if s:
            out.append([f"{lo}位" if lo == hi else f"{lo}〜{hi}位",
                        f"{s['n']:,}", pc(s["hit"]), f"{s['o']:.1f}倍",
                        pc(s["roi"]), pc(s["se"])])
    tbl(["モデルの順位", "点数", "的中率", "平均オッズ", "回収率", "誤差±"], out)

    print("\n[2] 市場の人気で切る  (対照)")
    out = []
    for lo, hi in ((1, 1), (2, 3), (4, 8), (9, 15), (16, 30), (31, 60),
                   (61, 120)):
        s = stat((NK >= lo) & (NK <= hi))
        if s:
            out.append([f"{lo}番人気" if lo == hi else f"{lo}〜{hi}番人気",
                        f"{s['n']:,}", pc(s["hit"]), f"{s['o']:.1f}倍",
                        pc(s["roi"]), pc(s["se"])])
    tbl(["市場の人気", "点数", "的中率", "平均オッズ", "回収率", "誤差±"], out)

    print("\n" + "=" * 62)
    print("[3] モデルの順位 × 市場の人気  (回収率±誤差、カッコは点数)")
    print("  『モデルは上位、市場は中位』のマスに歪みがあるはず")
    mb = [(1, 2), (3, 5), (6, 8), (9, 12), (13, 20)]
    nb = [(1, 3), (4, 8), (9, 15), (16, 30), (31, 120)]
    hd = ["モデル\\市場"] + [f"{a}〜{b}番" for a, b in nb]
    out = []
    for ma, mz in mb:
        line = [f"{ma}〜{mz}位"]
        for na, nz in nb:
            s = stat((MR >= ma) & (MR <= mz) & (NK >= na) & (NK <= nz), 300)
            line.append(f"{s['roi']*100:.0f}±{s['se']*100:.0f}({s['n']:,})"
                        if s else "—")
        out.append(line)
    tbl(hd, out)

    print("\n[4] 同じ表を的中率で")
    out = []
    for ma, mz in mb:
        line = [f"{ma}〜{mz}位"]
        for na, nz in nb:
            s = stat((MR >= ma) & (MR <= mz) & (NK >= na) & (NK <= nz), 300)
            line.append(f"{s['hit']*100:.1f}%" if s else "—")
        out.append(line)
    tbl(hd, out)

    print("\n" + "=" * 62)
    print("[5] 『モデル上位8組のうち、市場4〜8番人気の点だけ』を買う")
    base = MR <= 8
    out = []
    for lab, m in (("上位8組すべて", base),
                   ("うち市場1〜3番人気", base & (NK <= 3)),
                   ("うち市場4〜8番人気", base & (NK >= 4) & (NK <= 8)),
                   ("うち市場9番人気以降", base & (NK >= 9))):
        s = stat(m)
        if s:
            out.append([lab, f"{s['n']:,}", f"{s['n']/nr:.1f}点/R",
                        pc(s["hit"]), f"{s['o']:.1f}倍", pc(s["roi"]),
                        pc(s["se"]), f"{s['pl']:+,.0f}"])
    tbl(["買い方", "点数", "1Rあたり", "的中率", "平均オッズ", "回収率",
         "誤差±", "収支"], out)

    print("\n[6] 勝負レース(確率合計>=0.553)に限ってもう一度")
    sh = S8 >= 0.553
    out = []
    for lab, m in (("上位8組すべて", base & sh),
                   ("うち市場1〜3番人気", base & sh & (NK <= 3)),
                   ("うち市場4〜8番人気", base & sh & (NK >= 4) & (NK <= 8)),
                   ("うち市場9番人気以降", base & sh & (NK >= 9))):
        s = stat(m, 300)
        if s:
            out.append([lab, f"{s['n']:,}", pc(s["hit"]), f"{s['o']:.1f}倍",
                        pc(s["roi"]), pc(s["se"]), f"{s['pl']:+,.0f}"])
    tbl(["買い方", "点数", "的中率", "平均オッズ", "回収率", "誤差±", "収支"],
        out)

    print("\n[7] 年度で割る(モデル上位8組 × 市場4〜8番人気)")
    yr = (DT // 10000 - ((DT // 100) % 100 < 5)).astype(int)
    sel = base & (NK >= 4) & (NK <= 8)
    out = []
    for y in sorted(set(yr.tolist())):
        s = stat(sel & (yr == y), 300)
        if s:
            out.append([f"{y}年度", f"{s['n']:,}", pc(s["hit"]),
                        pc(s["roi"]), pc(s["se"])])
    tbl(["年度", "点数", "的中率", "回収率", "誤差±"], out)

    print("\n" + "=" * 62)
    print("判断の目安")
    print("  ・[3]で100%を超えるマスがあり、点数も十分なら本物の候補")
    print("  ・[5][6]で100%を誤差2つ分こえるかを見る")
    print("  ・25マス見ているので、1つくらい高く出るのは当たり前")
    print("  ・年度がそろわない条件は使わない")
    print("  ・締切時オッズでの数字。実運用はこれより落ちる")


if __name__ == "__main__":
    main()
