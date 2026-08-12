#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yosou_x15b.py -- 「勝負レース条件」と「払戻の水準」のどちらが効いているのか (Colab)

■ 前回の反省
  条件を緩めて母数を増やしたら回収率が下がった。
  ただしそれは「当たりにくいレースを混ぜたから」であって、
  1,333件の89.6%が上振れだった証拠にはならない。

  正しくは「追加された分だけ」を取り出して比べる。
    0.553以上だけ           1,333件  89.6%
    0.520〜0.553 の追加分    1,775件  ??%   ← これが知りたい

  追加分も89%なら払戻の水準が効いている。
  追加分が81%なら勝負レース条件が効いている。

■ あわせて
  払戻の水準を、勝負レースの内と外で別々に見る(交互作用の確認)。
  年度ごとに独立して再現するかも見る。

  使い方 (Colab)
    !pip -q install lightgbm
    !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
    !cp -r v22/yosou_model .
    %run v22/yosou_x15b.py
"""

import argparse
import glob
import gzip
import json
import os

import numpy as np

import yosou_train as YT
import yosou_train2 as YT2

MODEL = "yosou_model"
BUDGET = 10000.0
TH_P = 0.05


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

    X = np.full(m, np.nan)
    won = np.zeros(m, dtype=bool)
    for i in range(m):
        sel = np.where(P[i] >= TH_P)[0]
        if len(sel) == 0:
            continue
        X[i] = BUDGET / (1.0 / ODS[i][sel]).sum()
        won[i] = HI[i] in sel
    have = ~np.isnan(X)
    print(f"突き合わせ {m:,}レース  買えるレース {int(have.sum()):,}\n")

    def stat(mask, minn=200):
        mm = mask & have
        k = int(mm.sum())
        if k < minn:
            return None
        v = X[mm] * won[mm]
        cost = k * BUDGET
        roi = v.sum() / cost
        se = np.sqrt(((v - roi * BUDGET) ** 2).sum()) / cost
        return {"n": k, "hit": won[mm].mean(), "roi": roi, "se": se,
                "X": X[mm].mean(), "pl": v.sum() - cost}

    W = (X >= 15000) & (X < 20000)          # 払戻15,000〜20,000円

    print("=" * 62)
    print("[1] 確率合計の帯ごとに切り出す (払戻15,000〜20,000円・確率5%以上)")
    print("  緩めた分だけを取り出して比べる。重ねて見ない")
    out = []
    bands = [(0.553, 9.9), (0.520, 0.553), (0.500, 0.520), (0.450, 0.500),
             (0.400, 0.450), (0.0, 0.400)]
    for lo, hi in bands:
        s = stat(W & (S8 >= lo) & (S8 < hi), 100)
        if s:
            out.append([f"{lo:.3f}〜{hi:.3f}" if hi < 9 else f"{lo:.3f}以上",
                        f"{s['n']:,}", pc(s["hit"]), pc(s["roi"]),
                        pc(s["se"]), f"{s['pl']:+,.0f}"])
    tbl(["確率合計の帯", "レース", "的中率", "回収率", "誤差±", "収支"], out)
    print("  上の帯だけ高ければ勝負レース条件が効いている")
    print("  どの帯も同じくらいなら払戻の水準だけが効いている")

    print("\n" + "=" * 62)
    print("[2] 払戻の水準 × 勝負レースの内外 (交互作用)")
    hd = ["払戻", "勝負(0.553以上)", "0.45〜0.553", "0.45未満"]
    out = []
    for lo, hi in ((10000, 13000), (13000, 15000), (15000, 17000),
                   (17000, 20000), (20000, 25000), (25000, 10**9)):
        line = [f"{lo//1000}〜{hi//1000}千円" if hi < 10**8 else "25千円〜"]
        for a, b in ((0.553, 9.9), (0.450, 0.553), (0.0, 0.450)):
            s = stat((X >= lo) & (X < hi) & (S8 >= a) & (S8 < b), 100)
            line.append(f"{s['roi']*100:.0f}±{s['se']*100:.0f}({s['n']})"
                        if s else "—")
        out.append(line)
    tbl(hd, out)

    print("\n" + "=" * 62)
    print("[3] 勝負レースの中で、払戻の水準を切る")
    out = []
    base = S8 >= 0.553
    for lo, hi in ((0, 13000), (13000, 15000), (15000, 17000),
                   (17000, 20000), (20000, 10**9)):
        s = stat(base & (X >= lo) & (X < hi), 100)
        if s:
            out.append([f"{lo//1000}〜{hi//1000}千円" if hi < 10**8 else "20千円〜",
                        f"{s['n']:,}", pc(s["hit"]), pc(s["roi"]),
                        pc(s["se"]), f"{s['pl']:+,.0f}"])
    tbl(["払戻", "レース", "的中率", "回収率", "誤差±", "収支"], out)

    print("\n[4] 勝負レースの外で、同じ切り方をする (対照)")
    out = []
    for lo, hi in ((0, 13000), (13000, 15000), (15000, 17000),
                   (17000, 20000), (20000, 10**9)):
        s = stat(~base & (X >= lo) & (X < hi), 100)
        if s:
            out.append([f"{lo//1000}〜{hi//1000}千円" if hi < 10**8 else "20千円〜",
                        f"{s['n']:,}", pc(s["hit"]), pc(s["roi"]), pc(s["se"])])
    tbl(["払戻", "レース", "的中率", "回収率", "誤差±"], out)
    print("  外でも同じ山ができるなら、払戻の水準そのものが効いている")

    print("\n" + "=" * 62)
    print("[5] 年度で独立に再現するか (勝負レース × 15〜20千円)")
    yr = (DT // 10000 - ((DT // 100) % 100 < 5)).astype(int)
    out = []
    tot = stat(base & W, 100)
    for y in sorted(set(yr.tolist())):
        s = stat(base & W & (yr == y), 80)
        if s:
            out.append([f"{y}年度", f"{s['n']:,}", pc(s["hit"]), pc(s["roi"]),
                        pc(s["se"]), f"{s['pl']:+,.0f}"])
    if tot:
        out.append(["全体", f"{tot['n']:,}", pc(tot["hit"]), pc(tot["roi"]),
                    pc(tot["se"]), f"{tot['pl']:+,.0f}"])
    tbl(["年度", "レース", "的中率", "回収率", "誤差±", "収支"], out)

    print("\n[6] 場で割る (勝負レース × 15〜20千円)")
    VEN = {1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川",
           6: "浜名湖", 7: "蒲郡", 8: "常滑", 9: "津", 10: "三国",
           11: "びわこ", 12: "住之江", 13: "尼崎", 14: "鳴門", 15: "丸亀",
           16: "児島", 17: "宮島", 18: "徳山", 19: "下関", 20: "若松",
           21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村"}
    jc = cube["jcd"][idx][:, 0].astype(int)
    out = []
    for j in sorted(set(jc.tolist())):
        s = stat(base & W & (jc == j), 60)
        if s:
            out.append([VEN.get(j, j), f"{s['n']:,}", pc(s["hit"]),
                        pc(s["roi"]), pc(s["se"])])
    out.sort(key=lambda r: -float(r[3].rstrip("%")))
    tbl(["場", "レース", "的中率", "回収率", "誤差±"], out[:10])
    print("  上位だけ見ています。場ごとに大きく違うならノイズの可能性")

    print("\n" + "=" * 62)
    print("判断")
    print("  ・[1]で上の帯だけ高ければ勝負レース条件が本物")
    print("  ・[4]の外でも山ができるなら払戻の水準が本物")
    print("  ・[5]で年度がそろい、かつ100%を誤差2つ分こえるかを見る")
    print("  ・払戻6通り×確率合計6通りから選んでいることを忘れない")


if __name__ == "__main__":
    main()
