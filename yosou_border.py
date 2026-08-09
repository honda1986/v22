#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yosou_border.py -- 勝負レースの閾値(上位8組の確率合計)を実データから決める

  yosou_check3.py の結果:
    上位5%  的中率69.8%  回収率83.3%
    上位10% 的中率67.6%  回収率81.9%
    上位20% 的中率63.2%  回収率79.9%
  ここで使う「上位◯%」を、アプリで判定できる絶対値に直す。

  使い方 (Colab)
    !pip -q install lightgbm
    !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
    !rm -rf app && git clone --depth 1 https://github.com/honda1986/boatrace-app.git app
    !mkdir -p yosou_model && cp app/yosou_model/* yosou_model/
    %run v22/yosou_border.py
"""

import json
import os

import numpy as np

import yosou_train as YT
import yosou_train2 as YT2

MODEL = "yosou_model"
DFROM = "20250314"          # 学習に使っていない期間


def main():
    import lightgbm as lgb
    F3 = json.load(open(f"{MODEL}/features3.json", encoding="utf-8"))
    feats = F3["p1"]
    m1 = lgb.Booster(model_file=f"{MODEL}/lgb_p1.txt")
    m2 = lgb.Booster(model_file=f"{MODEL}/lgb_p2.txt")
    m3 = lgb.Booster(model_file=f"{MODEL}/lgb_p3.txt")

    df = YT2.load2()
    df = df[df["date"].astype(str) >= DFROM].copy()
    df, _ = YT.add_features(df)
    cube, Y1, Y2, Y3, date, n = YT2.to_cube(df, feats)
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
        v = m2.predict(X).astype(np.float32)
        P2[R, a, L] = np.maximum(v, 1e-9)
    P2 /= P2.sum(2, keepdims=True)

    P3 = np.zeros((n, 6, 6, 6))
    for a, b in pairs:
        X, _, _, R, L, _ = YT2.build(cube, feats, n,
                                     [np.full(n, a), np.full(n, b)])
        v = m3.predict(X).astype(np.float32)
        P3[R, a, b, L] = np.maximum(v, 1e-9)
    for a, b in pairs:
        P3[:, a, b] /= P3[:, a, b].sum(1, keepdims=True)

    CP = np.stack([p1[:, a] * P2[:, a, b] * P3[:, a, b, c] for a, b, c in tri], 1)
    sum8 = np.sort(CP, 1)[:, -8:].sum(1)
    sum6 = np.sort(CP, 1)[:, -6:].sum(1)

    print("\n" + "=" * 54)
    print("[1] 上位8組の確率合計 の分布")
    for q in (50, 60, 70, 80, 85, 90, 95, 99):
        print(f"  上位{100-q:2d}% の境界 : {np.quantile(sum8, q/100):.4f}")

    print("\n[2] 閾値ごとの、対象になるレースの割合")
    for th in (0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65):
        r = (sum8 >= th).mean()
        print(f"  {th:.2f} 以上 : {r*100:5.1f}%  "
              f"(1日{r*1900:.0f}レースくらい)")

    print("\n[3] おすすめ")
    th10 = float(np.quantile(sum8, 0.90))
    th20 = float(np.quantile(sum8, 0.80))
    print(f"  上位10% の閾値 = {th10:.4f}  → 的中率67.6% / 回収率81.9%")
    print(f"  上位20% の閾値 = {th20:.4f}  → 的中率63.2% / 回収率79.9%")
    print(f"\n  yosou.py に入れる値: SHOBU_TH = {th10:.3f}")
    print(f"  上位6組の確率合計の中央値(参考): {np.median(sum6):.3f}")


if __name__ == "__main__":
    main()
