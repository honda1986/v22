#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yosou_2t.py -- 予想モデルの確率を2連単に当ててみる (Google Colab)

■ すでに分かっていること
  v23_2t.py の検証(7,954レース):
    2連単30点の全部買い 58.3% / 3連単の全部買い 59.3%
    3連単プールと2連単プールの確率の相関 0.9807
    EVを上げるほど回収率が下がる。前半50.3% 後半106.3%で再現性なし
  → 2つのプールの間に歪みは無かった。

■ 今回試すこと
  前回は「3連単プールの確率」を2連単に当てた。
  今回は「予想モデルの確率」を当てる。モデルは2連単プールを見ていないので別の検定。
  1着の確率 p から、a→b の確率を p_a × p_b/(1-p_a) で作る(標準的な近似)。

  さらに、モデルの本命が1号艇でないレースだけに絞った場合も見る。

  使い方 (Colab)
    !pip -q install lightgbm
    !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
    %run v22/yosou_2t.py
"""

import argparse
import gzip
import glob
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

import yosou_train as YT
import v23_2t as T2

MODEL = "v22/yosou_model" if os.path.isdir("v22/yosou_model") else "yosou_model"
PAIRS, PIX = T2.PAIRS, T2.PAIR_IX


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
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--minutes", type=float, default=45)
    args = ap.parse_args()

    import lightgbm as lgb
    feats = json.load(open(f"{MODEL}/features.json", encoding="utf-8"))
    model = lgb.Booster(model_file=f"{MODEL}/lgb_yosou.txt")

    # 1) モデルの確率
    df = YT.load()
    df = df[df["date"].astype(str) >= args.dfrom].copy()
    df, _ = YT.add_features(df)
    df["p"] = YT.norm(model.predict(df[feats]), df["race"].values)
    P = {}
    for rid, g in df.sort_values(["race", "lane"]).groupby("race", sort=False):
        P[rid] = g["p"].values
    print(f"\nモデル確率 {len(P):,}レース")

    # 2) 着順(2連単の的中組)
    HIT = {}
    for p in sorted(glob.glob(os.path.join(YT.RAW, "*.json.gz"))):
        d = os.path.basename(p)[:8]
        if d < args.dfrom:
            continue
        with gzip.open(p, "rt", encoding="utf-8") as f:
            rd = json.load(f)
        for r in rd["races"]:
            if "error" in r or not r.get("hit"):
                continue
            a, b, _ = (int(x) for x in r["hit"].split("-"))
            HIT[f"{d}-{r['jcd']:02d}-{r['rno']}"] = (PIX[(a, b)], r["date"],
                                                     r["jcd"], r["rno"])

    keys = [k for k in P if k in HIT]
    random.seed(42)
    sample = random.sample(keys, min(args.n, len(keys)))
    sample.sort()
    print(f"2連単オッズを取りに行く {len(sample):,}レース")

    # 3) 2連単オッズ
    sess = requests.Session()
    sess.headers.update(T2.UA)
    sess.mount("https://", requests.adapters.HTTPAdapter(
        pool_connections=args.workers * 2, pool_maxsize=args.workers * 2))
    ok = 0
    for k in sample[:5]:
        _, _, jcd, rno = HIT[k]
        o, _ = T2.fetch_fun(sess, k[:8], jcd, rno)
        ok += (o is not None)
    print(f"  info.kyotei.fun 成功 {ok}/5")
    getter = (lambda k: T2.fetch_fun(sess, k[:8], HIT[k][2], HIT[k][3])) if ok >= 3 \
        else (lambda k: T2.fetch(sess, k[:8], HIT[k][2], HIT[k][3]))

    O2, KS = [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i in range(0, len(sample), 300):
            if (time.time() - t0) / 60 > args.minutes:
                print("  時間切れ。ここまでで集計します。")
                break
            chunk = sample[i:i + 300]
            for k, (o, _) in zip(chunk, ex.map(getter, chunk)):
                if o is not None:
                    O2.append(o)
                    KS.append(k)
            print(f"  {i+len(chunk):,}/{len(sample):,}  取得{len(O2):,}  "
                  f"{(time.time()-t0)/60:.1f}分", flush=True)
    if len(O2) < 500:
        sys.exit(f"取得 {len(O2)}件。少なすぎます。")

    O = np.stack(O2).astype(np.float64)
    n = len(O)
    print(f"\n集計対象 {n:,}レース")

    # 4) モデル確率 → 2連単確率 (p_a × p_b/(1-p_a))
    M = np.zeros((n, 30))
    for i, k in enumerate(KS):
        p = P[k]
        for j, (a, b) in enumerate(PAIRS):
            M[i, j] = p[a - 1] * p[b - 1] / max(1 - p[a - 1], 1e-9)
        M[i] /= M[i].sum()
    HM = np.zeros((n, 30), dtype=bool)
    hi = np.array([HIT[k][0] for k in KS])
    HM[np.arange(n), hi] = True
    PAY = O[np.arange(n), hi] * 100.0
    top_lane = np.array([int(np.argmax(P[k])) + 1 for k in KS])
    dates = np.array([int(k[:8]) for k in KS])

    def stat(mask_pt, mask_race=None):
        m = mask_pt if mask_race is None else (mask_pt & mask_race[:, None])
        c = int(m.sum())
        if c < 200:
            return None
        got = (HM & m) * PAY[:, None]
        r = got.sum() / (c * 100)
        se = np.sqrt(((got.sum(1) - r * m.sum(1) * 100) ** 2).sum()) / (c * 100)
        return {"pts": c, "hit": (HM & m).sum() / c, "roi": r, "se": se,
                "z": (r - 1) / se, "pl": got.sum() - c * 100}

    print("\n" + "=" * 58)
    print("[1] 対照実験")
    s = stat(np.ones_like(HM))
    print(f"  2連単30点を全部買った回収率 {pc(s['roi'])} ± {pc(s['se'])}")
    inv = 1.0 / O
    print(f"  2連単の控除率 {1-1/inv.sum(1).mean():.1%}")

    EV = M * O
    print("\n" + "=" * 58)
    print("[2] EV = モデル確率 × 2連単オッズ  の帯別")
    edges = [0, .7, .85, 1.0, 1.05, 1.1, 1.2, 1.4, 1.7, 99]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = stat((EV >= a) & (EV < b))
        if s:
            out.append([f"{a:.2f}〜{b:.2f}", f"{s['pts']:,}", pc(s["hit"]),
                        pc(s["roi"]), pc(s["se"]), f"{s['z']:+.1f}"])
    tbl(["EV帯", "点数", "的中率", "回収率", "誤差±", "z値"], out)

    print("\n[3] EV上限以上だけ買う")
    out = []
    for ev in (1.0, 1.05, 1.10, 1.20, 1.40):
        s = stat(EV >= ev)
        if s:
            out.append([f"{ev:.2f}", f"{s['pts']:,}", pc(s["hit"]), pc(s["roi"]),
                        pc(s["se"]), f"{s['z']:+.1f}", f"{s['pl']:+,.0f}"])
    tbl(["EV下限", "点数", "的中率", "回収率", "誤差±", "z値", "収支"], out)

    print("\n" + "=" * 58)
    print("[4] モデルの本命が1号艇かどうかで分ける")
    for lab, mr in (("本命が1号艇", top_lane == 1),
                    ("本命が1号艇でない", top_lane != 1)):
        print(f"\n  ◇ {lab}  ({int(mr.sum()):,}レース)")
        out = []
        for ev in (0.0, 1.0, 1.10, 1.20):
            s = stat(EV >= ev, mr)
            if s:
                out.append([f"{ev:.2f}" if ev else "全部", f"{s['pts']:,}",
                            pc(s["hit"]), pc(s["roi"]), pc(s["se"]),
                            f"{s['z']:+.1f}"])
        tbl(["EV下限", "点数", "的中率", "回収率", "誤差±", "z値"], out)
        # 本命の頭だけを2連単で流す
        idx = np.where(mr)[0]
        cost = ret = 0.0
        hits = 0
        for i in idx:
            tl = top_lane[i]
            cand = [j for j, (a, b) in enumerate(PAIRS) if a == tl]
            sel = sorted(cand, key=lambda j: O[i, j])[:3]
            cost += 300
            if hi[i] in sel:
                ret += PAY[i]
                hits += 1
        print(f"    本命頭から人気順3点: {len(idx):,}レース 的中{hits:,}"
              f"({hits/len(idx)*100:.1f}%) 回収率 {ret/cost*100:.1f}%")

    print("\n" + "=" * 58)
    print("[5] 前半後半 (EV1.10以上)")
    half = np.median(dates)
    out = []
    for lab, mr in (("前半", dates <= half), ("後半", dates > half)):
        s = stat(EV >= 1.10, mr)
        if s:
            out.append([lab, f"{s['pts']:,}", pc(s["hit"]), pc(s["roi"]),
                        pc(s["se"]), f"{s['z']:+.1f}"])
    tbl(["期間", "点数", "的中率", "回収率", "誤差±", "z値"], out)

    print("\n" + "=" * 58)
    print("判断: 100%を誤差2つ分うわまわり、前半後半とも同じ向きでなければ採用しない")
    print("      締切時オッズでの数字。実運用はこれより落ちる")


if __name__ == "__main__":
    main()
