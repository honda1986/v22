#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yosou_wariyasu.py -- 「確率10%以上の割安舟券だけ買う」を検証する (Colab)

■ 検証する主張(中日スポーツ 2023年の記事)
    確率が10％以上の割安舟券だけを買うという戦略をとると
    回収率は93-99％まで上昇する
    しかし10％以下にまで対象を広げると回収率は80％前後に下がる

  割安 = 想定オッズ(1÷予想確率) < 実オッズ  →  予想確率 × オッズ > 1

■ これまでとの違い
  EVで絞る検証は何度もやって全部だめだった。
  ただし「予想確率が10%以上」という制限をかけた形は試していない。
  3連単で確率10%以上の組は1レースに0〜2点しかない。落ち穂拾いの戦略。

■ 注意
  記事の対象は2,145レース。こちらは34,018レースで測る。
  記事自身も「大数の法則から逃れられないためかもしれない」と書いている。

  使い方 (Colab)
    !pip -q install lightgbm
    !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
    !cp -r v22/yosou_model .
    %run v22/yosou_wariyasu.py
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

    # オッズ
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
                np.array(r["odds"], dtype=np.float64), r["hit"])
    cidx = {c: i for i, c in enumerate(combos)}
    reorder = np.array([cidx[c] for c in names])   # tri順 → raw odds順

    rows = []
    for i in range(n):
        if keys[i] not in OD:
            continue
        o_all, hit = OD[keys[i]]
        o = o_all[reorder]                          # tri順に並べたオッズ
        p = CP[i]
        h = np.zeros(120, dtype=bool)
        h[names.index(hit)] = True
        rows.append((np.full(120, date[i]), p, o, h))
    P = np.concatenate([z[1] for z in rows])
    O = np.concatenate([z[2] for z in rows])
    H = np.concatenate([z[3] for z in rows])
    DT = np.concatenate([z[0] for z in rows])
    print(f"突き合わせ {len(rows):,}レース / {len(P):,}点\n")

    EV = P * O                                      # 1より大きいと割安

    def stat(m):
        k = int(m.sum())
        if k < 100:
            return None
        ret = (O[m] * H[m] * 100).sum()
        cost = k * 100
        roi = ret / cost
        v = O[m] * H[m] * 100
        se = np.sqrt(((v - roi * 100) ** 2).sum()) / cost
        return {"n": k, "hit": H[m].mean(), "roi": roi, "se": se,
                "pl": ret - cost}

    print("=" * 58)
    print("[1] 記事の主張の再現  『予想確率◯%以上 かつ 割安(EV>1)』")
    out = []
    for th in (0.10, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.0):
        m = (P >= th) & (EV > 1.0)
        s = stat(m)
        if s:
            out.append([f"{th*100:.1f}%以上" if th else "制限なし",
                        f"{s['n']:,}", pc(s["hit"]), pc(s["roi"]),
                        pc(s["se"]), f"{s['pl']:+,.0f}"])
    tbl(["予想確率", "点数", "的中率", "回収率", "誤差±", "収支"], out)
    print("  記事の主張: 10%以上なら93-99% / 広げると80%前後")

    print("\n[2] 割安の度合いを変える (予想確率10%以上)")
    out = []
    for ev in (1.0, 1.05, 1.1, 1.2, 1.3, 1.5):
        s = stat((P >= 0.10) & (EV > ev))
        if s:
            out.append([f"EV>{ev:.2f}", f"{s['n']:,}", pc(s["hit"]),
                        pc(s["roi"]), pc(s["se"]), f"{s['pl']:+,.0f}"])
    tbl(["条件", "点数", "的中率", "回収率", "誤差±", "収支"], out)

    print("\n[3] 対照: 割安でない側も見る (予想確率10%以上)")
    out = []
    for lab, m in (("割安(EV>1)", (P >= 0.10) & (EV > 1.0)),
                   ("割高(EV<=1)", (P >= 0.10) & (EV <= 1.0)),
                   ("全部", P >= 0.10)):
        s = stat(m)
        if s:
            out.append([lab, f"{s['n']:,}", pc(s["hit"]), pc(s["roi"]),
                        pc(s["se"]), f"{s['pl']:+,.0f}"])
    tbl(["区分", "点数", "的中率", "回収率", "誤差±", "収支"], out)
    print("  割安と割高で差が無ければ、モデルはオッズを出し抜けていない")

    print("\n[4] 年度で割る (予想確率10%以上・割安)")
    yr = (DT // 10000 - ((DT // 100) % 100 < 5)).astype(int)
    out = []
    for y in sorted(set(yr.tolist())):
        s = stat((P >= 0.10) & (EV > 1.0) & (yr == y))
        if s:
            out.append([f"{y}年度", f"{s['n']:,}", pc(s["hit"]), pc(s["roi"]),
                        pc(s["se"])])
    tbl(["年度", "点数", "的中率", "回収率", "誤差±"], out)

    print("\n[5] 較正: 予想確率どおりに当たっているか")
    out = []
    for lo, hi in ((0, .005), (.005, .01), (.01, .02), (.02, .05),
                   (.05, .10), (.10, .20), (.20, 1.0)):
        m = (P >= lo) & (P < hi)
        if m.sum() < 100:
            continue
        out.append([f"{lo*100:.1f}〜{hi*100:.0f}%", f"{int(m.sum()):,}",
                    pc(P[m].mean()), pc(H[m].mean()),
                    pc((1 / O[m]).mean() * 0.75)])
    tbl(["予想確率の帯", "点数", "モデル", "実測", "市場(0.75/o)"], out)

    print("\n" + "=" * 58)
    print("判断の目安")
    print("  ・[3]で割安と割高に差が無ければ、オッズを出し抜けていない")
    print("  ・100%を誤差2つ分こえ、年度もそろって初めて意味がある")
    print("  ・締切時オッズでの数字。実運用はこれより落ちる")


if __name__ == "__main__":
    main()
