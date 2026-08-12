#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yosou_icchi.py -- モデルと市場が一致しているレースだけ買う (Colab)

■ 発想(はむさん)
  これまではずっと「モデルが市場とズレている場面」を探していた(EVで割安を探す)。
  全部だめだった。モデルが市場に負けているので、ズレはモデルの誤差だった。

  今回は逆。「両者が一致している = どちらも自信がある = 読みやすいレース」
  だけを選ぶ。市場を超える必要がない。
  運の標準偏差が90という壁を、レース選択で避けるという発想。

■ 一致度の測り方(3通り)
  corr   … 120通りの確率の相関(順位相関)
  overlap… モデルの上位8組と市場の上位8組が何点かぶるか(0〜8)
  same1  … モデルの1位と市場の1番人気が同じ組か

■ 見どころ
  的中率は確実に上がるはず(読みやすいレースなので)。
  回収率が上がるかは分からない。
  「市場は本命を過小評価している」(1着で実測59.3% 対 市場56.5%, z=+9.9)ので、
  本命が明確なレースでその効果が濃くなる可能性がある。

  使い方 (Colab)
    !pip -q install lightgbm
    !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
    !cp -r v22/yosou_model .
    %run v22/yosou_icchi.py
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
    print("モデルの確率を計算しました")

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
                np.array(r["odds"], dtype=np.float64), r["hit"], r["pay_3t"])
    cidx = {c: i for i, c in enumerate(combos)}
    reo = np.array([cidx[c] for c in names])       # tri順に並べ替える索引
    N = args.pts

    rows = []
    for i in range(n):
        if keys[i] not in OD:
            continue
        o_all, hit, pv = OD[keys[i]]
        o = o_all[reo]
        q = (1.0 / o)
        q = q / q.sum()                            # 市場の確率(tri順)
        p = CP[i]
        hi = names.index(hit)

        mp = np.argsort(-p)[:N]                    # モデルの上位N
        mq = np.argsort(-q)[:N]                    # 市場の上位N
        # 順位相関(スピアマン)を自前で計算
        rp = np.empty(120); rp[np.argsort(-p)] = np.arange(120)
        rq = np.empty(120); rq[np.argsort(-q)] = np.arange(120)
        corr = np.corrcoef(rp, rq)[0, 1]

        rows.append({
            "date": date[i], "hit": hi, "pay": pv,
            "corr": corr,
            "overlap": len(set(mp.tolist()) & set(mq.tolist())),
            "same1": int(mp[0] == mq[0]),
            "in_model": int(hi in mp), "in_mkt": int(hi in mq),
            "ret_model": pv if hi in mp else 0.0,
            "ret_mkt": pv if hi in mq else 0.0,
            "sum8": p[mp].sum(),
            "ninki": int(np.where(np.argsort(-q) == hi)[0][0]) + 1,
        })
    R = pd.DataFrame(rows)
    print(f"突き合わせ {len(R):,}レース\n")

    def stat(m, col="ret_model"):
        k = int(m.sum())
        if k < 300:
            return None
        v = R.loc[m, col].values
        cost = k * N * 100
        roi = v.sum() / cost
        se = np.sqrt(((v - roi * N * 100) ** 2).sum()) / cost
        hit = (v > 0).mean()
        return {"n": k, "hit": hit, "roi": roi, "se": se, "pl": v.sum() - cost}

    print("=" * 58)
    print(f"[0] 対照: 上位{N}組を全レース買った場合")
    out = []
    for lab, col in (("モデルの上位", "ret_model"), ("市場の上位(人気順)", "ret_mkt")):
        s = stat(pd.Series(True, index=R.index), col)
        out.append([lab, f"{s['n']:,}", pc(s["hit"]), pc(s["roi"]),
                    pc(s["se"]), f"{s['pl']:+,.0f}"])
    tbl(["買い方", "レース", "的中率", "回収率", "誤差±", "収支"], out)

    print("\n" + "=" * 58)
    print(f"[1] 上位{N}組の重なり (モデルと市場が何点かぶるか)")
    out = []
    for k in range(N + 1):
        s = stat(R["overlap"] == k)
        if s:
            out.append([f"{k}点かぶり", f"{s['n']:,}", pc(s["hit"]),
                        pc(s["roi"]), pc(s["se"]), f"{s['pl']:+,.0f}"])
    tbl(["一致度", "レース", "的中率", "回収率", "誤差±", "収支"], out)

    print("\n[2] 順位相関で分ける")
    qs = np.unique(np.quantile(R["corr"], np.linspace(0, 1, 6)))
    out = []
    for i in range(len(qs) - 1):
        m = (R["corr"] >= qs[i]) & (R["corr"] < qs[i + 1]
                                    if i < len(qs) - 2 else R["corr"] <= qs[i + 1])
        s = stat(m)
        if s:
            out.append([f"{qs[i]:.3f}〜{qs[i+1]:.3f}", f"{s['n']:,}",
                        pc(s["hit"]), pc(s["roi"]), pc(s["se"]),
                        f"{s['pl']:+,.0f}"])
    tbl(["相関", "レース", "的中率", "回収率", "誤差±", "収支"], out)

    print("\n[3] 本命が一致しているか")
    out = []
    for lab, m in (("モデル1位=1番人気", R["same1"] == 1),
                   ("違う", R["same1"] == 0)):
        s = stat(m)
        if s:
            out.append([lab, f"{s['n']:,}", pc(s["hit"]), pc(s["roi"]),
                        pc(s["se"]), f"{s['pl']:+,.0f}"])
    tbl(["区分", "レース", "的中率", "回収率", "誤差±", "収支"], out)

    print("\n" + "=" * 58)
    print(f"[4] 一致度で絞る (重なりが多い順・上位{N}組)")
    out = []
    for th in (0, 5, 6, 7, 8):
        s = stat(R["overlap"] >= th)
        if s:
            out.append([f"{th}点以上" if th else "全部", f"{s['n']:,}",
                        f"{s['n']/len(R)*100:.0f}%", pc(s["hit"]),
                        pc(s["roi"]), pc(s["se"]), f"{s['pl']:+,.0f}"])
    tbl(["条件", "レース", "割合", "的中率", "回収率", "誤差±", "収支"], out)

    print("\n[5] 一致 × 確率合計 (両方の条件を重ねる)")
    th8 = np.quantile(R["sum8"], 0.80)
    out = []
    for lab, m in (("一致7点以上", R["overlap"] >= 7),
                   ("確率合計 上位20%", R["sum8"] >= th8),
                   ("両方", (R["overlap"] >= 7) & (R["sum8"] >= th8))):
        s = stat(m)
        if s:
            out.append([lab, f"{s['n']:,}", pc(s["hit"]), pc(s["roi"]),
                        pc(s["se"]), f"{s['pl']:+,.0f}"])
    tbl(["条件", "レース", "的中率", "回収率", "誤差±", "収支"], out)

    print("\n[6] 年度で割る (一致7点以上)")
    yr = (R["date"] // 10000 - ((R["date"] // 100) % 100 < 5)).astype(int)
    out = []
    for y in sorted(set(yr.tolist())):
        s = stat((R["overlap"] >= 7) & (yr == y))
        if s:
            out.append([f"{y}年度", f"{s['n']:,}", pc(s["hit"]), pc(s["roi"]),
                        pc(s["se"])])
    tbl(["年度", "レース", "的中率", "回収率", "誤差±"], out)

    print("\n[7] 参考: 何番人気で決着したか")
    out = []
    for lo, hi in ((1, 1), (2, 3), (4, 8), (9, 20), (21, 50), (51, 120)):
        m = (R["ninki"] >= lo) & (R["ninki"] <= hi)
        k = int(m.sum())
        if k < 100:
            continue
        out.append([f"{lo}〜{hi}番人気" if lo != hi else "1番人気",
                    f"{k:,}", pc(k / len(R)),
                    f"{R.loc[m,'pay'].mean():,.0f}円"])
    tbl(["人気", "レース", "割合", "平均払戻"], out)

    print("\n" + "=" * 58)
    print("判断の目安")
    print("  ・的中率は上がるはず。回収率が上がるかが今回の見どころ")
    print("  ・100%を誤差2つ分こえ、年度もそろって初めて意味がある")
    print("  ・締切時オッズでの数字。実運用はこれより落ちる")


if __name__ == "__main__":
    main()
