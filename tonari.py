#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tonari.py -- 内側の艇との強弱差が着順に効くかを測る (Google Colab)

■ 仮説(はむさん)
  絶対的な強さではなく「内側の隣が誰か」で着順が変わる。
  3号艇が差そうとしても、2号艇が強くて先に締めれば行き場がない。
  逆に内側が弱ければ、実力以上に走れる。

■ 測り方
  2〜6号艇について「自分 − 内側の艇」の差を作り、
  市場の想定(3連単オッズから逆算)と実測1着率のズレを差の大きさ別に見る。
  1号艇は内側がいないので対象外。

  指標: 全国勝率 / 今節得点率 / 平均ST / 今節ST / モーター2連率
  ST は小さいほど良いので符号を反転して「自分が速いほどプラス」に揃える。

■ v22のモデルとの違い
  モデルは n_win_dev(レース平均との差)を既に使っている。
  入っていないのは「隣接という位置関係」。
  3号艇にとって2号艇が強いのと6号艇が強いのは意味が違う、という点。

  使い方 (Colab)
    !pip -q install lightgbm
    !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
    %run v22/tonari.py
"""

import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

import yosou_train as YT

RAW = YT.RAW


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


def load_market(dfrom):
    """レースごとの市場1着確率(3連単オッズから逆算)"""
    combos = None
    out = {}
    for p in sorted(glob.glob(os.path.join(RAW, "*.json.gz"))):
        d = os.path.basename(p)[:8]
        if d < dfrom:
            continue
        with gzip.open(p, "rt", encoding="utf-8") as f:
            rd = json.load(f)
        if combos is None:
            combos = rd["combos"]
            cix = {c: i for i, c in enumerate(combos)}
            first = np.array([int(c.split("-")[0]) for c in combos])
        for r in rd["races"]:
            if "error" in r or r.get("n_odds") != 120 or not r.get("hit"):
                continue
            if not r.get("pay_3t"):
                continue
            o = np.array(r["odds"], dtype=np.float64)
            hi = cix[r["hit"]]
            if abs(o[hi] * 100 - r["pay_3t"]) > 10:
                continue
            inv = 1.0 / o
            q = inv / inv.sum()
            out[f"{d}-{r['jcd']:02d}-{r['rno']}"] = np.array(
                [q[first == l].sum() for l in range(1, 7)])
    return out


# 指標: (表示名, 列名, 大きいほど良いか)
METRICS = [
    ("全国勝率", "n_win", True),
    ("今節得点率", "tok", True),
    ("平均ST", "avg_st", False),
    ("今節ST", "st_setsu", False),
    ("モーター2連率", "m_2ren", True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="dfrom", default="20230501")
    args = ap.parse_args()

    df = YT.load()
    df = df[df["date"].astype(str) >= args.dfrom].copy()
    print(f"\n対象 {df['race'].nunique():,}レース ({args.dfrom}〜)")

    MK = load_market(args.dfrom)
    print(f"オッズが揃ったレース {len(MK):,}")

    df = df.sort_values(["race", "lane"])
    rows = []
    for rid, g in df.groupby("race", sort=False):
        q = MK.get(rid)
        if q is None or len(g) != 6:
            continue
        L = g.to_dict("records")
        for i in range(1, 6):                    # 2〜6号艇
            rec = {"race": rid, "date": L[i]["date"], "lane": i + 1,
                   "q": q[i], "won": L[i]["y"]}
            for name, col, big in METRICS:
                a, b = L[i].get(col), L[i - 1].get(col)   # 自分, 内側
                if a is None or b is None or (
                        isinstance(a, float) and np.isnan(a)) or (
                        isinstance(b, float) and np.isnan(b)):
                    rec[col] = np.nan
                    continue
                d = (a - b) if big else (b - a)   # 自分が優れているほどプラス
                rec[col] = d
            rows.append(rec)
    R = pd.DataFrame(rows)
    print(f"艇×レース(2〜6号艇) {len(R):,}行\n")

    # 較正(市場の下駄を外す)
    edges = np.unique(np.quantile(R["q"], np.linspace(0, 1, 21)))
    bins = np.clip(np.digitize(R["q"], edges[1:-1]), 0, len(edges) - 2)
    cal = np.zeros(len(edges) - 1)
    for k in range(len(edges) - 1):
        m = bins == k
        cal[k] = R.loc[m, "won"].mean() if m.sum() >= 100 else R.loc[m, "q"].mean()
    R["exp"] = cal[bins]
    print("=" * 58)
    print("[0] 較正の確認")
    print(f"  市場の平均 {pc(R['q'].mean())}  実測 {pc(R['won'].mean())}  "
          f"→ この差を較正で外します")

    def stat(m):
        k = int(m.sum())
        if k < 500:
            return None
        a = R.loc[m, "won"].mean()
        e = R.loc[m, "exp"].mean()
        q = R.loc[m, "q"].mean()
        se = np.sqrt(max(a * (1 - a), 1e-9) / k)
        return {"n": k, "act": a, "exp": e, "d": a - e, "z": (a - e) / se,
                "roi": 0.75 * a / max(q, 1e-9)}

    def show(title, col, mask=None, q=5):
        base = R[col].notna() if mask is None else (R[col].notna() & mask)
        if base.sum() < 3000:
            print(f"\n  ◇ {title}: データ不足")
            return
        v = R.loc[base, col]
        qs = np.quantile(v, np.linspace(0, 1, q + 1))
        qs = np.unique(qs)
        out = []
        for i in range(len(qs) - 1):
            lo, hi = qs[i], qs[i + 1]
            m = base & (R[col] >= lo) & (R[col] < hi if i < len(qs) - 2
                                         else R[col] <= hi)
            s = stat(m)
            if s:
                out.append([f"{lo:+.2f}〜{hi:+.2f}", f"{s['n']:,}", pc(s["exp"]),
                            pc(s["act"]), f"{s['d']*100:+.2f}", f"{s['z']:+.1f}",
                            pc(s["roi"])])
        print(f"\n  ◇ {title}")
        tbl(["自分−内側", "行数", "較正の期待", "実測", "差(pt)", "z値", "想定回収率"], out)

    print("\n" + "=" * 58)
    print("[1] 内側の艇との差  (2〜6号艇まとめて)")
    print("  プラス = 自分の方が優れている。STは符号を反転済み")
    print("  想定回収率 = 75% × 実測 ÷ 市場。100%には 実測÷市場 が1.33必要")
    for name, col, _ in METRICS:
        show(name, col)

    print("\n" + "=" * 58)
    print("[2] 枠番ごとに分ける  (全国勝率の差)")
    for lane in (2, 3, 4, 5, 6):
        show(f"{lane}号艇  自分−{lane-1}号艇", "n_win", R["lane"] == lane, q=4)

    print("\n" + "=" * 58)
    print("[3] 年度別の再現  (差が最も大きい/小さい層)")
    yr = (R["date"] // 10000 - ((R["date"] // 100) % 100 < 5)).astype(int)
    years = sorted(set(yr.tolist()))
    head = ["指標", "層"] + [str(y) for y in years] + ["全体", "z値"]
    out = []
    for name, col, _ in METRICS:
        base = R[col].notna()
        if base.sum() < 3000:
            continue
        lo, hi = np.quantile(R.loc[base, col], [0.2, 0.8])
        for lab, m in ((f"下位20%", base & (R[col] <= lo)),
                       (f"上位20%", base & (R[col] >= hi))):
            s = stat(m)
            if not s:
                continue
            line = [name, lab]
            for y in years:
                t = stat(m & (yr == y))
                line.append(f"{t['d']*100:+.2f}" if t else "—")
            line += [f"{s['d']*100:+.2f}", f"{s['z']:+.1f}"]
            out.append(line)
    tbl(head, out)

    print("\n" + "=" * 58)
    print("判断の目安")
    print("  ・|z|が3未満は採用しない(5指標×5層=25通り見ているので上振れる)")
    print("  ・年度がそろって同じ符号でなければ採用しない")
    print("  ・想定回収率が100%未満なら、効果が本物でも金にはならない")
    print("  ・締切時オッズでの数字。実運用はこれより落ちる")


if __name__ == "__main__":
    main()
