#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag.py -- backfill で集めた raw/ が本当に正しいかを確認する

「数字が良すぎるときは必ずデータを疑う」を分析前にやる。
ここでおかしければパーサを直して raw を取り直す。

  python diag.py
"""

import argparse
import glob
import gzip
import json
import os
from collections import Counter

DEFAULT_FLAGS = [("avg_st", 0.17), ("tenji", 0.0), ("weight", 50),
                 ("age", 30), ("n_win", 0.0), ("m_2ren", 0.0)]


def pct(a, b):
    return f"{100.0*a/b:.1f}%" if b else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="raw")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.out, "*.json.gz")))
    if not files:
        print(f"{args.out}/ にデータがありません")
        return

    races, errors, days = [], Counter(), 0
    for p in files:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            d = json.load(f)
        days += 1
        for r in d["races"]:
            (errors.update([r["error"]]) if "error" in r else races.append(r))
    combos = json.load(gzip.open(files[0], "rt", encoding="utf-8"))["combos"]
    idx = {c: i for i, c in enumerate(combos)}

    n = len(races)
    print("=" * 50)
    print(f"日数 {days}   レース {n}件   取れなかった {sum(errors.values())}件")
    print(f"期間 {os.path.basename(files[0])[:8]} 〜 {os.path.basename(files[-1])[:8]}")
    if errors:
        print("\n[取れなかった理由] ※http_404 は非開催なので正常")
        for k, v in errors.most_common(8):
            print(f"  {k}: {v}")
    if n == 0:
        print("\n★1件も取れていません。backfill.py --check で中身を確認してください。")
        return

    # 1. デフォルト値のまま = パース失敗の疑い
    print("\n[1] デフォルト値のまま残っている率(高いとパース失敗)")
    boats = n * 6
    for f, dv in DEFAULT_FLAGS:
        c = sum(1 for r in races for e in r["entries"] if e.get(f) == dv)
        mark = "" if c < boats * 0.05 else "   ← 要確認"
        print(f"  {f:<9} = {dv:<6} {pct(c, boats):>7}{mark}")

    # 2. オッズ
    print("\n[2] オッズ")
    full = sum(1 for r in races if r.get("n_odds") == 120)
    print(f"  120点そろっている {pct(full, n)}")
    exact = near = bad = nohit = 0
    for r in races:
        hit, pay = r.get("hit"), r.get("pay_3t")
        if not hit or not pay:
            nohit += 1
            continue
        o = r["odds"][idx[hit]]
        if o is None:
            nohit += 1
            continue
        diff = abs(o * 100 - pay)
        exact += diff < 1
        near += 1 <= diff <= 10
        bad += diff > 10
    tot = exact + near + bad
    print(f"  的中目のオッズ×100 = 払戻:  完全一致 {pct(exact,tot)}  "
          f"±10円 {pct(near,tot)}  不一致 {pct(bad,tot)}")
    print(f"  照合できず {nohit}件 {pct(nohit,n)}")
    if tot and bad > tot * 0.05:
        print("  ← 不一致が多い。オッズか払戻の読み取りを疑う")

    # 3. 控除率
    ded = []
    for r in races:
        vals = [o for o in r["odds"] if o]
        if len(vals) == 120:
            s = sum(1.0 / o for o in vals)
            ded.append(1 - 1 / s)
    if ded:
        ded.sort()
        q = lambda p: ded[min(int(len(ded) * p), len(ded) - 1)]
        print(f"\n[3] 控除率(理論25%)  中央値 {q(.5)*100:.1f}%  "
              f"5%点 {q(.05)*100:.1f}%  95%点 {q(.95)*100:.1f}%")
        if not (0.22 < q(.5) < 0.28):
            print("  ← 25%から外れている。オッズの読み取りを疑う")

    # 4. 常識チェック
    print("\n[4] 常識チェック")
    c1, lane1, pays = Counter(), 0, []
    for r in races:
        if not r.get("hit"):
            continue
        w = int(r["hit"][0])
        lane1 += (w == 1)
        cin = {e["lane"]: e.get("course_in", e["lane"]) for e in r["entries"]}
        c1[cin.get(w, w)] += 1
        if r.get("pay_3t"):
            pays.append(r["pay_3t"])
    tot2 = sum(c1.values())
    print(f"  1号艇の1着率 {pct(lane1,tot2)}  (目安 54〜57%)")
    print("  1着コース: " + "  ".join(f"{c}={pct(c1[c],tot2)}" for c in range(1, 7)))
    if pays:
        pays.sort()
        print(f"  3連単配当 中央値 {pays[len(pays)//2]:,}円  平均 {sum(pays)//len(pays):,}円")
        print(f"  120点全部買いの回収率 {sum(pays)/(len(pays)*120*100)*100:.1f}%  "
              f"(75%前後なら正常)")

    # 5. 分布
    by_jcd = Counter(r["jcd"] for r in races)
    print(f"\n[5] 場数 {len(by_jcd)}   1日あたり平均 {n//max(days,1)}レース")
    print("=" * 50)
    print("全部正常なら analyze.py に進んでOK")


if __name__ == "__main__":
    main()
