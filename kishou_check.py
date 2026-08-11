#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kishou_check.py -- rawに入っている気象データが使えるか確かめる (Colab)

  backfill.py は気温・風速・波高・水温を保存している。
  ただし風向きは入っていない(追い風か向かい風かが分からない)。
  まず「どれだけ埋まっているか」「値が妥当か」「1着率と関係があるか」を見る。

  使い方 (Colab)
    !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
    %run v22/kishou_check.py
"""

import glob
import gzip
import json
import os

import numpy as np
import pandas as pd

RAW = "v22/raw" if os.path.isdir("v22/raw") else "raw"
KEYS = ["temp", "wind", "wave", "water_temp"]
NAME = {"temp": "気温", "wind": "風速", "wave": "波高", "water_temp": "水温"}


def tbl(header, rows):
    if not rows:
        print("  (該当なし)")
        return
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    print("  " + "  ".join(str(h).rjust(w[i]) for i, h in enumerate(header)))
    for r in rows:
        print("  " + "  ".join(str(v).rjust(w[i]) for i, v in enumerate(r)))


def main():
    rows = []
    files = sorted(glob.glob(os.path.join(RAW, "*.json.gz")))
    print(f"raw {len(files)}日")
    for k, p in enumerate(files):
        d = os.path.basename(p)[:8]
        with gzip.open(p, "rt", encoding="utf-8") as f:
            rd = json.load(f)
        for r in rd["races"]:
            if "error" in r or not r.get("hit"):
                continue
            w = r.get("weather") or {}
            rows.append({"date": int(d), "jcd": r["jcd"], "rno": r["rno"],
                         "win1": 1 if r["hit"].startswith("1-") else 0,
                         "hit1": int(r["hit"].split("-")[0]),
                         **{c: w.get(c) for c in KEYS}})
        if (k + 1) % 300 == 0:
            print(f"  {k+1}/{len(files)}日  {len(rows):,}レース", flush=True)
    R = pd.DataFrame(rows)
    print(f"\n{len(R):,}レース\n")

    print("=" * 58)
    print("[1] どれだけ埋まっているか")
    out = []
    for c in KEYS:
        v = R[c]
        ok = v.notna()
        out.append([NAME[c], f"{ok.mean()*100:.1f}%", f"{int(ok.sum()):,}",
                    f"{v.min():.1f}" if ok.any() else "—",
                    f"{v.median():.1f}" if ok.any() else "—",
                    f"{v.max():.1f}" if ok.any() else "—"])
    tbl(["項目", "埋まり", "件数", "最小", "中央", "最大"], out)
    if R["wind"].notna().mean() < 0.5:
        print("\n  ★風速がほとんど入っていません。使えません。")
        return

    print("\n[2] 年度ごとの埋まり具合 (途中から取れていないか)")
    yr = (R["date"] // 10000 - ((R["date"] // 100) % 100 < 5)).astype(int)
    out = []
    for y in sorted(set(yr.tolist())):
        m = yr == y
        out.append([f"{y}年度", f"{int(m.sum()):,}"] +
                   [f"{R.loc[m, c].notna().mean()*100:.0f}%" for c in KEYS])
    tbl(["年度", "レース"] + [NAME[c] for c in KEYS], out)

    print("\n" + "=" * 58)
    print("[3] 風速と1号艇の1着率")
    W = R[R["wind"].notna()]
    out = []
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8),
                   (8, 99)):
        m = (W["wind"] >= lo) & (W["wind"] < hi)
        if m.sum() < 300:
            continue
        a = W.loc[m, "win1"].mean()
        se = np.sqrt(a * (1 - a) / m.sum())
        out.append([f"{lo}〜{hi}m" if hi < 99 else "8m〜", f"{int(m.sum()):,}",
                    f"{a*100:.1f}%", f"±{se*100:.1f}"])
    tbl(["風速", "レース", "1号艇1着率", "誤差"], out)
    print("  風向きが無いので、追い風と向かい風が混ざっています")

    print("\n[4] 風速と1着艇の枠番 (荒れているか)")
    out = []
    for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 99)):
        m = (W["wind"] >= lo) & (W["wind"] < hi)
        if m.sum() < 300:
            continue
        line = [f"{lo}〜{hi}m" if hi < 99 else "6m〜", f"{int(m.sum()):,}"]
        for l in range(1, 7):
            line.append(f"{(W.loc[m,'hit1']==l).mean()*100:.1f}%")
        out.append(line)
    tbl(["風速", "レース"] + [f"{l}号艇" for l in range(1, 7)], out)

    print("\n[5] 波高と1号艇の1着率")
    V = R[R["wave"].notna()]
    out = []
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 99)):
        m = (V["wave"] >= lo) & (V["wave"] < hi)
        if m.sum() < 300:
            continue
        a = V.loc[m, "win1"].mean()
        se = np.sqrt(a * (1 - a) / m.sum())
        out.append([f"{lo}〜{hi}cm" if hi < 99 else "5cm〜", f"{int(m.sum()):,}",
                    f"{a*100:.1f}%", f"±{se*100:.1f}"])
    tbl(["波高", "レース", "1号艇1着率", "誤差"], out)

    print("\n" + "=" * 58)
    print("判断")
    print("  ・埋まりが9割を超えていれば特徴量に足せる")
    print("  ・風速で1号艇の1着率が動いていれば、意味のある変数")
    print("  ・風向きが無いのが弱点。追い風と向かい風が打ち消し合う")
    print("  ・ただし気象は直前情報として公開済み。市場は織り込んでいるはず")


if __name__ == "__main__":
    main()
