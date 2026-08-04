#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evband.py -- 期待値が最大になる買い方を実測で決める

■ 分かっていること
  公衆は穴を買いすぎている。オッズが安い組ほど回収率が高い。
    1〜5倍 86.9% / 20〜40倍 74.6% / 400倍以上 27.4%
  これはモデルが市場を超える必要がない、唯一残った実測の事実。

■ 決めること
  「オッズいくら以下の組だけ買う」の閾値。
  1〜5倍という括りは粗すぎるので、細かく割って測る。

■ 検証の作法
  年度ごとに分けて出す。3ブロックとも同じ形が出なければ採用しない。
  誤差を必ず出す。回収率だけを見て決めない。

  使い方: python evband.py
"""

import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np

BANDS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0,
         13.0, 16.0, 20.0, 30.0, 50.0, 100.0, 1e9]
THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0]


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


def load(outdir):
    files = sorted(glob.glob(os.path.join(outdir, "*.json.gz")))
    if not files:
        sys.exit(f"{outdir}/ にデータがありません")
    combos = None
    O, H, P, D = [], [], [], []
    for p in files:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            d = json.load(f)
        if combos is None:
            combos = d["combos"]
            cix = {c: i for i, c in enumerate(combos)}
        for r in d["races"]:
            if "error" in r or r.get("n_odds") != 120:
                continue
            if not r.get("hit") or not r.get("pay_3t"):
                continue
            o = np.array(r["odds"], dtype=np.float32)
            hi = cix[r["hit"]]
            if abs(o[hi] * 100 - r["pay_3t"]) > 10:
                continue                     # 返還レース
            O.append(o)
            H.append(hi)
            P.append(r["pay_3t"])
            D.append(int(r["date"]))
    return (np.stack(O), np.array(H), np.array(P, dtype=np.float64),
            np.array(D, dtype=np.int64))


def year_of(d):
    """5月始まりの年度(配列対応)"""
    d = np.asarray(d)
    return d // 10000 - ((d // 100) % 100 < 5).astype(np.int64)


def roi_se(ret, cost):
    if cost.sum() < 5000:
        return None
    r = float(ret.sum() / cost.sum())
    se = float(np.sqrt(((ret - r * cost) ** 2).sum()) / cost.sum())
    return r, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="raw")
    args = ap.parse_args()

    O, H, PAY, D = load(args.out)
    n = len(O)
    YR = year_of(D)
    years = sorted(set(YR.tolist()))
    print(f"対象 {n:,}レース  ({D.min()} 〜 {D.max()})")
    print("年度別: " + "  ".join(f"{y}年度 {int((YR==y).sum()):,}" for y in years))

    HM = np.zeros_like(O, dtype=bool)
    HM[np.arange(n), H] = True

    # ---------------- 1) 細かいオッズ帯 ----------------
    print("\n" + "=" * 58)
    print("[1] オッズ帯別の実測回収率  (1点=100円で買った場合)")
    rows = []
    for a, b in zip(BANDS[:-1], BANDS[1:]):
        m = (O >= a) & (O < b)
        c = int(m.sum())
        if c < 300:
            continue
        got = (HM & m) * (O * 100)
        r = got.sum() / (c * 100)
        se = np.sqrt(((got.sum(1) - r * m.sum(1) * 100) ** 2).sum()) / (c * 100)
        lab = f"{a:g}〜{b:g}" if b < 1e8 else f"{a:g}以上"
        rows.append([lab, f"{c:,}", pc((HM & m).sum() / c), pc(r), pc(se)])
    tbl(["オッズ", "点数", "的中率", "回収率", "誤差±"], rows)
    print("  控除率25%なので理論値は75%。それより高い帯=買われ足りない")

    # ---------------- 2) 年度ごとに同じ形か ----------------
    print("\n" + "=" * 58)
    print("[2] 同じ表を年度別に  (3つとも同じ形なら本物)")
    head = ["オッズ"] + [f"{y}年度" for y in years]
    rows = []
    for a, b in zip(BANDS[:-1], BANDS[1:]):
        if b > 20 and b < 1e8:
            continue
        line = [f"{a:g}〜{b:g}" if b < 1e8 else f"{a:g}以上"]
        keep = True
        for y in years:
            sel = YR == y
            m = (O[sel] >= a) & (O[sel] < b)
            c = int(m.sum())
            if c < 200:
                keep = False
                break
            got = (HM[sel] & m) * (O[sel] * 100)
            line.append(pc(got.sum() / (c * 100)))
        if keep:
            rows.append(line)
    tbl(head, rows)

    # ---------------- 3) 閾値ごとの成績 ----------------
    print("\n" + "=" * 58)
    print("[3] 「オッズX以下の組を全部買う」")
    rows = []
    for x in THRESHOLDS:
        m = O <= x
        cnt = m.sum(1).astype(float)
        ret = ((HM & m) * (O * 100)).sum(1)
        s = roi_se(ret, cnt * 100)
        if not s:
            continue
        r, se = s
        played = int((cnt >= 1).sum())
        rows.append([f"{x:g}", f"{played:,}", pc(played / n), f"{int(cnt.sum()):,}",
                     f"{cnt[cnt>=1].mean():.1f}", pc(r), pc(se),
                     f"{(r-1)/se:+.1f}", f"{ret.sum()-cnt.sum()*100:+,.0f}"])
    tbl(["X以下", "レース", "割合", "点数", "点/R", "回収率", "誤差±", "z値", "収支"], rows)

    # ---------------- 4) 1番人気だけ ----------------
    print("\n" + "=" * 58)
    print("[4] 「1番人気1点だけ、そのオッズがX以下のときだけ買う」")
    fav = O.argmin(1)
    fav_o = O[np.arange(n), fav]
    fav_win = (fav == H)
    rows = []
    for x in THRESHOLDS:
        m = fav_o <= x
        c = int(m.sum())
        if c < 300:
            continue
        ret = (fav_win & m) * fav_o * 100
        s = roi_se(ret[m], np.full(c, 100.0))
        r, se = s
        rows.append([f"{x:g}", f"{c:,}", pc(c / n), pc(fav_win[m].mean()),
                     pc(r), pc(se), f"{(r-1)/se:+.1f}", f"{ret.sum()-c*100:+,.0f}"])
    tbl(["X以下", "レース", "割合", "的中率", "回収率", "誤差±", "z値", "収支"], rows)

    # ---------------- 5) 年度別の再現 ----------------
    print("\n" + "=" * 58)
    print("[5] 有力な閾値を年度別に  (全部100%近ければ採用、割れるなら不採用)")
    head = ["買い方"] + [f"{y}年度" for y in years] + ["全体", "誤差±"]
    rows = []
    for x in (3.0, 4.0, 5.0, 6.0, 8.0):
        for name, mask, odds_used in (
                ("X以下を全部", O <= x, O),
                ("1番人気のみ", None, None)):
            if name == "X以下を全部":
                line = [f"オッズ{x:g}以下を全部"]
                cells = []
                for y in years:
                    sel = YR == y
                    mm = O[sel] <= x
                    cnt = mm.sum(1).astype(float)
                    ret = ((HM[sel] & mm) * (O[sel] * 100)).sum(1)
                    s = roi_se(ret, cnt * 100)
                    cells.append(pc(s[0]) if s else "—")
                cnt = (O <= x).sum(1).astype(float)
                ret = ((HM & (O <= x)) * (O * 100)).sum(1)
                s = roi_se(ret, cnt * 100)
                if s:
                    rows.append(line + cells + [pc(s[0]), pc(s[1])])
            else:
                line = [f"1番人気(オッズ{x:g}以下)"]
                cells = []
                for y in years:
                    sel = YR == y
                    mm = fav_o[sel] <= x
                    if mm.sum() < 200:
                        cells.append("—")
                        continue
                    ret = (fav_win[sel] & mm) * fav_o[sel] * 100
                    cells.append(pc(ret[mm].sum() / (mm.sum() * 100)))
                mm = fav_o <= x
                ret = (fav_win & mm) * fav_o * 100
                s = roi_se(ret[mm], np.full(int(mm.sum()), 100.0)) if mm.sum() else None
                if s:
                    rows.append(line + cells + [pc(s[0]), pc(s[1])])
    tbl(head, rows)

    print("\n" + "=" * 58)
    print("注意: 締切時オッズです。実運用は判定が10分前なので、必ずこれより落ちます。")
    print("回収率が100%を超える行は出ません。出たら誤差を疑ってください。")


if __name__ == "__main__":
    main()
