#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evsplit.py -- 「オッズX以下」の中で、買うレースと買わないレースを分ける軸を探す

■ 前提(evband.py の測定結果)
    オッズ4以下  3,902レース(3.0%)   87.2% ± 2.4%
    オッズ5以下 13,588レース(10.3%)  84.9% ± 1.3%
    オッズ6以下 29,713レース(22.5%)  84.0% ± 0.9%
  そして安い側の帯は 82〜89% で横ばい。オッズをこれ以上下げても分かれない。
  だから別の軸が要る。

■ 目標
  全体84.9%に対して 88〜92% の部分集合。+3〜7ポイント。
  100%は狙わない。控除率25%の内側の話なので届かない。

■ 作法
  ・軸は12本。区分を含めて60通りくらい試す。最大値は必ず上振れる。
  ・4年度すべてで同じ方向に出た軸だけを候補にする。
  ・誤差とz値を必ず出す。回収率だけで決めない。
  ・進入コースは使わない(raw のものは本番進入で漏れるため)。

  使い方: python evsplit.py --max-odds 5.0
"""

import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np

VENUE = {1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
         7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
         13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
         19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村"}
CLS = {4: "A1", 3: "A2", 2: "B1", 1: "B2"}


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
    combos = cix = None
    rec = []
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
                continue
            rec.append((o, hi, int(r["date"]), r["jcd"], r["rno"], r["entries"]))
    return rec, combos


def year_of(d):
    d = np.asarray(d)
    return d // 10000 - ((d // 100) % 100 < 5).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="raw")
    ap.add_argument("--max-odds", type=float, default=5.0)
    args = ap.parse_args()

    rec, combos = load(args.out)
    first_of = np.array([int(c.split("-")[0]) for c in combos])
    print(f"全体 {len(rec):,}レース")

    # ---- オッズ条件を満たすレースだけ抜き出し、1レース1行にまとめる ----
    rows = []
    for o, hi, date, jcd, rno, ent in rec:
        sel = np.where(o <= args.max_odds)[0]
        if len(sel) == 0:
            continue
        e = {x["lane"]: x for x in ent}
        b1 = e.get(1, {})
        tenji = [e[l].get("tenji") or 0 for l in range(1, 7)]
        st = [e[l].get("avg_st") or 0.2 for l in range(1, 7)]
        q = (1.0 / o) / (1.0 / o).sum()
        rows.append({
            "n": len(sel), "ret": float(o[hi] * 100) if hi in sel else 0.0,
            "cost": len(sel) * 100.0, "won": hi in sel,
            "date": date, "jcd": jcd, "rno": rno,
            "min_o": float(o[sel].min()),
            "top3q": float(np.sort(q)[::-1][:3].sum()),
            "lane1_first": int((first_of[sel] == 1).all()),
            "cls1": b1.get("cls_val") or 0,
            "nwin1": b1.get("n_win") or 0.0,
            "m2ren1": b1.get("m_2ren") or 0.0,
            "tenji1_rank": 1 + sum(1 for t in tenji if 0 < t < (tenji[0] or 99)),
            "st1_rank": 1 + sum(1 for s in st if s < st[0]),
            "f1": b1.get("f_count") or 0,
        })
    if not rows:
        sys.exit("条件を満たすレースがありません")

    n = len(rows)
    RET = np.array([r["ret"] for r in rows])
    COST = np.array([r["cost"] for r in rows])
    YR = year_of(np.array([r["date"] for r in rows]))
    years = sorted(set(YR.tolist()))

    def stat(m):
        c = COST[m].sum()
        if c < 30000:
            return None
        r = float(RET[m].sum() / c)
        se = float(np.sqrt(((RET[m] - r * COST[m]) ** 2).sum()) / c)
        return {"n": int(m.sum()), "pts": int(COST[m].sum() // 100),
                "roi": r, "se": se, "z": (r - 1) / se,
                "pl": float(RET[m].sum() - c)}

    base = stat(np.ones(n, bool))
    print(f"オッズ{args.max_odds:g}以下のレース {n:,}件 / {base['pts']:,}点")
    print(f"全体 回収率 {pc(base['roi'])} ± {pc(base['se'])}   収支 {base['pl']:+,.0f}円")
    print("年度別: " + "  ".join(
        f"{y} {pc(stat(YR==y)['roi'])}" for y in years if stat(YR == y)))

    # ---- 軸の定義 ----
    def qbins(vals, k=4):
        v = np.array(vals, dtype=float)
        e = np.unique(np.quantile(v, np.linspace(0, 1, k + 1)))
        b = np.clip(np.digitize(v, e[1:-1]), 0, len(e) - 2)
        lab = {i: f"{e[i]:.3g}〜{e[i+1]:.3g}" for i in range(len(e) - 1)}
        return b, lab

    G = lambda k: np.array([r[k] for r in rows])
    axes = []
    axes.append(("買える点数", G("n"), {i: f"{i}点" for i in range(1, 9)}))
    b, l = qbins(G("min_o"), 4); axes.append(("最安オッズ", b, l))
    b, l = qbins(G("top3q"), 4); axes.append(("市場の集中度(上位3点)", b, l))
    axes.append(("買い目が1号艇頭のみ", G("lane1_first"), {0: "他艇も含む", 1: "1号艇頭のみ"}))
    axes.append(("1号艇の級別", G("cls1"), CLS))
    b, l = qbins(G("nwin1"), 4); axes.append(("1号艇の全国勝率", b, l))
    b, l = qbins(G("m2ren1"), 4); axes.append(("1号艇のモーター2連率", b, l))
    axes.append(("1号艇の展示順位", G("tenji1_rank"), {i: f"{i}位" for i in range(1, 7)}))
    axes.append(("1号艇の平均ST順位", G("st1_rank"), {i: f"{i}位" for i in range(1, 7)}))
    axes.append(("1号艇のF数", np.minimum(G("f1"), 2), {0: "F0", 1: "F1", 2: "F2以上"}))
    axes.append(("R番号", G("rno"), {i: f"{i}R" for i in range(1, 13)}))
    axes.append(("場", G("jcd"), VENUE))

    print("\n" + "=" * 58)
    print("[1] 軸ごとの回収率")
    print("  誤差とz値を見ること。z値2未満は偶然の範囲")
    cands = []
    for name, b, lab in axes:
        print(f"\n  ◇ {name}")
        rs, best = [], None
        for k in sorted(set(np.asarray(b).tolist())):
            s = stat(b == k)
            if not s:
                continue
            rs.append([lab.get(k, k), f"{s['n']:,}", f"{s['pts']:,}",
                       pc(s["roi"]), pc(s["se"]), f"{s['z']:+.1f}",
                       f"{s['pl']:+,.0f}"])
            if best is None or s["roi"] > best[1]["roi"]:
                best = (k, s)
        rs.sort(key=lambda r: -float(r[3][:-1]))
        tbl(["区分", "レース", "点数", "回収率", "誤差±", "z値", "収支"], rs)
        if best and best[1]["roi"] > base["roi"]:
            cands.append((name, b, lab, best[0], best[1]))

    # ---- 年度別の再現 ----
    print("\n" + "=" * 58)
    print("[2] 良かった区分を年度別に  (4年度そろって全体を上回るものだけ採用)")
    cands.sort(key=lambda c: -c[4]["roi"])
    head = ["軸", "区分", "レース", "全体"] + [str(y) for y in years]
    out = []
    for name, b, lab, k, s in cands[:12]:
        line = [name[:12], str(lab.get(k, k))[:10], f"{s['n']:,}", pc(s["roi"])]
        good = 0
        for y in years:
            t = stat((b == k) & (YR == y))
            if not t:
                line.append("—")
                continue
            line.append(pc(t["roi"]))
            base_y = stat(YR == y)
            if base_y and t["roi"] > base_y["roi"]:
                good += 1
        line.append(f"{good}/{len(years)}")
        out.append(line)
    tbl(head + ["勝ち"], out)

    print("\n" + "=" * 58)
    print(f"判断の目安")
    print(f"  ・全体 {pc(base['roi'])} に対して +3〜7ポイントが現実的な目標")
    print(f"  ・z値2未満、または『勝ち』が4/4でないものは使わない")
    print(f"  ・軸を2つ以上重ねない。重ねるほど当たって見えるが再現しない")
    print("  ・締切時オッズでの数字。実運用はこれより落ちる")


if __name__ == "__main__":
    main()
