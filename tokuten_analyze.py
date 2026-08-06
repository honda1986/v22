#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tokuten_analyze.py -- 節内の動機(得点率・準優ボーダー)が市場に織り込まれているかを測る

■ 仮説(事前に凍結)
  予選終盤、準優ボーダー付近の選手は無理をし、当確圏の選手は無理をしない。
  市場がこれを織り込んでいなければ、実測1着率が市場想定からズレる。
  ★符号は事前に決めない(両側)。ファンが勝負駆けを買いすぎている可能性もある。

■ 今までの5回と何が違うか
  モデルが市場を超える必要がない。市場確率と実測の突き合わせだけ。
  しかも得点率は一般戦では公式が出しておらず(得点率早見はG2以上のみ)、
  織り込まれにくい条件が揃っている。

■ 較正について
  市場のq1はもともと本命を過小評価している(実測55.7% 対 市場53.0%)。
  素で比べるとどのグループも「市場より良い」と出てしまう。
  先にq1帯別の実測1着率で較正曲線を作り、そこからのズレだけを見る。

■ 漏れ対策
  ・tokuten の得点率は D日の朝の状態(--check で白判定済み)。買う時点で入手可能。
  ・raw の course_in は本番進入なので一切使わない。

  使い方: python tokuten_analyze.py
"""

import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np


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


def is_final_race(name):
    """準優・優勝・特別選抜など、予選以外か"""
    n = name or ""
    return any(k in n for k in ("準優", "優勝", "特選", "選抜"))


def load(raw_dir, tok_dir):
    tok_files = {os.path.basename(p)[:8]: p
                 for p in glob.glob(os.path.join(tok_dir, "*.json.gz"))}
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.json.gz")))
    if not tok_files:
        sys.exit(f"{tok_dir}/ がありません")

    combos = None
    rows = []
    n_raw = n_join = 0
    for p in raw_files:
        d = os.path.basename(p)[:8]
        if d not in tok_files:
            continue
        with gzip.open(p, "rt", encoding="utf-8") as f:
            rawd = json.load(f)
        with gzip.open(tok_files[d], "rt", encoding="utf-8") as f:
            tokd = json.load(f)
        if combos is None:
            combos = rawd["combos"]
            cix = {c: i for i, c in enumerate(combos)}
            first = np.array([int(c.split("-")[0]) for c in combos])
            M1 = np.zeros((120, 6), dtype=np.float64)
            for i, b in enumerate(first):
                M1[i, b - 1] = 1.0

        # tokuten 側を (jcd,rno,lane) で索引
        tk = {}
        meta = {}
        for jcd_s, v in tokd.get("venues", {}).items():
            j = int(jcd_s)
            for r in v.get("races", []):
                meta[(j, r["rno"])] = (r.get("name", ""), v.get("day_no"),
                                       v.get("n_days"))
                for x in r["lanes"]:
                    tk[(j, r["rno"], x["lane"])] = x

        for r in rawd["races"]:
            if "error" in r or r.get("n_odds") != 120:
                continue
            if not r.get("hit") or not r.get("pay_3t"):
                continue
            o = np.array(r["odds"], dtype=np.float64)
            hi = cix[r["hit"]]
            if abs(o[hi] * 100 - r["pay_3t"]) > 10:
                continue                       # 返還
            n_raw += 1
            key = (r["jcd"], r["rno"])
            if key not in meta:
                continue
            name, day_no, n_days = meta[key]
            if day_no is None or n_days is None:
                continue
            inv = 1.0 / o
            q = inv / inv.sum()
            q1 = q @ M1                        # 艇ごとの市場1着確率
            joined = False
            for e in r["entries"]:
                x = tk.get((r["jcd"], r["rno"], e["lane"]))
                if not x or x.get("tokuten") is None:
                    continue
                runs = x.get("runs", [])
                trouble = any(not isinstance(u.get("fin"), int) for u in runs)
                joined = True
                cls_map = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
                rows.append((
                    int(d), r["jcd"], r["rno"], e["lane"],
                    float(x.get("toban") or 0),
                    float(cls_map.get(x.get("cls"), 0)),
                    float(x.get("st_setsu") or 0),
                    float(e.get("avg_st") or 0),
                    float(x["tokuten"]),
                    x.get("rank") or 0,                 # 節内順位
                    float(x.get("genten") or 0.0),
                    len(runs),
                    1 if trouble else 0,
                    day_no, n_days,
                    0 if is_final_race(name) else 1,    # 予選相当か
                    q1[e["lane"] - 1],
                    1 if e.get("rank") == 1 else 0,
                    o[hi] * 100.0, hi, r["jcd"] * 100 + r["rno"],
                ))
            n_join += joined
    print(f"raw {n_raw:,}レース中 {n_join:,}レースで得点率が結合できました "
          f"({n_join/max(n_raw,1)*100:.1f}%)")
    cols = ["date", "jcd", "rno", "lane", "toban", "cls", "st_s", "st_avg",
            "tok", "srank",
            "genten", "nruns",
            "trouble", "day_no", "n_days", "is_yosen", "q1", "won",
            "pay", "hitix", "rkey"]
    A = np.array(rows, dtype=np.float64)
    return {c: A[:, i] for i, c in enumerate(cols)}, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--tok", default="tokuten")
    ap.add_argument("--min-runs", type=int, default=2,
                    help="この走数以上の選手だけを対象(初日・2日目を外す)")
    args = ap.parse_args()

    D, _ = load(args.raw, args.tok)
    n = len(D["date"])
    print(f"艇×レース {n:,}行\n")

    yr = (D["date"] // 10000 - ((D["date"] // 100) % 100 < 5)).astype(int)
    years = sorted(set(yr.tolist()))

    # 準優の有無とボーダー、予選最終日までの残り日数
    has_j = D["n_days"] >= 5
    border = np.where(has_j, 18.0, 6.0)
    last_yosen = np.where(has_j, D["n_days"] - 2, D["n_days"] - 1)
    days_left = last_yosen - D["day_no"]

    base = (D["is_yosen"] == 1) & (D["nruns"] >= args.min_runs) & (days_left >= 0)
    print(f"対象: 予選 かつ {args.min_runs}走以上 かつ 予選最終日まで  "
          f"{int(base.sum()):,}行\n")

    # ---------------- 0) 健全性 ----------------
    print("=" * 58)
    print("[0] データの健全性")
    print(f"  得点率  中央値 {np.median(D['tok'][base]):.2f}  "
          f"平均 {D['tok'][base].mean():.2f}")
    print(f"  節内順位 中央値 {np.median(D['srank'][base]):.0f}")
    print(f"  市場の1着確率 平均 {pc(D['q1'][base].mean())}  "
          f"実測1着率 {pc(D['won'][base].mean())}")
    print(f"  → この差が『市場は本命を過小評価』の分。以降は較正で外します")
    rows = []
    for nd in sorted(set(D["n_days"][base].astype(int).tolist())):
        m = base & (D["n_days"] == nd)
        if m.sum() < 500:
            continue
        rows.append([f"{nd}日制", f"{int(m.sum()):,}",
                     f"{int(border[m][0])}位", f"{int(last_yosen[m][0])}日目"])
    tbl(["節の長さ", "行数", "ボーダー", "予選最終日"], rows)

    # ---------------- 1) 較正 ----------------
    print("\n" + "=" * 58)
    print("[1] 市場の1着確率の較正 (この曲線からのズレを見る)")
    q = D["q1"]
    edges = np.quantile(q[base], np.linspace(0, 1, 21))
    edges = np.unique(edges)
    bins = np.clip(np.digitize(q, edges[1:-1]), 0, len(edges) - 2)
    cal = np.zeros(len(edges) - 1)
    rows = []
    for k in range(len(edges) - 1):
        m = base & (bins == k)
        if m.sum() < 100:
            cal[k] = q[m].mean() if m.sum() else 0.0
            continue
        cal[k] = D["won"][m].mean()
        if k % 4 == 0 or k == len(edges) - 2:
            rows.append([f"{edges[k]:.3f}〜{edges[k+1]:.3f}", f"{int(m.sum()):,}",
                         pc(q[m].mean()), pc(cal[k]),
                         f"{(cal[k]-q[m].mean())*100:+.1f}"])
    tbl(["市場q1の帯", "行数", "市場", "実測", "差(pt)"], rows)
    EXP = cal[bins]                     # 較正済みの期待1着確率

    def stat(m):
        k = int(m.sum())
        if k < 300:
            return None
        a = D["won"][m].mean()
        e = EXP[m].mean()
        mk = D["q1"][m].mean()          # 素の市場想定(買うときに直面する値)
        se = np.sqrt(max(a * (1 - a), 1e-9) / k)
        return {"n": k, "act": a, "exp": e, "d": a - e, "z": (a - e) / se,
                "roi": 0.75 * a / max(mk, 1e-9)}

    def show(title, groups, mask):
        print(f"\n  ◇ {title}")
        rows = []
        for label, g in groups:
            s = stat(mask & g)
            if not s:
                continue
            rows.append([label, f"{s['n']:,}", pc(s["exp"]), pc(s["act"]),
                         f"{s['d']*100:+.2f}", f"{s['z']:+.1f}", pc(s["roi"])])
        tbl(["区分", "行数", "較正の期待", "実測", "差(pt)", "z値", "想定回収率"], rows)

    # ---------------- 2) 動機グループ ----------------
    print("\n" + "=" * 58)
    print("[2] 動機グループ別  (差が0から離れていれば市場が織り込んでいない)")
    print("  z値は両側。2未満は偶然の範囲。ここでは符号を事前に決めていません")
    print("  想定回収率 = 75% × 実測1着率 ÷ 市場の1着確率")
    print("    その選手の頭を市場の配分どおりに買ったときの目安。")
    print("    100%に届くには『実測 ÷ 市場』が 1.33 以上必要です")

    tok = D["tok"]
    g_tok = [("7.5以上(当確)", tok >= 7.5),
             ("6.5〜7.5", (tok >= 6.5) & (tok < 7.5)),
             ("5.5〜6.5(ボーダー)", (tok >= 5.5) & (tok < 6.5)),
             ("4.5〜5.5", (tok >= 4.5) & (tok < 5.5)),
             ("4.5未満(圏外)", tok < 4.5)]
    show("得点率", g_tok, base)

    dist = D["srank"] - border
    g_rank = [("ボーダーより10位以上上", dist <= -10),
              ("-9〜-4", (dist > -10) & (dist <= -4)),
              ("-3〜+3(ボーダー圏)", (dist > -4) & (dist <= 3)),
              ("+4〜+10", (dist > 3) & (dist <= 10)),
              ("+11以上(圏外)", dist > 10)]
    show("ボーダーまでの順位差", g_rank, base & (D["srank"] > 0))

    g_day = [(f"予選最終日", days_left == 0),
             ("残り1日", days_left == 1),
             ("残り2日", days_left == 2),
             ("残り3日以上", days_left >= 3)]
    show("予選最終日まで", g_day, base)

    g_tr = [("節内に事故なし", D["trouble"] == 0),
            ("節内に事故あり", D["trouble"] == 1)]
    show("節内の事故(F・転覆等)", g_tr, base)

    # ---------------- 2.5) 節平均ST ----------------
    print("\n" + "=" * 58)
    print("[2.5] 節平均ST  (その節・その水面での実際のスタートの切れ)")
    hs = base & (D["st_s"] > 0)
    print(f"  節STが取れている行 {int(hs.sum()):,} / {int(base.sum()):,}")
    if hs.sum() < 5000:
        print("  ★データ不足。tokuten.py を --force で取り直してください")
    else:
        st = D["st_s"]
        g_st = [("0.10未満(抜群)", st < 0.10),
                ("0.10〜0.13", (st >= 0.10) & (st < 0.13)),
                ("0.13〜0.16", (st >= 0.13) & (st < 0.16)),
                ("0.16〜0.19", (st >= 0.16) & (st < 0.19)),
                ("0.19以上(鈍い)", st >= 0.19)]
        show("節平均STの絶対値", g_st, hs)

        # 期別平均STとの差 = 普段より切れているか
        both = hs & (D["st_avg"] > 0)
        dif = D["st_s"] - D["st_avg"]
        g_d = [("0.03以上速い", dif <= -0.03),
               ("0.01〜0.03速い", (dif > -0.03) & (dif <= -0.01)),
               ("ほぼ同じ", (dif > -0.01) & (dif < 0.01)),
               ("0.01〜0.03遅い", (dif >= 0.01) & (dif < 0.03)),
               ("0.03以上遅い", dif >= 0.03)]
        show("節ST − 期別平均ST (普段との差)", g_d, both)

        # レース内での相対
        print("\n  ◇ レース内で節STが最速か")
        rk = np.zeros(n)
        key = D["date"] * 10000 + D["jcd"] * 100 + D["rno"]
        order = np.lexsort((D["st_s"], key))
        uk, start = np.unique(key[order], return_index=True)
        for i in range(len(uk)):
            e_ = start[i + 1] if i + 1 < len(uk) else len(order)
            idxs = order[start[i]:e_]
            valid = [j for j in idxs if D["st_s"][j] > 0]
            for p_, j in enumerate(valid):
                rk[j] = p_ + 1
        g_rk = [(f"{k}位", rk == k) for k in range(1, 7)]
        show("節STのレース内順位", g_rk, hs & (rk > 0))

    # ---------------- 3) 予選最終日に限定 ----------------
    print("\n" + "=" * 58)
    print("[3] 予選最終日だけに絞る  (動機が最も強い日)")
    last = base & (days_left == 0)
    print(f"  対象 {int(last.sum()):,}行")
    show("得点率", g_tok, last)
    show("ボーダーまでの順位差", g_rank, last & (D["srank"] > 0))

    # ---------------- 4) 年度別 ----------------
    print("\n" + "=" * 58)
    print("[4] 年度別の再現  (4年度そろって同じ符号でなければ採用しない)")
    head = ["区分"] + [f"{y}年度" for y in years] + ["全体", "z値"]
    rows = []
    for label, g in g_tok + g_rank:
        m = base & g & (D["srank"] > 0)
        s = stat(m)
        if not s:
            continue
        line = [label[:16]]
        for y in years:
            t = stat(m & (yr == y))
            line.append(f"{t['d']*100:+.2f}" if t else "—")
        line += [f"{s['d']*100:+.2f}", f"{s['z']:+.1f}"]
        rows.append(line)
    tbl(head, rows)

    print("\n" + "=" * 58)
    print("判断の目安")
    print("  ・|z|が3未満なら採用しない(グループを20通り近く見ているので上振れる)")
    print("  ・4年度そろって同じ符号でなければ採用しない")
    print("  ・差が+1ポイント程度では、控除率25%は埋まらない")
    print("  ・想定回収率が100%未満なら、効果が本物でも金にはならない")
    print("  ・締切時オッズでの数字。実運用はこれより落ちる")


if __name__ == "__main__":
    main()
