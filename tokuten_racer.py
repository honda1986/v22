#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tokuten_racer.py -- 動機への反応は「人による」のかを検証する

■ なぜ必要か
  tokuten_analyze.py は全選手を平均していた。
  押す選手と引く選手が同数いれば、平均はゼロになる。
  実際にはボーダー圏で無理をする選手としない選手がいるはず、という指摘。

■ 検出力の限界(先に断っておく)
  21万行 ÷ 約1,600人 = 1人あたり134行。うちボーダー圏は約20行。
  個人ごとの推定はノイズだらけで、当てにならない。
  そこで個人を当てにいかず、次の2つで測る。

  1. 前期(2023-24年度)で選手ごとの反応を測り、5分位に束ねる。
     後期(2025-26年度)でその5分位が再現するか。
     → 束ねればノイズが潰れる。再現すれば「人による」が実在する。

  2. 級別 × 動機グループ。1区分3万行あるので検出力が高い。

■ 漏れ対策
  較正・期待値は tokuten_analyze.py と同じ。course_in は使わない。

  使い方: python tokuten_racer.py
"""

import argparse
import sys

import numpy as np

import tokuten_analyze as TA


def pc(x):
    return f"{x*100:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--tok", default="tokuten")
    ap.add_argument("--min-runs", type=int, default=2)
    ap.add_argument("--min-rows", type=int, default=40,
                    help="前期にこの行数以上ある選手だけを対象")
    ap.add_argument("--bd-lo", type=float, default=5.0)
    ap.add_argument("--bd-hi", type=float, default=7.0,
                    help="選手別の検証では帯を広げて1人あたりの行数を稼ぐ")
    ap.add_argument("--min-bd", type=int, default=8)
    args = ap.parse_args()

    D, _ = TA.load(args.raw, args.tok)
    n = len(D["date"])
    yr = (D["date"] // 10000 - ((D["date"] // 100) % 100 < 5)).astype(int)

    has_j = D["n_days"] >= 5
    border = np.where(has_j, 18.0, 6.0)
    last_yosen = np.where(has_j, D["n_days"] - 2, D["n_days"] - 1)
    days_left = last_yosen - D["day_no"]
    base = (D["is_yosen"] == 1) & (D["nruns"] >= args.min_runs) & (days_left >= 0)

    # 較正(tokuten_analyze と同じ手順)
    q = D["q1"]
    edges = np.unique(np.quantile(q[base], np.linspace(0, 1, 21)))
    bins = np.clip(np.digitize(q, edges[1:-1]), 0, len(edges) - 2)
    cal = np.zeros(len(edges) - 1)
    for k in range(len(edges) - 1):
        m = base & (bins == k)
        cal[k] = D["won"][m].mean() if m.sum() >= 100 else (
            q[m].mean() if m.sum() else 0.0)
    EXP = cal[bins]
    RES = D["won"] - EXP                    # 較正からのズレ

    tob = D["toban"].astype(np.int64)
    ok = base & (tob > 0)
    years = sorted(set(yr[ok].tolist()))
    half = years[:len(years) // 2] or years[:1]
    A = ok & np.isin(yr, half)              # 前期
    B = ok & ~np.isin(yr, half)             # 後期
    print(f"対象 {int(ok.sum()):,}行   前期{half} {int(A.sum()):,}行 / "
          f"後期 {int(B.sum()):,}行")
    print(f"選手数 {len(set(tob[ok].tolist())):,}人\n")

    bd = (D["tok"] >= args.bd_lo) & (D["tok"] < args.bd_hi)   # ボーダー圏(広め)

    def per_racer(mask):
        """選手ごとの (行数, ズレの平均) を返す"""
        t = tob[mask]
        r = RES[mask]
        uq, inv = np.unique(t, return_inverse=True)
        cnt = np.bincount(inv)
        ssum = np.bincount(inv, weights=r)
        return uq, cnt, ssum / np.maximum(cnt, 1)

    # ---------------- 1) 全体のズレは選手ごとに持続するか ----------------
    print("=" * 58)
    print("[1] 選手ごとの『市場とのズレ』は持続するか")
    print("  前期の成績で5分位に分け、後期で再現するかを見る")
    uA, cA, mA = per_racer(A)
    uB, cB, mB = per_racer(B)
    keep = cA >= args.min_rows
    uA, cA, mA = uA[keep], cA[keep], mA[keep]
    idx = {t: i for i, t in enumerate(uB)}
    sel = np.array([idx.get(t, -1) for t in uA])
    have = sel >= 0
    uA, cA, mA, sel = uA[have], cA[have], mA[have], sel[have]
    cB2, mB2 = cB[sel], mB[sel]
    print(f"  前期{args.min_rows}行以上かつ後期にも出走 {len(uA):,}人")

    if len(uA) < 100:
        print("  ★人数が少なすぎます。--min-rows を下げてください")
    else:
        qs = np.quantile(mA, np.linspace(0, 1, 6))
        gb = np.clip(np.digitize(mA, qs[1:-1]), 0, 4)
        rows = []
        for k in range(5):
            m = gb == k
            wA = (mA[m] * cA[m]).sum() / cA[m].sum()
            wB = (mB2[m] * cB2[m]).sum() / cB2[m].sum()
            seB = np.sqrt(0.18 * 0.82 / cB2[m].sum())
            rows.append([f"第{k+1}五分位", f"{int(m.sum()):,}",
                         f"{int(cA[m].sum()):,}", f"{wA*100:+.2f}",
                         f"{int(cB2[m].sum()):,}", f"{wB*100:+.2f}",
                         f"{wB/seB:+.1f}"])
        TA.tbl(["区分", "人数", "前期行数", "前期ズレ", "後期行数",
                "後期ズレ", "後期z"], rows)
        r = np.corrcoef(mA, mB2)[0, 1]
        print(f"  前期と後期の相関 {r:+.3f}")
        print("  第1と第5が後期でも同じ向きに離れていれば、選手差は実在する")

    # ---------------- 2) ボーダー反応の選手差 ----------------
    print("\n" + "=" * 58)
    print("[2] 『ボーダー圏でどう走るか』は選手ごとに違うか")
    print("  ボーダー圏のズレ − その選手の全体のズレ = ボーダー反応")
    uA2, cA2, mA2 = per_racer(A & bd)
    uB2, cB2b, mB2b = per_racer(B & bd)
    base_map = dict(zip(uA, mA))
    keep2 = cA2 >= args.min_bd
    uA2, cA2, mA2 = uA2[keep2], cA2[keep2], mA2[keep2]
    idx2 = {t: i for i, t in enumerate(uB2)}
    sel2 = np.array([idx2.get(t, -1) for t in uA2])
    have2 = (sel2 >= 0) & np.array([t in base_map for t in uA2])
    uA2, cA2, mA2, sel2 = uA2[have2], cA2[have2], mA2[have2], sel2[have2]
    respA = mA2 - np.array([base_map[t] for t in uA2])
    cB3, mB3 = cB2b[sel2], mB2b[sel2]
    print(f"  帯 {args.bd_lo}〜{args.bd_hi}  前期{args.min_bd}行以上かつ"
          f"後期にも該当 {len(uA2):,}人")
    print(f"  1人あたり前期 {cA2.mean():.0f}行 → "
          f"個人の推定誤差は ±{np.sqrt(0.18*0.82/cA2.mean())*100:.1f}ポイント")

    if len(uA2) < 100:
        print("  ★人数が少なすぎて判定できません")
    else:
        qs = np.quantile(respA, np.linspace(0, 1, 6))
        gb = np.clip(np.digitize(respA, qs[1:-1]), 0, 4)
        rows = []
        for k in range(5):
            m = gb == k
            wA = (respA[m] * cA2[m]).sum() / cA2[m].sum()
            wB = (mB3[m] * cB3[m]).sum() / cB3[m].sum()
            seB = np.sqrt(0.18 * 0.82 / max(cB3[m].sum(), 1))
            rows.append([f"第{k+1}五分位", f"{int(m.sum()):,}",
                         f"{int(cA2[m].sum()):,}", f"{wA*100:+.2f}",
                         f"{int(cB3[m].sum()):,}", f"{wB*100:+.2f}",
                         f"{wB/seB:+.1f}"])
        TA.tbl(["区分", "人数", "前期行数", "前期の反応", "後期行数",
                "後期ズレ", "後期z"], rows)
        avg = cB3.sum() / 5
        thr = 3 * np.sqrt(0.18 * 0.82 / max(avg, 1)) * 100
        print(f"  1五分位あたり後期 {avg:,.0f}行 → "
              f"z=3で拾えるのは {thr:.1f}ポイント以上の差")
        print("  前期で押した選手が後期も押していれば、『人による』が実在する")

    # ---------------- 3) 過分散の検定 ----------------
    print("\n" + "=" * 58)
    print("[3] 選手ごとのばらつきは、偶然より大きいか")
    uu, cc, mm = per_racer(ok)
    k2 = cc >= args.min_rows
    obs = np.var(mm[k2])
    expv = np.mean([0.18 * 0.82 / c for c in cc[k2]])
    print(f"  対象 {int(k2.sum()):,}人")
    print(f"  実測のばらつき(分散) {obs:.5f}")
    print(f"  偶然だけならこの値   {expv:.5f}")
    print(f"  比 {obs/expv:.2f}")
    if obs / expv > 1.15:
        print("  → 偶然より大きい。選手ごとの差が実在する可能性")
    else:
        print("  → 偶然の範囲。選手ごとの差は見えない")

    print("\n  ボーダー圏だけで見たばらつき")
    uu2, cc2, mm2 = per_racer(ok & bd)
    k3 = cc2 >= args.min_bd * 2
    if k3.sum() >= 50:
        obs2 = np.var(mm2[k3])
        exp2 = np.mean([0.18 * 0.82 / c for c in cc2[k3]])
        print(f"    対象 {int(k3.sum()):,}人  実測 {obs2:.5f} / 偶然 {exp2:.5f}"
              f"  比 {obs2/exp2:.2f}")
        print("    → 比が1.15を超えれば『ボーダーでの走り方に個人差がある』")
    else:
        print("    人数不足")

    # ---------------- 4) 級別 × 動機 ----------------
    print("\n" + "=" * 58)
    print("[4] 級別 × 動機グループ  (1区分が大きいので検出力が高い)")
    CLS = {4: "A1", 3: "A2", 2: "B1", 1: "B2"}
    groups = [("7.5以上(当確)", D["tok"] >= 7.5),
              ("5.5〜6.5(ボーダー)", bd),
              ("4.5未満(圏外)", D["tok"] < 4.5)]
    head = ["級別"] + [g[0] for g in groups]
    rows = []
    for c in (4, 3, 2, 1):
        line = [CLS[c]]
        for _, g in groups:
            m = base & g & (D["cls"] == c)
            k = int(m.sum())
            if k < 500:
                line.append("—")
                continue
            d = RES[m].mean()
            se = np.sqrt(max(D["won"][m].mean() * (1 - D["won"][m].mean()), 1e-9) / k)
            line.append(f"{d*100:+.2f} (z{d/se:+.1f})")
        rows.append(line)
    TA.tbl(head, rows)
    print("  数字はズレ(ポイント)。同じ級別の中で動機グループ間に差があるか")

    print("\n" + "=" * 58)
    print("判断の目安")
    print("  ・[1][2] は後期のz値と符号の一致だけを見る。前期は決め方なので当たって当然")
    print("  ・[3] の比が1.15未満なら、選手差を追う意味はない")
    print("  ・どれも1ポイント程度のズレでは控除率25%は埋まらない")


if __name__ == "__main__":
    main()
