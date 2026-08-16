#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yosou_nagashi.py -- 3着全流しは現行の買い方より良いか (Colab)

■ 考え方
  現行は「確率5%以上の組」を払戻均等で買う。3着まで指定するので
  1着2着が合っていても3着を外すと丸損になる(ヒモ抜け)。

  全流しは (1着,2着) のペアを単位にして、3着の4点を全部買う。
  ヒモ抜けが消える代わりに点数が増え、1本あたりの取り分が減る。
  どちらが勝つかを実測する。

  配分はどちらも払戻均等。yosou_bunpai.py で
  「配分では回収率は動かない」と分かっているので、
  差が出るとすれば買う組の選び方から。

■ 対象
  勝負レース(上位8組の確率合計 >= 0.553、払戻15,000〜20,000円)
  1レース1万円。締切時オッズ。

  使い方 (Colab)
    !pip -q install lightgbm
    !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
    !cp -r v22/yosou_model .
    %run v22/yosou_nagashi.py
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
BUDGET = 10000.0
BUY_P = 0.05                    # 現行: この確率以上の組を買う
SHOBU_TH = 0.553                # 勝負レース: 上位8組の確率合計
PAY_LO, PAY_HI = 15000, 20000   # 勝負レース: 1万円あたりの払戻


def pc(x):
    return f"{x*100:.1f}%"


def tbl(header, rows):
    if not rows:
        print("  (該当なし)")
        return
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows))
         for i, h in enumerate(header)]
    print("  " + "  ".join(str(h).rjust(w[i]) for i, h in enumerate(header)))
    for r in rows:
        print("  " + "  ".join(str(v).rjust(w[i]) for i, v in enumerate(r)))


def load_odds(dfrom):
    combos = None
    out = {}
    for p in sorted(glob.glob(os.path.join(YT.RAW, "*.json.gz"))):
        d = os.path.basename(p)[:8]
        if d < dfrom:
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
            out[f"{d}-{r['jcd']:02d}-{r['rno']}"] = (
                np.array(r["odds"], dtype=np.float64), r["hit"])
    return out, combos


def buy_eq(cn, o_all, cidx):
    """払戻均等配分。買い目リストから (当たったときの払戻X, オッズ配列) を返す"""
    od = np.array([o_all[cidx[c]] for c in cn])
    if not np.all(od > 0):
        return None, None
    return BUDGET / (1.0 / od).sum(), od


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="dfrom", default="20250314")
    ap.add_argument("--pairs", type=int, default=3,
                    help="全流しで試すペア数の上限")
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
        X, _, _, R_, L, _ = YT2.build(cube, feats, n, [np.full(n, a)])
        P2[R_, a, L] = np.maximum(m2.predict(X).astype(np.float32), 1e-9)
    P2 /= P2.sum(2, keepdims=True)
    P3 = np.zeros((n, 6, 6, 6))
    for a, b in pairs:
        X, _, _, R_, L, _ = YT2.build(cube, feats, n,
                                      [np.full(n, a), np.full(n, b)])
        P3[R_, a, b, L] = np.maximum(m3.predict(X).astype(np.float32), 1e-9)
    for a, b in pairs:
        P3[:, a, b] /= P3[:, a, b].sum(1, keepdims=True)
    CP = np.stack([p1[:, a] * P2[:, a, b] * P3[:, a, b, c]
                   for a, b, c in tri], 1)
    names = [f"{a+1}-{b+1}-{c+1}" for a, b, c in tri]
    print("確率を計算しました")

    # tri はペア単位で並んでいるので、reshape でペア確率が出る
    pair_p = CP.reshape(n, 30, 4).sum(2)
    pair_cn = np.array(names).reshape(30, 4)
    pair_nm = [f"{a+1}-{b+1}" for a, b in pairs]

    OD, combos = load_odds(args.dfrom)
    cidx = {c: i for i, c in enumerate(combos)}
    sum8 = np.sort(CP, 1)[:, -8:].sum(1)
    K = args.pairs

    rows = []
    for i in range(n):
        if keys[i] not in OD:
            continue
        o_all, hit = OD[keys[i]]

        sel = np.where(CP[i] >= BUY_P)[0]
        if len(sel) == 0:
            continue
        cn = [names[j] for j in sel]
        Xc, _ = buy_eq(cn, o_all, cidx)
        if Xc is None:
            continue

        rec = {"date": date[i], "sum8": sum8[i],
               "X_cur": Xc, "pts_cur": len(cn),
               "hit_cur": hit in cn,
               "p_cur": float(CP[i][sel].sum())}

        ordp = np.argsort(-pair_p[i])
        bad = False
        for k in range(1, K + 1):
            cnk = pair_cn[ordp[:k]].ravel().tolist()
            Xk, _ = buy_eq(cnk, o_all, cidx)
            if Xk is None:
                bad = True
                break
            rec[f"X_n{k}"] = Xk
            rec[f"hit_n{k}"] = hit in cnk
            rec[f"p_n{k}"] = float(pair_p[i][ordp[:k]].sum())
        if bad:
            continue
        rec["top_pair"] = pair_nm[ordp[0]]
        rows.append(rec)

    R = pd.DataFrame(rows)
    print(f"対象 {len(R):,}件\n")

    def stat(m, col_X, col_hit, minn=150):
        k = int(m.sum())
        if k < minn:
            return None
        won = R.loc[m, col_hit].values
        ret = np.where(won, R.loc[m, col_X].values, 0.0)
        cost = k * BUDGET
        roi = ret.sum() / cost
        se = np.sqrt(((ret - roi * BUDGET) ** 2).sum()) / cost
        return {"n": k, "hit": won.mean(), "roi": roi, "se": se,
                "z": (roi - 1.0) / se if se > 0 else 0.0,
                "X": R.loc[m, col_X].mean(), "pl": ret.sum() - cost}


    def line(label, s, pts=None):
        return [label, f"{s['n']:,}", pts if pts else "",
                f"{s['X']:,.0f}", pc(s["hit"]), pc(s["roi"]),
                pc(s["se"]), f"{s['z']:+.1f}", f"{s['pl']:+,.0f}"]

    HEAD = ["買い方", "レース", "点数", "平均払戻", "的中率",
            "回収率", "誤差±", "z", "収支"]

    # ---------------------------------------------------------
    shobu = ((R["sum8"] >= SHOBU_TH) & (R["X_cur"] >= PAY_LO)
             & (R["X_cur"] < PAY_HI))

    print("=" * 72)
    print("[0] ベースラインの再現確認")
    print("    既知の値: 1,333レースで 的中率53.5% / 回収率89.6%(±2.3)")
    print("=" * 72)
    s = stat(shobu, "X_cur", "hit_cur")
    if s:
        tbl(HEAD, [line("現行(確率5%以上)", s,
                        f"{R.loc[shobu, 'pts_cur'].mean():.1f}")])
        print("\n  ここが既知の値から大きくずれていれば、"
              "以下の比較も信用できません。")
    else:
        print("  件数不足。--from を早めてください。")

    # ---------------------------------------------------------
    print("\n" + "=" * 72)
    print("[1] 同じ勝負レースで、買い方だけ替える  ← 本命の比較")
    print("=" * 72)
    out = []
    if s:
        out.append(line("現行(確率5%以上)", s,
                        f"{R.loc[shobu, 'pts_cur'].mean():.1f}"))
    for k in range(1, K + 1):
        sk = stat(shobu, f"X_n{k}", f"hit_n{k}")
        if sk:
            out.append(line(f"全流し {k}ペア", sk, f"{4*k}"))
    tbl(HEAD, out)
    print("\n  レースの選び方は現行のまま。買い目だけ差し替えた比較です。")

    # ---------------------------------------------------------
    print("\n" + "=" * 72)
    print("[2] 全流しの側で払戻の窓を選び直す")
    print("    現行の窓(15,000〜20,000円)が全流しにも合うとは限らない")
    print("=" * 72)
    for k in range(1, K + 1):
        out = []
        for lo, hi in ((0, 8000), (8000, 12000), (12000, 15000),
                       (15000, 20000), (20000, 30000), (30000, 10**9)):
            m = ((R["sum8"] >= SHOBU_TH) & (R[f"X_n{k}"] >= lo)
                 & (R[f"X_n{k}"] < hi))
            sk = stat(m, f"X_n{k}", f"hit_n{k}")
            if sk:
                out.append(line(f"{lo:,}〜{hi:,}円" if hi < 10**8
                                else "30,000円〜", sk, f"{4*k}"))
        if out:
            print(f"\n  全流し {k}ペア({4*k}点)")
            tbl(HEAD, out)

    # ---------------------------------------------------------
    print("\n" + "=" * 72)
    print("[3] 年度で割る  (符号がそろわなければ採用しない)")
    print("=" * 72)
    yr = (R["date"] // 10000 - ((R["date"] // 100) % 100 < 5)).astype(int)
    head = ["年度", "現行"] + [f"全流し{k}ペア" for k in range(1, K + 1)]
    out = []
    for y in sorted(set(yr.tolist())):
        m = shobu & (yr == y)
        row = [f"{y}年度"]
        sy = stat(m, "X_cur", "hit_cur", 80)
        row.append(f"{sy['roi']*100:.0f}±{sy['se']*100:.0f}({sy['n']})"
                   if sy else "—")
        for k in range(1, K + 1):
            sk = stat(m, f"X_n{k}", f"hit_n{k}", 80)
            row.append(f"{sk['roi']*100:.0f}±{sk['se']*100:.0f}({sk['n']})"
                       if sk else "—")
        out.append(row)
    tbl(head, out)

    # ---------------------------------------------------------
    print("\n" + "=" * 72)
    print("[4] 参考: 勝負レース以外でも見る")
    print("=" * 72)
    out = []
    for label, m in (("全レース", pd.Series(True, index=R.index)),
                     ("確率合計0.553以上", R["sum8"] >= SHOBU_TH),
                     ("勝負レース", shobu)):
        sc = stat(m, "X_cur", "hit_cur")
        if sc:
            out.append(line(f"{label} / 現行", sc, ""))
        for k in range(1, K + 1):
            sk = stat(m, f"X_n{k}", f"hit_n{k}")
            if sk:
                out.append(line(f"{label} / 全流し{k}ペア", sk, f"{4*k}"))
    tbl(HEAD, out)

    # ---------------------------------------------------------
    print("\n" + "=" * 72)
    print("[5] 参考: ヒモ抜けがどれだけ拾えたか")
    print("=" * 72)
    m = shobu
    nm = int((~R.loc[m, "hit_cur"] & R.loc[m, "hit_n2"]).sum())
    lost = int((R.loc[m, "hit_cur"] & ~R.loc[m, "hit_n2"]).sum())
    both = int((R.loc[m, "hit_cur"] & R.loc[m, "hit_n2"]).sum())
    print(f"  勝負レース {int(m.sum()):,}件のうち")
    print(f"    どちらも的中          {both:,}件")
    print(f"    全流し2ペアだけ的中    {nm:,}件  ← ヒモ抜けを拾った分")
    print(f"    現行だけ的中          {lost:,}件  ← ペアから外れた分")
    print(f"  的中率の差 {(nm - lost) / max(int(m.sum()), 1) * 100:+.1f}pt")

    print("\n" + "=" * 72)
    print("判断の目安")
    print("  ・[0] が 53.5% / 89.6% 付近でなければ、以下は読まない")
    print("  ・[1] で回収率が現行を上回り、かつ |z| が3以上で、")
    print("    [3] の年度で符号がそろって初めて意味がある")
    print("  ・的中率だけ上がって回収率が同じなら、単に点数を増やしただけ")
    print("  ・締切時オッズでの数字。実運用はこれより落ちる")


if __name__ == "__main__":
    main()
