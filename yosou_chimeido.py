#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yosou_chimeido.py -- 有名な選手は買われすぎているか (Colab)

■ 発想
  これまで17回、いろいろな軸でレースを選んできた。
  「読みやすさ」で選ぶ軸(確率合計・本命の強さ・一致度・場・日目)は
  9回とも的中率は上がるが回収率は平らだった。読みやすいレースは
  市場も正しく値付けしているので当然。

  唯一動いたのが「払戻の水準」= 市場のどこが歪んでいるかの軸。
  そこで、市場の癖をもう1つ狙う。

  モデルは選手の「実力」しか見ていない。「知名度」は見ていない。
  峰竜太のようなSG常連は実力以上に買われるはず(競馬でいう武豊人気)。
  ここはモデルと市場が構造的に違う数少ない場所。

■ 知名度の代理変数(追加取得なしで作れるもの)
  n_race    直近1年の出走回数        … 露出が多い選手ほど覚えられる
  a1_rate   その期間のA1率           … 格の高さ
  win_rate  その期間の1着率          … 勝っている印象
  toban     登録番号(小さいほどベテラン)
  ★ tokuten に toban があるので、そこから数える(raw には無い)

■ 測ること
  本命艇の知名度で層別して、的中率と回収率を出す。
  「有名選手が本命のレースは回収率が低い」が仮説。
  逆が出れば「無名が強いレースは狙い目」ということになる。

  使い方 (Colab)
    !pip -q install lightgbm
    !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
    !cp -r v22/yosou_model .
    %run v22/yosou_chimeido.py
"""

import argparse
import glob
import gzip
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import yosou_train as YT
import yosou_train2 as YT2

MODEL = "yosou_model"
NPTS = 8


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


def load_fame(dfrom, window_days=365):
    """tokuten から、各日時点での『直近1年の出走数・A1率』を作る。
    その日より前のデータだけを使う(先の情報を見ない)"""
    files = sorted(glob.glob(os.path.join(YT.TOK, "*.json.gz")))
    hist = []            # (date, toban, cls) の並び
    for p in files:
        d = int(os.path.basename(p)[:8])
        with gzip.open(p, "rt", encoding="utf-8") as f:
            td = json.load(f)
        for v in td.get("venues", {}).values():
            for r in v.get("races", []):
                for x in r["lanes"]:
                    t = x.get("toban")
                    if t:
                        hist.append((d, int(t), x.get("cls")))
    H = pd.DataFrame(hist, columns=["date", "toban", "cls"])
    print(f"出走記録 {len(H):,}件 / 選手 {H['toban'].nunique():,}人")
    return H


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

    H = load_fame(args.dfrom)

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
        X, _, _, R_, L_, _ = YT2.build(cube, feats, n, [np.full(n, a)])
        P2[R_, a, L_] = np.maximum(m2.predict(X).astype(np.float32), 1e-9)
    P2 /= P2.sum(2, keepdims=True)
    P3 = np.zeros((n, 6, 6, 6))
    for a, b in pairs:
        X, _, _, R_, L_, _ = YT2.build(cube, feats, n,
                                       [np.full(n, a), np.full(n, b)])
        P3[R_, a, b, L_] = np.maximum(m3.predict(X).astype(np.float32), 1e-9)
    for a, b in pairs:
        P3[:, a, b] /= P3[:, a, b].sum(1, keepdims=True)
    CP = np.stack([p1[:, a] * P2[:, a, b] * P3[:, a, b, c] for a, b, c in tri], 1)
    names = [f"{a+1}-{b+1}-{c+1}" for a, b, c in tri]
    print("確率を計算しました")

    # tokuten から (date, jcd, rno, lane) → toban を索引にする
    tk = {}
    for p in sorted(glob.glob(os.path.join(YT.TOK, "*.json.gz"))):
        d = os.path.basename(p)[:8]
        if d < args.dfrom:
            continue
        with gzip.open(p, "rt", encoding="utf-8") as f:
            td = json.load(f)
        for js, v in td.get("venues", {}).items():
            for r in v.get("races", []):
                for x in r["lanes"]:
                    if x.get("toban"):
                        tk[(d, int(js), r["rno"], x["lane"])] = int(x["toban"])

    # オッズと結果
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
                r["hit"], r["pay_3t"], np.array(r["odds"], dtype=np.float64))

    # 知名度: その日より前の365日で数える
    H = H.sort_values("date")
    dates = sorted(set(H["date"].tolist()))
    rows = []
    miss = 0
    for i in range(n):
        k = keys[i]
        if k not in OD:
            continue
        d, jc, rn = k[:8], int(k[9:11]), int(k[12:])
        top = int(np.argmax(p1[i])) + 1
        tb = tk.get((d, jc, rn, top))
        if tb is None:
            miss += 1
            continue
        di = int(d)
        lo = di - 10000                       # ざっくり1年前
        sub = H[(H["toban"] == tb) & (H["date"] < di) & (H["date"] >= lo)]
        hit, pay, o = OD[k]
        cb = [names[j] for j in np.argsort(-CP[i])[:NPTS]]
        rows.append({
            "date": di, "toban": tb,
            "n_race": len(sub),
            "a1_rate": (sub["cls"] == "A1").mean() if len(sub) else np.nan,
            "top_p": p1[i].max(),
            "won": hit in cb, "pay": pay,
            "ret": pay if hit in cb else 0.0,
        })
    R = pd.DataFrame(rows)
    print(f"突き合わせ {len(R):,}レース  (登録番号が無く除外 {miss:,})\n")

    def stat(m):
        k = int(m.sum())
        if k < 300:
            return None
        v = R.loc[m, "ret"].values
        cost = k * NPTS * 100
        roi = v.sum() / cost
        se = np.sqrt(((v - roi * NPTS * 100) ** 2).sum()) / cost
        return {"n": k, "hit": R.loc[m, "won"].mean(), "roi": roi, "se": se,
                "pl": v.sum() - cost}

    print("=" * 60)
    print("[0] 対照: 全レースで上位8組を買う")
    s = stat(pd.Series(True, index=R.index))
    print(f"  {s['n']:,}レース  的中率 {pc(s['hit'])}  回収率 {pc(s['roi'])}"
          f" ± {pc(s['se'])}")

    print("\n[1] 本命艇の『直近1年の出走回数』で分ける")
    print("  出走が多い = 露出が多い = 覚えられている、という代理")
    qs = np.unique(np.quantile(R["n_race"].dropna(), np.linspace(0, 1, 6)))
    out = []
    for i in range(len(qs) - 1):
        m = (R["n_race"] >= qs[i]) & (R["n_race"] < qs[i + 1]
                                      if i < len(qs) - 2 else R["n_race"] <= qs[i + 1])
        s = stat(m)
        if s:
            out.append([f"{qs[i]:.0f}〜{qs[i+1]:.0f}走", f"{s['n']:,}",
                        pc(s["hit"]), pc(s["roi"]), pc(s["se"]),
                        f"{s['pl']:+,.0f}"])
    tbl(["直近1年の出走", "レース", "的中率", "回収率", "誤差±", "収支"], out)

    print("\n[2] 本命艇の『A1率』で分ける")
    A = R[R["a1_rate"].notna()]
    out = []
    for lo, hi in ((0, .01), (.01, .3), (.3, .7), (.7, .99), (.99, 1.01)):
        m = R.index.isin(A.index[(A["a1_rate"] >= lo) & (A["a1_rate"] < hi)])
        s = stat(pd.Series(m, index=R.index))
        if s:
            lab = ("A1経験なし" if hi <= .01 else
                   ("ずっとA1" if lo >= .99 else f"A1率 {lo*100:.0f}〜{hi*100:.0f}%"))
            out.append([lab, f"{s['n']:,}", pc(s["hit"]), pc(s["roi"]),
                        pc(s["se"]), f"{s['pl']:+,.0f}"])
    tbl(["本命のA1率", "レース", "的中率", "回収率", "誤差±", "収支"], out)

    print("\n[3] 登録番号で分ける(小さいほどベテラン)")
    qs = np.unique(np.quantile(R["toban"], np.linspace(0, 1, 6)))
    out = []
    for i in range(len(qs) - 1):
        m = (R["toban"] >= qs[i]) & (R["toban"] < qs[i + 1]
                                     if i < len(qs) - 2 else R["toban"] <= qs[i + 1])
        s = stat(m)
        if s:
            out.append([f"{qs[i]:.0f}〜{qs[i+1]:.0f}", f"{s['n']:,}",
                        pc(s["hit"]), pc(s["roi"]), pc(s["se"])])
    tbl(["登録番号", "レース", "的中率", "回収率", "誤差±"], out)

    print("\n" + "=" * 60)
    print("[4] 出走回数 × 本命の強さ  (回収率のみ)")
    tp = R["top_p"]
    hd = ["直近1年の出走"] + ["本命〜50%", "50〜65%", "65%〜"]
    qs = np.unique(np.quantile(R["n_race"].dropna(), [0, .33, .66, 1]))
    out = []
    for i in range(len(qs) - 1):
        line = [f"{qs[i]:.0f}〜{qs[i+1]:.0f}走"]
        base = (R["n_race"] >= qs[i]) & (R["n_race"] <= qs[i + 1])
        for a, b in ((0, .50), (.50, .65), (.65, 1.01)):
            s = stat(base & (tp >= a) & (tp < b))
            line.append(f"{s['roi']*100:.0f}±{s['se']*100:.0f}({s['n']})"
                        if s else "—")
        out.append(line)
    tbl(hd, out)

    print("\n[5] 年度で割る(出走が最も多い層と最も少ない層)")
    yr = (R["date"] // 10000 - ((R["date"] // 100) % 100 < 5)).astype(int)
    qs = np.quantile(R["n_race"].dropna(), [0.2, 0.8])
    out = []
    for lab, m in (("出走が少ない(下位20%)", R["n_race"] <= qs[0]),
                   ("出走が多い(上位20%)", R["n_race"] >= qs[1])):
        for y in sorted(set(yr.tolist())):
            s = stat(m & (yr == y))
            if s:
                out.append([f"{lab} {y}年度", f"{s['n']:,}", pc(s["hit"]),
                            pc(s["roi"]), pc(s["se"])])
    tbl(["区分", "レース", "的中率", "回収率", "誤差±"], out)

    print("\n" + "=" * 60)
    print("判断の目安")
    print("  ・有名(出走が多い・A1率が高い)ほど回収率が低ければ、仮説どおり")
    print("  ・差が2ポイント以内なら、知名度は織り込まれている")
    print("  ・年度がそろわない条件は使わない")
    print("  ・締切時オッズでの数字。実運用はこれより落ちる")


if __name__ == "__main__":
    main()
