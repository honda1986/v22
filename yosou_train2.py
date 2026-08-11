#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yosou_train2.py -- 1着・2着・3着の3段モデルを作る (Google Colab)

■ 考え方
  2着は「誰が1着か」で顔ぶれが変わる。
  1号艇が逃げたなら2着は差した2号艇かまくり差しの3号艇。
  3号艇がまくったなら2着は1号艇か4号艇。
  なので「1着艇との枠番差」「内側か外側か」を特徴量に入れて条件付きで学習する。

  p1(a)       … a が1着
  p2(b | a)   … a が1着のとき b が2着
  p3(c | a,b) … a,b のとき c が3着

  掛け合わせれば3連単120通りの確率が出る。
  各艇の 2着率・3着率・3連対率 も出せるので、
  1着確率が同じ5%で並んでも区別できるようになる。

■ 使う情報
  1着モデルと同じ46特徴量 + 1着艇(2着モデル)や2着艇(3着モデル)との関係

■ 正直な前提
  1着で市場に勝てなかったので、3連単でも勝てない(検証済み)。
  買い目を出す道具ではなく、表示を厚くするためのもの。

  使い方 (Colab)
    !pip -q install lightgbm
    !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
    %run v22/yosou_train2.py
"""

import glob
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("pip install lightgbm を先に実行してください")

import yosou_train as YT

OUT = "yosou_model"
# 1着艇・2着艇との差を取る指標
REL = ["n_win", "tok", "m_2ren", "avg_st", "st_setsu", "c_win"]
PARAMS = dict(objective="binary", learning_rate=0.04, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbose=-1, seed=42)


def load2():
    """着順(1〜3着)まで取る。hit文字列から作るので確実"""
    tokf = {os.path.basename(p)[:8]: p
            for p in glob.glob(os.path.join(YT.TOK, "*.json.gz"))}
    rawf = sorted(glob.glob(os.path.join(YT.RAW, "*.json.gz")))
    print(f"raw {len(rawf)}日 / tokuten {len(tokf)}日")
    rows = []
    t0 = time.time()
    for k, p in enumerate(rawf):
        d = os.path.basename(p)[:8]
        with gzip.open(p, "rt", encoding="utf-8") as f:
            rd = json.load(f)
        tk, meta = {}, {}
        if d in tokf:
            with gzip.open(tokf[d], "rt", encoding="utf-8") as f:
                td = json.load(f)
            for js, v in td.get("venues", {}).items():
                j = int(js)
                for r in v.get("races", []):
                    meta[(j, r["rno"])] = (r.get("name", ""), v.get("day_no"),
                                           v.get("n_days"))
                    for x in r["lanes"]:
                        tk[(j, r["rno"], x["lane"])] = x
        for r in rd["races"]:
            if "error" in r or not r.get("hit"):
                continue
            try:
                a, b, c = (int(z) for z in r["hit"].split("-"))
            except ValueError:
                continue
            name, day_no, n_days = meta.get((r["jcd"], r["rno"]), ("", None, None))
            if len(r["entries"]) != 6:
                continue
            for e in r["entries"]:
                x = tk.get((r["jcd"], r["rno"], e["lane"]), {})
                rows.append({
                    "date": int(d), "jcd": r["jcd"], "rno": r["rno"],
                    "race": f"{d}-{r['jcd']:02d}-{r['rno']}",
                    **{col: e.get(col) for col in YT.CARD},
                    "tok": x.get("tokuten"), "srank": x.get("rank"),
                    "genten": x.get("genten"), "nruns": x.get("n_runs"),
                    "st_setsu": x.get("st_setsu"), "c_win": x.get("c_win"),
                    "c_ren3": x.get("c_ren3"), "c_st": x.get("c_st"),
                    "day_no": day_no, "n_days": n_days,
                    "is_final": 1 if any(w in (name or "")
                                         for w in ("準優", "優勝", "選抜")) else 0,
                    "y": 1 if e["lane"] == a else 0,
                    "y2": 1 if e["lane"] == b else 0,
                    "y3": 1 if e["lane"] == c else 0,
                })
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(rawf)}日  {len(rows):,}行  "
                  f"{time.time()-t0:.0f}秒", flush=True)
    df = pd.DataFrame(rows)
    print(f"読み込み完了 {len(df):,}行 / {df['race'].nunique():,}レース  "
          f"{time.time()-t0:.0f}秒")
    return df


def YT2norm(v, g):
    uq, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, weights=v)
    return v / np.maximum(s[inv], 1e-12)


def to_cube(df, feats):
    """(レース, 6艇) の形に組み替える"""
    df = df.sort_values(["race", "lane"])
    n = df["race"].nunique()
    cube = {c: df[c].values.astype(np.float32).reshape(n, 6) for c in feats}
    y1 = df["y"].values.reshape(n, 6)
    y2 = df["y2"].values.reshape(n, 6)
    y3 = df["y3"].values.reshape(n, 6)
    date = df["date"].values.reshape(n, 6)[:, 0]
    return cube, y1, y2, y3, date, n


def build(cube, feats, n, fixed, target=None):
    """fixed = [1着の艇index] または [1着, 2着]。
    残りの艇を候補として、関係の特徴量を足した行列を作る"""
    reps = 6 - len(fixed)
    ridx = np.repeat(np.arange(n), 6)
    lane = np.tile(np.arange(6), n)
    keep = np.ones(n * 6, dtype=bool)
    for f in fixed:
        keep &= (lane != f[ridx])
    R, LN = ridx[keep], lane[keep]

    cols, names = [], []
    for c in feats:                          # 自分の特徴量
        cols.append(cube[c][R, LN]); names.append(c)
    for k, f in enumerate(fixed):            # 1着(と2着)との関係
        tag = ["w1", "w2"][k]
        fl = f[R]
        cols.append((LN - fl).astype(np.float32)); names.append(f"{tag}_lanediff")
        cols.append((LN < fl).astype(np.float32)); names.append(f"{tag}_inside")
        cols.append((fl + 1).astype(np.float32)); names.append(f"{tag}_lane")
        for c in REL:
            if c not in cube:
                continue
            cols.append(cube[c][R, LN] - cube[c][R, fl])
            names.append(f"{tag}_d_{c}")
        if "p1" in cube:
            cols.append(cube["p1"][R, fl]); names.append(f"{tag}_p1")
    X = np.column_stack(cols).astype(np.float32)
    y = target[R, LN] if target is not None else None
    return X, y, names, R, LN, reps


def norm_by(v, group, size):
    """group ごとに合計1に正規化"""
    s = np.bincount(group, weights=v, minlength=size)
    return v / np.maximum(s[group], 1e-12)


def main():
    df = load2()
    df, feats = YT.add_features(df)
    dates = np.sort(df["date"].unique())
    cut = dates[int(len(dates) * 0.75)]
    print(f"\n学習 〜{cut} / 検証 {cut}〜  特徴量 {len(feats)}個")

    cube, Y1, Y2, Y3, date, n = to_cube(df, feats)
    tr = date < cut
    va_cut = dates[int(len(dates) * 0.65)]
    print(f"レース数 学習{int(tr.sum()):,} / 検証{int((~tr).sum()):,}")

    # 場コードは順序に意味がないのでカテゴリとして扱う
    #   (数値のままだと「jcd<=5.5」のような範囲分割しかできず、
    #    戸田のような特殊水面を個別に学習できない)
    cat1 = [feats.index("jcd")] if "jcd" in feats else []

    # ---------------- 1着 ----------------
    print("\n--- 1着モデル ---")
    Xf = np.column_stack([cube[c].ravel() for c in feats]).astype(np.float32)
    rr = np.repeat(np.arange(n), 6)
    m_tr = np.repeat(date < va_cut, 6)
    m_va = np.repeat((date >= va_cut) & tr, 6)
    m1 = lgb.train(PARAMS,
                   lgb.Dataset(Xf[m_tr], Y1.ravel()[m_tr],
                               categorical_feature=cat1),
                   num_boost_round=3000,
                   valid_sets=[lgb.Dataset(Xf[m_va], Y1.ravel()[m_va],
                                           categorical_feature=cat1)],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
    p1 = norm_by(m1.predict(Xf).astype(np.float32), rr, n).reshape(n, 6)
    cube["p1"] = p1
    print(f"  木の数 {m1.best_iteration}")

    te0 = np.repeat(~tr, 6)
    pte = YT2norm(p1.ravel()[te0], np.repeat(np.arange(n), 6)[te0])
    yte = Y1.ravel()[te0]
    ll1 = -np.log(np.clip(pte[yte == 1], 1e-12, None)).mean()
    print(f"  検証での対数損失 {ll1:.4f}  (市場は約1.146)")

    w1 = np.argmax(Y1, axis=1)               # 実際の1着
    w2 = np.argmax(Y2, axis=1)               # 実際の2着

    # ---------------- 2着 ----------------
    print("\n--- 2着モデル (1着が誰かを条件にする) ---")
    X2, y2, n2names, R2, L2, _ = build(cube, feats, n, [w1], Y2)
    t2 = date[R2] < va_cut
    v2 = (date[R2] >= va_cut) & tr[R2]
    cat2 = [n2names.index("jcd")] if "jcd" in n2names else []
    m2 = lgb.train(PARAMS,
                   lgb.Dataset(X2[t2], y2[t2], categorical_feature=cat2),
                   num_boost_round=3000,
                   valid_sets=[lgb.Dataset(X2[v2], y2[v2],
                                           categorical_feature=cat2)],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"  木の数 {m2.best_iteration}  特徴量 {len(n2names)}個")
    imp = pd.Series(m2.feature_importance("gain"), index=n2names)
    print("  効いた特徴量 上位8:")
    for k, v in imp.sort_values(ascending=False).head(8).items():
        print(f"    {k:<18} {v:,.0f}")

    # ---------------- 3着 ----------------
    print("\n--- 3着モデル (1着と2着を条件にする) ---")
    X3, y3, n3names, R3, L3, _ = build(cube, feats, n, [w1, w2], Y3)
    t3 = date[R3] < va_cut
    v3 = (date[R3] >= va_cut) & tr[R3]
    cat3 = [n3names.index("jcd")] if "jcd" in n3names else []
    m3 = lgb.train(PARAMS,
                   lgb.Dataset(X3[t3], y3[t3], categorical_feature=cat3),
                   num_boost_round=3000,
                   valid_sets=[lgb.Dataset(X3[v3], y3[v3],
                                           categorical_feature=cat3)],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"  木の数 {m3.best_iteration}  特徴量 {len(n3names)}個")

    # ---------------- 精度 ----------------
    print("\n" + "=" * 54)
    print("[検証] 学習に使っていない期間")
    te = ~tr
    idx = np.where(te)[0]

    # 2着: 実際の1着を与えたときに当たるか
    sel2 = te[R2]
    pr2 = m2.predict(X2[sel2]).astype(np.float32)
    g2 = R2[sel2]
    uq, inv = np.unique(g2, return_inverse=True)
    pr2 = norm_by(pr2, inv, len(uq))
    yy2 = y2[sel2]
    ll2 = -np.log(np.clip(pr2[yy2 == 1], 1e-12, None)).mean()
    top2 = np.zeros(len(uq), dtype=bool)
    for i in range(len(uq)):
        m = inv == i
        top2[i] = yy2[m][np.argmax(pr2[m])] == 1
    print(f"  2着 対数損失 {ll2:.4f}  (5艇から均等なら {np.log(5):.4f})")
    print(f"     1番手の的中率 {top2.mean()*100:.1f}%  (均等なら20.0%)")

    sel3 = te[R3]
    pr3 = m3.predict(X3[sel3]).astype(np.float32)
    g3 = R3[sel3]
    uq3, inv3 = np.unique(g3, return_inverse=True)
    pr3 = norm_by(pr3, inv3, len(uq3))
    yy3 = y3[sel3]
    ll3 = -np.log(np.clip(pr3[yy3 == 1], 1e-12, None)).mean()
    top3 = np.zeros(len(uq3), dtype=bool)
    for i in range(len(uq3)):
        m = inv3 == i
        top3[i] = yy3[m][np.argmax(pr3[m])] == 1
    print(f"  3着 対数損失 {ll3:.4f}  (4艇から均等なら {np.log(4):.4f})")
    print(f"     1番手の的中率 {top3.mean()*100:.1f}%  (均等なら25.0%)")

    print("\n  ※ 2着・3着は『実際の1着(2着)を教えた上で』の成績です。")
    print("     実運用ではそこも予想するので、3連単の精度はこれより落ちます。")

    os.makedirs(OUT, exist_ok=True)
    m1.save_model(f"{OUT}/lgb_p1.txt")
    m2.save_model(f"{OUT}/lgb_p2.txt")
    m3.save_model(f"{OUT}/lgb_p3.txt")
    json.dump({"p1": feats, "p2": n2names, "p3": n3names, "rel": REL},
              open(f"{OUT}/features3.json", "w"))
    print(f"\n{OUT}/ に3つのモデルを保存しました")
    print("  lgb_p1.txt / lgb_p2.txt / lgb_p3.txt / features3.json")


if __name__ == "__main__":
    main()
