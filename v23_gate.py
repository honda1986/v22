#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v23_gate.py -- v23を作る前の関門テスト (Google Colab で実行)

■ 何を確かめるか
  v22 は市場(オッズ)を一切見ずに確率を出していた。結果、市場に勝てなかった。
    対数損失  モデル 3.8137  /  市場 3.7126
    配合の重み a(モデル)=0.14  b(市場)=0.96

  v23 の設計は「市場の確率を特徴量に入れて、市場がどこで間違っているかを学習させる」。
  それが本当に効くのかを、いちばん簡単な『1着を当てる』問題だけで先に測る。

  A 市場のみ      オッズから作った1着確率だけ
  B ファンダのみ  v22と同じ考え方(オッズを見ない)
  C ファンダ+市場 v23の形

  CがAを有意に下回れば作る価値がある。下回らなければ3連単は絶対に無理。

■ 漏れ対策
  raw の course_in は本番進入(レース後にしか分からない)と判明したので使わない。
  進入コース・前付け・コース差分はすべて特徴量から外す。

■ 使い方 (Colab)
  !git clone --depth 1 https://github.com/honda1986/v22.git
  !pip -q install lightgbm
  %run v22/v23_gate.py
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

# ============================================================ 設定
RAW_DIR = "v22/raw" if os.path.isdir("v22/raw") else "raw"
TRAIN_END = "20250430"      # ここまでで学習
TEST_START = "20260101"     # ここから検証(間は空ける)
OUT_DIR = "v23_out"

EV_MIN, TOP_N, MAX_POINTS = 1.10, 12, 8

BASE = ["lane", "cls_val", "age", "weight", "f_count", "avg_st",
        "n_win", "n_2ren", "l_win", "l_2ren", "m_2ren", "b_2ren", "tenji"]
MKT = ["q1", "q2", "q3", "q1_rank", "q1_logit", "q1_share"]


# ============================================================ 読み込み
def combo_position_index(combos):
    """各(着順位置, 艇番)に対応する組番のインデックス"""
    idx = [[[] for _ in range(7)] for _ in range(3)]
    for i, c in enumerate(combos):
        for pos, b in enumerate(int(x) for x in c.split("-")):
            idx[pos][b].append(i)
    return [[np.array(idx[p][b]) for b in range(7)] for p in range(3)]


def load(raw_dir):
    files = sorted(glob.glob(os.path.join(raw_dir, "*.json.gz")))
    if not files:
        sys.exit(f"{raw_dir} にデータがありません")
    print(f"{len(files)}日分を読み込みます")

    combos = None
    ent_rows, odds_list, meta = [], [], []
    t0 = time.time()
    for k, p in enumerate(files):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            d = json.load(f)
        if combos is None:
            combos = d["combos"]
        for r in d["races"]:
            if "error" in r or r.get("n_odds") != 120:
                continue
            if not r.get("hit") or not r.get("pay_3t"):
                continue
            ri = len(meta)
            w1 = int(r["hit"].split("-")[0])
            for e in r["entries"]:
                ent_rows.append((int(r["date"]), r["jcd"], r["rno"], ri,
                                 *[e.get(f) for f in BASE],
                                 1 if e["lane"] == w1 else 0))
            odds_list.append(np.array(r["odds"], dtype=np.float32))
            meta.append((int(r["date"]), r["hit"], r["pay_3t"]))
        if (k + 1) % 150 == 0:
            print(f"  {k+1}/{len(files)}日  {len(meta):,}レース  "
                  f"{time.time()-t0:.0f}秒", flush=True)

    ODDS = np.stack(odds_list)
    del odds_list
    df = pd.DataFrame(ent_rows,
                      columns=["date", "jcd", "rno", "race"] + BASE + ["y"])
    del ent_rows

    # 市場の着順位置ごとの確率をまとめて計算(1レースずつ回すと遅い)
    M = np.zeros((3, 120, 6), dtype=np.float32)
    for i, c in enumerate(combos):
        for pos, b in enumerate(int(x) for x in c.split("-")):
            M[pos, i, b - 1] = 1.0
    inv = 1.0 / ODDS
    Q = inv / inv.sum(1, keepdims=True)
    for pos, name in enumerate(("q1", "q2", "q3")):
        qp = Q @ M[pos]                      # レース×艇
        df[name] = qp[df["race"].values, df["lane"].values - 1]
    del inv, Q

    print(f"読み込み完了 {len(meta):,}レース / {len(df):,}行  "
          f"{time.time()-t0:.0f}秒")
    return df, ODDS, meta, combos


# ============================================================ 特徴量
def add_features(df):
    """レース内での相対化。進入コース関連は漏れるので一切使わない。"""
    df = df.copy()
    df.loc[df["tenji"] <= 0, "tenji"] = np.nan

    g = df.groupby("race")
    for col, asc in [("n_win", False), ("m_2ren", False),
                     ("avg_st", True), ("tenji", True)]:
        df[f"{col}_dev"] = df[col] - g[col].transform("mean")
        df[f"{col}_rank"] = g[col].rank(ascending=asc, method="min")

    df = df.sort_values(["race", "lane"]).reset_index(drop=True)
    for col in ["avg_st", "n_win", "tenji"]:
        s = df.groupby("race")[col]
        df[f"{col}_diff_in"] = (df[col] - s.shift(1)).fillna(0.0)
        df[f"{col}_diff_out"] = (df[col] - s.shift(-1)).fillna(0.0)

    # 市場側の相対量
    df["q1_rank"] = df.groupby("race")["q1"].rank(ascending=False, method="min")
    df["q1_logit"] = np.log(np.clip(df["q1"], 1e-6, 1 - 1e-6) /
                            (1 - np.clip(df["q1"], 1e-6, 1 - 1e-6)))
    df["q1_share"] = df["q1"] / df.groupby("race")["q1"].transform("max")

    feats = (BASE + ["jcd", "rno"] +
             [f"{c}_dev" for c in ["n_win", "m_2ren", "avg_st", "tenji"]] +
             [f"{c}_rank" for c in ["n_win", "m_2ren", "avg_st", "tenji"]] +
             [f"{c}_diff_{d}" for c in ["avg_st", "n_win", "tenji"]
              for d in ["in", "out"]])
    return df, feats


# ============================================================ 学習
def train(df, feats, tr, va):
    p = dict(objective="binary", learning_rate=0.04, num_leaves=63,
             min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, verbose=-1, seed=42)
    ds_tr = lgb.Dataset(df.loc[tr, feats], df.loc[tr, "y"])
    ds_va = lgb.Dataset(df.loc[va, feats], df.loc[va, "y"])
    return lgb.train(p, ds_tr, num_boost_round=3000, valid_sets=[ds_va],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def race_probs(raw, race_ids):
    """艇ごとの生スコアをレース内で正規化して1着確率にする"""
    s = pd.Series(raw).groupby(race_ids).transform("sum").values
    return raw / np.maximum(s, 1e-12)


def logloss(p, y, race_ids):
    """的中艇の確率の負の対数(レース単位)"""
    d = pd.DataFrame({"p": p, "y": y, "r": race_ids})
    w = d[d["y"] == 1]
    return float(-np.log(np.clip(w["p"].values, 1e-12, None)).mean())


# ============================================================ main
def main():
    df, ODDS, meta, combos = load(RAW_DIR)
    df, feats = add_features(df)

    dates = df["date"].values
    is_tr = dates <= int(TRAIN_END)
    is_te = dates >= int(TEST_START)
    print(f"\n学習 {is_tr.sum()//6:,}レース (〜{TRAIN_END})")
    print(f"検証 {is_te.sum()//6:,}レース ({TEST_START}〜)")
    if is_te.sum() == 0:
        sys.exit("検証期間のデータがありません")

    # 学習期間の後ろ15%を早期終了用に取る
    tr_dates = np.sort(np.unique(dates[is_tr]))
    cut = tr_dates[int(len(tr_dates) * 0.85)]
    tr = is_tr & (dates < cut)
    va = is_tr & (dates >= cut)
    print(f"  うち早期終了用 {va.sum()//6:,}レース ({cut}〜)")

    sets = {"A 市場のみ": MKT,
            "B ファンダのみ": feats,
            "C ファンダ+市場": feats + MKT}

    rid = df["race"].values
    y = df["y"].values
    res, probs = {}, {}
    for name, cols in sets.items():
        print(f"\n--- {name} ({len(cols)}特徴量) ---")
        m = train(df, cols, tr, va)
        raw = m.predict(df.loc[is_te, cols])
        p = race_probs(raw, rid[is_te])
        probs[name] = p
        res[name] = logloss(p, y[is_te], rid[is_te])
        print(f"  木の数 {m.best_iteration}   検証側の対数損失 {res[name]:.4f}")
        if name == "C ファンダ+市場":
            imp = pd.Series(m.feature_importance("gain"), index=cols)
            print("  効いた特徴量 上位10:")
            for k, v in imp.sort_values(ascending=False).head(10).items():
                print(f"    {k:<18} {v:,.0f}")
            os.makedirs(OUT_DIR, exist_ok=True)
            m.save_model(f"{OUT_DIR}/lgb_p1_v23.txt")
            json.dump(cols, open(f"{OUT_DIR}/lgb_p1_v23_features.json", "w"))

    # 素の市場確率そのもの(モデルを通さない)
    q_only = df.loc[is_te, "q1"].values
    q_only = race_probs(q_only, rid[is_te])
    res["  参考 素の市場確率"] = logloss(q_only, y[is_te], rid[is_te])

    print("\n" + "=" * 54)
    print("[関門] 1着の対数損失(小さいほど当てている)")
    for k in ["A 市場のみ", "B ファンダのみ", "C ファンダ+市場", "  参考 素の市場確率"]:
        print(f"  {k:<20} {res[k]:.4f}")
    gain = res["A 市場のみ"] - res["C ファンダ+市場"]
    print(f"\n  C が A を {gain:+.4f} 上回っています")
    n_race = int(is_te.sum() // 6)
    se = 1.0 / np.sqrt(n_race)
    print(f"  検証 {n_race:,}レース  目安の誤差 ±{se:.4f}")
    if gain > 2 * se:
        print("  → 市場を有意に超えています。v23を作る価値があります。")
    elif gain > 0:
        print("  → 超えていますが誤差の範囲。判断保留。")
    else:
        print("  → 市場を超えていません。3連単は無理です。")

    # ---------------- 実利の確認 ----------------
    print("\n" + "=" * 54)
    print("[実利] 市場の3連単確率の『1着部分だけ』を差し替えて買う")
    print("  p(a-b-c) = 市場の確率 × モデルの1着確率 / 市場の1着確率")
    print("  勝てるはずの部分だけ直す、いちばん素直な使い方")

    race_ids_te = np.unique(rid[is_te])
    pos_map = {r: i for i, r in enumerate(race_ids_te)}
    O = ODDS[race_ids_te]
    inv = 1.0 / O
    Q = inv / inv.sum(1, keepdims=True)
    pidx = combo_position_index(combos)
    first_of = np.array([int(c.split("-")[0]) for c in combos])

    hit_i = np.array([combos.index(meta[r][1]) for r in race_ids_te])
    pay = np.array([meta[r][2] for r in race_ids_te], dtype=np.float64)

    q1_race = np.zeros((len(race_ids_te), 6))
    for b in range(1, 7):
        q1_race[:, b - 1] = Q[:, pidx[0][b]].sum(1)

    def build(pname):
        p = probs[pname].reshape(-1, 6)      # レース×艇 (laneでソート済み)
        ratio = p / np.maximum(q1_race, 1e-9)
        return Q * ratio[:, first_of - 1]

    def roi(Pc):
        order = np.argsort(-Pc, axis=1)[:, :TOP_N]
        ev = np.take_along_axis(Pc, order, 1) * np.take_along_axis(O, order, 1)
        ht = (order == hit_i[:, None])
        ok0 = ev >= EV_MIN
        sel = np.argsort(-np.where(ok0, ev, -1e9), axis=1)[:, :MAX_POINTS]
        ok = np.take_along_axis(ok0, sel, 1)
        got = (np.take_along_axis(ht * pay[:, None], sel, 1) * ok).sum(1)
        cnt = ok.sum(1)
        cost = cnt.sum() * 100.0
        if cost < 10000:
            return None
        r = got.sum() / cost
        se2 = np.sqrt(((got - r * cnt * 100) ** 2).sum()) / cost
        return (int((cnt >= 1).sum()), int(cnt.sum()), r, se2,
                got.sum() - cost)

    for name in ["B ファンダのみ", "C ファンダ+市場"]:
        s = roi(build(name))
        if s is None:
            print(f"  {name:<16} 買い目なし")
        else:
            n_r, n_p, r, se2, pl = s
            print(f"  {name:<16} {n_r:,}レース {n_p:,}点  "
                  f"回収率 {r*100:.1f}% ± {se2*100:.1f}%  "
                  f"z={(r-1)/se2:+.1f}  収支 {pl:+,.0f}円")

    print("\n" + "=" * 54)
    print("注意: 締切時オッズで計算しています。実運用は必ずこれより落ちます。")
    print(f"モデルは {OUT_DIR}/ に保存しました。")


if __name__ == "__main__":
    main()
