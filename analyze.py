#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze.py -- raw/ の過去レースに predict.py のモデルを当てて買い目条件を検証する

やること
  0) predict.py 本体と同じ確率が出るか自動照合(ここがズレたら以降は全部無意味)
  1) キャリブレーション   予測20%のレースは本当に20%当たるか
  2) 市場の歪み           オッズ帯別の実測回収率(穴が過大人気かを実測)
  3) 確率順位別           1位〜15位を1点買いした場合の的中率と回収率
  4) EV帯別               EV_MIN=1.10 が妥当かを実測で決める
  5) 現行ルールの成績     EV_MIN/TOP_N_PROB/MAX_POINTS をそのまま適用
  6) グリッド探索         3つの設定を総当たり
  7) 期間分割             前半で決めた設定が後半でも生きるか(これが本番)
  8) 軸別                 場/R/風/本命度 → 勝負レースの絞り込み候補

使い方
  python analyze.py                 # 全部
  python analyze.py --limit 2000    # 動作確認用に2000レースだけ
  python analyze.py --verify 100    # 照合レース数(既定100)
"""

import argparse
import glob
import gzip
import json
import os
import sys
import time

import numpy as np

import predict as P

F1 = P.features_p1
F2 = P.features_p2
F3 = P.features_p3
W1F = ["lane", "cls_val", "avg_st", "n_win", "m_2ren", "tenji", "course_in", "maezuke"]
W2F = ["lane", "cls_val", "avg_st", "n_win", "m_2ren", "tenji", "course_in"]

ROW_KEYS = ["lane", "cls_val", "age", "weight", "f_count", "avg_st",
            "n_win", "n_2ren", "l_win", "l_2ren", "m_2ren", "b_2ren",
            "tenji", "course_in"]

OUT_LINES = []


def say(s=""):
    print(s, flush=True)
    OUT_LINES.append(s)


# ============================================================ 確率計算(バッチ版)
def _recs(df, jcd):
    df = df.copy()
    df["jcd"] = jcd
    cols = list(df.columns)
    out = {}
    for _, r in df.iterrows():
        out[int(r["lane"])] = {c: r[c] for c in cols}
    return out


def combo_probs_fast(feats_df, jcd, combo_idx):
    """predict.predict_combo_probs と同じ値を、3回のpredictで出す。"""
    recs = _recs(feats_df, jcd)

    X1 = np.array([[recs[l][c] for c in F1] for l in range(1, 7)], dtype=float)
    r1 = np.asarray(P.m_p1.predict(X1), dtype=float)
    s1 = r1.sum()
    p1 = r1 / s1 if s1 > 0 else r1

    X2, key2 = [], []
    for w1 in range(1, 7):
        wr = recs[w1]
        for cand in range(1, 7):
            if cand == w1:
                continue
            cr = recs[cand]
            feat = {f: cr[f] for f in F1 if f in cr}
            for f in W1F:
                feat["w1_" + f] = wr[f]
            feat["w1_lane_diff"] = cr["lane"] - wr["lane"]
            feat["w1_course_diff"] = cr["course_in"] - wr["course_in"]
            X2.append([feat.get(c, 0.0) for c in F2])
            key2.append((w1, cand))
    r2 = np.asarray(P.m_p2.predict(np.array(X2, dtype=float)), dtype=float)
    p2 = {}
    for b in range(6):
        seg = r2[b * 5:(b + 1) * 5]
        ss = seg.sum()
        for k in range(5):
            w1, cand = key2[b * 5 + k]
            p2[(w1, cand)] = seg[k] / ss if ss > 0 else 0.0

    X3, key3 = [], []
    for w1 in range(1, 7):
        wr = recs[w1]
        for w2 in range(1, 7):
            if w2 == w1:
                continue
            w2r = recs[w2]
            for cand in range(1, 7):
                if cand in (w1, w2):
                    continue
                cr = recs[cand]
                feat = {f: cr[f] for f in F1 if f in cr}
                for f in W1F:
                    feat["w1_" + f] = wr[f]
                feat["w1_lane_diff"] = cr["lane"] - wr["lane"]
                feat["w1_course_diff"] = cr["course_in"] - wr["course_in"]
                for f in W2F:
                    feat["w2_" + f] = w2r[f]
                feat["w2_lane_diff"] = cr["lane"] - w2r["lane"]
                X3.append([feat.get(c, 0.0) for c in F3])
                key3.append((w1, w2, cand))
    r3 = np.asarray(P.m_p3.predict(np.array(X3, dtype=float)), dtype=float)

    out = np.zeros(120, dtype=np.float64)
    for b in range(30):
        seg = r3[b * 4:(b + 1) * 4]
        ss = seg.sum()
        for k in range(4):
            w1, w2, w3 = key3[b * 4 + k]
            v = seg[k] / ss if ss > 0 else 0.0
            out[combo_idx[f"{w1}-{w2}-{w3}"]] = p1[w1 - 1] * p2[(w1, w2)] * v
    return out


def race_rows(race):
    return [{k: e[k] for k in ROW_KEYS} for e in race["entries"]]


# ============================================================ 読み込み
def load(outdir, limit=None, dfrom=None, dto=None, exf=None, ext=None):
    files = sorted(glob.glob(os.path.join(outdir, "*.json.gz")))
    if not files:
        sys.exit(f"{outdir}/ にデータがありません")
    combos = None
    races = []
    for p in files:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            d = json.load(f)
        if combos is None:
            combos = d["combos"]
        if dfrom and d["date"] < dfrom:
            continue
        if dto and d["date"] > dto:
            continue
        if exf and ext and exf <= d["date"] <= ext:
            continue
        for r in d["races"]:
            if "error" in r:
                continue
            if r.get("n_odds") != 120 or not r.get("hit") or not r.get("pay_3t"):
                continue
            races.append(r)
        if limit and len(races) >= limit:
            break
    if limit:
        races = races[:limit]
    return races, combos


# ============================================================ 表示補助
def tbl(header, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(header)]
    say("  " + "  ".join(str(h).rjust(widths[i]) for i, h in enumerate(header)))
    for r in rows:
        say("  " + "  ".join(str(v).rjust(widths[i]) for i, v in enumerate(r)))


def pc(x):
    return f"{x*100:.1f}%"


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="raw")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--from", dest="dfrom", help="YYYYMMDD この日以降だけ使う")
    ap.add_argument("--to", dest="dto", help="YYYYMMDD この日以前だけ使う")
    ap.add_argument("--ex-from", dest="exf", default="20250501",
                    help="学習期間の開始。この範囲は自動で除外する")
    ap.add_argument("--ex-to", dest="ext", default="20260430",
                    help="学習期間の終了")
    ap.add_argument("--split", help="前半/後半の境目 YYYYMMDD(省略時はレース数で半分)")
    ap.add_argument("--train-end", default="20260501",
                    help="モデルの学習データの最終日。これ以前は学習済みなので警告を出す")
    ap.add_argument("--verify", type=int, default=100)
    ap.add_argument("--md", default="analysis.md")
    args = ap.parse_args()

    if not (P.m_p1 and P.m_p2 and P.m_p3):
        sys.exit("モデルが読み込めません(lgb_p*_v22.txt を確認)")

    races, combos = load(args.out, args.limit, args.dfrom, args.dto,
                         args.exf, args.ext)
    if not races:
        sys.exit("該当するレースがありません(--from/--to を確認)")
    combo_idx = {c: i for i, c in enumerate(combos)}
    d0, d1 = races[0]["date"], races[-1]["date"]
    say(f"対象レース {len(races):,}件  ({d0} 〜 {d1})")
    if args.exf and args.ext:
        say(f"学習期間 {args.exf}〜{args.ext} は除外済み")
    if args.train_end and d0 <= args.train_end and not (args.exf and args.ext):
        leak = sum(1 for r in races if r["date"] <= args.train_end)
        say(f"★警告: 学習データ期間(〜{args.train_end})のレースが {leak:,}件 "
            f"({leak/len(races)*100:.0f}%) 混ざっています。")
        say(f"  モデルは答えを知っているので回収率は必ず高く出ます。")
        say(f"  本当の成績を見るには --from {args.train_end} を付けてください。")
    else:
        say("学習データ期間との重なりなし = 正真正銘の期間外検証")

    # ---------------- 0) 照合 ----------------
    say("\n[0] predict.py 本体との照合")
    step = max(1, len(races) // args.verify)
    worst = 0.0
    nchk = 0
    t0 = time.time()
    for r in races[::step][:args.verify]:
        rows = race_rows(r)
        feats = P.make_race_features(rows)
        ref = P.predict_combo_probs(feats, r["jcd"])
        fast = combo_probs_fast(feats, r["jcd"], combo_idx)
        for c, v in ref.items():
            worst = max(worst, abs(v - fast[combo_idx[c]]))
        nchk += 1
    say(f"  {nchk}レースで比較  最大差 {worst:.3e}  ({time.time()-t0:.0f}秒)")
    if worst > 1e-9:
        sys.exit("★ 本体と一致しません。以降の数字は信用できないので中止します。")
    say("  一致 OK")

    # ---------------- 確率を全レース分計算 ----------------
    n = len(races)
    say(f"\n確率を計算中… {n:,}レース")
    Pm = np.zeros((n, 120), dtype=np.float32)
    Od = np.zeros((n, 120), dtype=np.float32)
    Hit = np.zeros(n, dtype=np.int16)
    Pay = np.zeros(n, dtype=np.float32)
    Jcd = np.zeros(n, dtype=np.int16)
    Rno = np.zeros(n, dtype=np.int16)
    Wind = np.full(n, np.nan, dtype=np.float32)
    Day = np.zeros(n, dtype=np.int64)

    t0 = time.time()
    for i, r in enumerate(races):
        feats = P.make_race_features(race_rows(r))
        Pm[i] = combo_probs_fast(feats, r["jcd"], combo_idx)
        Od[i] = np.array(r["odds"], dtype=np.float32)
        Hit[i] = combo_idx[r["hit"]]
        Pay[i] = r["pay_3t"]
        Jcd[i] = r["jcd"]
        Rno[i] = r["rno"]
        w = (r.get("weather") or {}).get("wind")
        if w is not None:
            Wind[i] = w
        Day[i] = int(r["date"])
        if (i + 1) % 2000 == 0:
            el = time.time() - t0
            say(f"  {i+1:,}/{n:,}  {el:.0f}秒  残り約{el/(i+1)*(n-i-1):.0f}秒")
    say(f"  完了 {time.time()-t0:.0f}秒")

    # 返還レースを除外(オッズ×100と払戻が合わないもの)
    hit_odds = Od[np.arange(n), Hit]
    keep = np.abs(hit_odds * 100 - Pay) <= 10
    say(f"\n返還等で除外 {int((~keep).sum()):,}件 → 残り {int(keep.sum()):,}件")
    Pm, Od, Hit, Pay = Pm[keep], Od[keep], Hit[keep], Pay[keep]
    Jcd, Rno, Wind, Day = Jcd[keep], Rno[keep], Wind[keep], Day[keep]
    n = len(Pm)
    HitM = np.zeros((n, 120), dtype=bool)
    HitM[np.arange(n), Hit] = True

    # ---------------- 1) キャリブレーション ----------------
    say("\n" + "=" * 58)
    say("[1] キャリブレーション  予測確率 vs 実際の的中率")
    edges = [0, .002, .005, .01, .02, .03, .05, .08, .12, .20, .35, 1.01]
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (Pm >= a) & (Pm < b)
        cnt = int(m.sum())
        if cnt < 200:
            continue
        pred = float(Pm[m].mean())
        act = float(HitM[m].mean())
        lab = f"{a*100:g}〜{b*100:g}%" if b <= 1 else f"{a*100:g}%以上"
        rows.append([lab, f"{cnt:,}", pc(pred), pc(act),
                     f"{act/pred:.2f}" if pred else "-"])
    tbl(["予測帯", "点数", "平均予測", "実測", "実測/予測"], rows)
    say("  実測/予測が1.00に近いほど確率が正しい。1未満=自信過剰")

    # ---------------- 2) 市場の歪み ----------------
    say("\n" + "=" * 58)
    say("[2] オッズ帯別の実測回収率(市場の歪み)")
    oedges = [1, 5, 10, 20, 40, 80, 160, 400, 1e9]
    rows = []
    for a, b in zip(oedges[:-1], oedges[1:]):
        m = (Od >= a) & (Od < b)
        cnt = int(m.sum())
        if cnt < 200:
            continue
        act = float(HitM[m].mean())
        imp = float((0.75 / Od[m]).mean())
        roi = float((HitM[m] * Od[m]).mean())
        rows.append([f"{a:.0f}〜{b:.0f}" if b < 1e8 else f"{a:.0f}以上",
                     f"{cnt:,}", pc(imp), pc(act), pc(roi)])
    tbl(["オッズ", "点数", "市場想定", "実測的中", "回収率"], rows)
    say("  回収率が75%より高い帯=市場が過小評価、低い帯=買われすぎ")

    # ---------------- 3) 確率順位別 ----------------
    say("\n" + "=" * 58)
    say("[3] モデルの確率順位別(その順位を1点だけ買った場合)")
    order = np.argsort(-Pm, axis=1)
    K = 20
    TOPI = order[:, :K]
    Ptop = np.take_along_axis(Pm, TOPI, 1)
    Otop = np.take_along_axis(Od, TOPI, 1)
    HitTop = (TOPI == Hit[:, None])
    rows = []
    for k in range(15):
        act = float(HitTop[:, k].mean())
        rows.append([k + 1, pc(float(Ptop[:, k].mean())), pc(act),
                     f"{float(Otop[:,k].mean()):.0f}",
                     pc(float((HitTop[:, k] * Pay).mean() / 100))])
    tbl(["順位", "平均予測", "実測的中", "平均オッズ", "回収率"], rows)

    # ---------------- 4) EV帯別 ----------------
    say("\n" + "=" * 58)
    say("[4] EV帯別(確率上位12点の中だけ)")
    EVtop = Ptop * Otop
    PayTop = HitTop * Pay[:, None]
    eedges = [0, .7, .85, 1.0, 1.05, 1.1, 1.2, 1.35, 1.6, 2.0, 1e9]
    for label, lim in [("確率上位12点", 12), ("全120点", None)]:
        say(f"\n  ◇ {label}")
        if lim:
            EV, HITX, PAYX = EVtop[:, :lim], HitTop[:, :lim], PayTop[:, :lim]
        else:
            EV, HITX, PAYX = Pm * Od, HitM, HitM * Pay[:, None]
        rows = []
        for a, b in zip(eedges[:-1], eedges[1:]):
            m = (EV >= a) & (EV < b)
            cnt = int(m.sum())
            if cnt < 200:
                continue
            rows.append([f"{a:.2f}〜{b:.2f}" if b < 1e8 else f"{a:.2f}以上",
                         f"{cnt:,}", pc(float(HITX[m].mean())),
                         pc(float(PAYX[m].sum() / (cnt * 100)))])
        tbl(["EV帯", "点数", "的中率", "回収率"], rows)

    # ---------------- 5)6) ルール適用とグリッド ----------------
    colidx = np.arange(K)[None, :]

    def apply_rule(ev_min, top_n, maxp, sel_mask=None):
        m = (colidx < top_n) & (EVtop >= ev_min)
        if sel_mask is not None:
            m = m & sel_mask[:, None]
        evm = np.where(m, EVtop, -1.0)
        od = np.argsort(-evm, axis=1)[:, :maxp]
        ok = np.take_along_axis(m, od, 1)
        pay = np.take_along_axis(PayTop, od, 1) * ok
        cnt = ok.sum(1)
        ret = pay.sum(1)
        hit = (pay > 0).any(1)
        played = cnt >= 1
        pts = int(cnt.sum())
        if pts == 0:
            return None
        cost = cnt * 100.0
        roi = float(ret.sum() / cost.sum())
        se = float(np.sqrt(((ret - roi * cost) ** 2).sum()) / cost.sum())
        return {"races": int(played.sum()), "points": pts,
                "avg_pts": cnt[played].mean(),
                "hit": float(hit[played].mean()),
                "roi": roi, "se": se,
                "pl": float(ret.sum() - pts * 100),
                "played": played, "ret": ret, "cnt": cnt}

    say("\n" + "=" * 58)
    say(f"[5] 現行ルール  EV_MIN={P.EV_MIN} TOP_N={P.TOP_N_PROB} MAX={P.MAX_POINTS}")
    cur = apply_rule(P.EV_MIN, P.TOP_N_PROB, P.MAX_POINTS)
    if cur:
        say(f"  買い目が出たレース {cur['races']:,} / {n:,} ({pc(cur['races']/n)})")
        say(f"  平均 {cur['avg_pts']:.1f}点   延べ {cur['points']:,}点")
        say(f"  レース的中率 {pc(cur['hit'])}   回収率 {pc(cur['roi'])} ± {pc(cur['se'])}")
        say(f"  収支 {cur['pl']:+,.0f}円 (100円/点)")

    say("\n" + "=" * 58)
    say("[6] グリッド探索(回収率順・上位20)")
    min_races = min(500, max(50, n // 20))
    grid = []
    for top_n in [6, 8, 10, 12, 16, 20]:
        for ev_min in [1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50]:
            for maxp in [2, 3, 4, 5, 6, 8, 12]:
                if maxp > top_n:
                    continue
                r = apply_rule(ev_min, top_n, maxp)
                if r and r["races"] >= min_races:
                    grid.append((ev_min, top_n, maxp, r))
    grid.sort(key=lambda g: -g[3]["roi"])
    tbl(["EV", "TOP", "MAX", "レース", "平均点", "的中率", "回収率", "誤差±", "収支"],
        [[f"{e:.2f}", t, m, f"{r['races']:,}", f"{r['avg_pts']:.1f}",
          pc(r["hit"]), pc(r["roi"]), pc(r["se"]), f"{r['pl']:+,.0f}"]
         for e, t, m, r in grid[:20]])
    say("  誤差±は1シグマ。回収率が100%を誤差2つ分うわまわっていなければ偶然の範囲")

    # ---------------- 7) 期間分割 ----------------
    say("\n" + "=" * 58)
    say("[7] 期間分割検証  前半で選んだ設定が後半で通用するか")
    if args.split:
        mid = int(args.split)
    else:
        uniq = np.sort(np.unique(Day))
        cum = np.cumsum([(Day == u).sum() for u in uniq])
        mid = uniq[int(np.searchsorted(cum, cum[-1] / 2))]
    first = Day <= mid
    second = ~first
    say(f"  前半 {int(first.sum()):,}レース (〜{int(mid)})   後半 {int(second.sum()):,}レース")

    def roi_on(mask, ev_min, top_n, maxp):
        r = apply_rule(ev_min, top_n, maxp, sel_mask=mask)
        return r

    cand = []
    for top_n in [6, 8, 10, 12, 16, 20]:
        for ev_min in [1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50]:
            for maxp in [2, 3, 4, 5, 6, 8, 12]:
                if maxp > top_n:
                    continue
                a = roi_on(first, ev_min, top_n, maxp)
                if a and a["races"] >= max(50, min_races // 2):
                    cand.append((a["roi"], ev_min, top_n, maxp))
    cand.sort(key=lambda x: -x[0])
    rows = []
    for roi_a, e, t, m in cand[:10]:
        b = roi_on(second, e, t, m)
        rows.append([f"{e:.2f}", t, m, pc(roi_a),
                     (pc(b["roi"]) + " ± " + pc(b["se"])) if b else "-",
                     f"{b['races']:,}" if b else "-"])
    tbl(["EV", "TOP", "MAX", "前半回収率", "後半回収率", "後半レース"], rows)
    b_cur = roi_on(second, P.EV_MIN, P.TOP_N_PROB, P.MAX_POINTS)
    a_cur = roi_on(first, P.EV_MIN, P.TOP_N_PROB, P.MAX_POINTS)
    if a_cur and b_cur:
        say(f"  現行設定: 前半 {pc(a_cur['roi'])} → 後半 {pc(b_cur['roi'])}")
    say("  前半だけ良くて後半で落ちる設定=偶然。両方100%超が残らなければ勝ち筋なし")

    # ---------------- 8) 軸別 ----------------
    say("\n" + "=" * 58)
    say("[8] 現行ルールでの軸別成績(勝負レースの絞り込み候補)")
    if cur:
        played, ret, cnt = cur["played"], cur["ret"], cur["cnt"]

        def brk(name, keys, labels=None):
            rows = []
            for k in sorted(set(keys.tolist())):
                m = played & (keys == k)
                cost = cnt[m] * 100.0
                if cost.sum() < 30000:
                    continue
                roi = float(ret[m].sum() / cost.sum())
                se = float(np.sqrt(((ret[m] - roi * cost) ** 2).sum()) / cost.sum())
                rows.append([labels.get(k, k) if labels else k, f"{int(m.sum()):,}",
                             pc(roi), pc(se), roi])
            rows.sort(key=lambda r: -r[4])
            say(f"\n  ◇ {name}")
            tbl(["区分", "レース", "回収率", "誤差±"], [r[:4] for r in rows])

        brk("場別", Jcd, P.JCD_NAME)
        brk("R番号別", Rno)

        wb = np.digitize(np.nan_to_num(Wind, nan=-1), [0, 2, 4, 6, 8])
        wl = {0: "不明", 1: "0-1m", 2: "2-3m", 3: "4-5m", 4: "6-7m", 5: "8m以上"}
        brk("風速別", wb, wl)

        p1lane1 = Pm[:, [combo_idx[c] for c in combos if c.startswith("1-")]].sum(1)
        hb = np.digitize(p1lane1, [.3, .4, .5, .6, .7])
        hl = {0: "1号艇30%未満", 1: "30-40%", 2: "40-50%", 3: "50-60%",
              4: "60-70%", 5: "70%以上"}
        brk("1号艇の予想1着率", hb, hl)

    say("\n" + "=" * 58)
    say("注意: ここの回収率は締切時オッズで計算しています。")
    say("実運用は10分前のオッズでEV判定するので、必ずこれより落ちます。")

    with open(args.md, "w", encoding="utf-8") as f:
        f.write("# 買い目条件の検証\n\n```\n" + "\n".join(OUT_LINES) + "\n```\n")
    print(f"\n→ {args.md} に保存しました")


if __name__ == "__main__":
    main()
