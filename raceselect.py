#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select.py -- 「どのレースを買うか」の条件を探す

analyze.py は買い目(組)の絞り込みだった。こちらはレース単位の絞り込み。
現行ルールは88%のレースで買っている＝ほぼ全部買い。そこを削る。

考え方
  的中しやすいレース ≠ 儲かるレース。
  堅いレースは当たるが配当が安く、差し引きマイナスになる。
  探すべきは「モデルの確率が市場のオッズとズレていて、かつモデルが正しいレース」。

指標
  p_max      モデル1位の確率(本命度)
  p_top12    上位12点の確率合計(絞れ具合)
  max_ev     上位12点の中の最大EV
  n_ev15     EV1.5以上の点数
  edge1      1号艇 モデル確率 − 市場確率(モデルが市場とどれだけ違うか)
  kl         モデルと市場の食い違いの大きさ(全120点)
  fav_odds   1番人気のオッズ(市場から見た堅さ)
  tenji1     1号艇の展示タイム順位
  maezuke    前付けの有無
  cls1       1号艇の級別

使い方
  python select.py --from 20260501
  python select.py --from 20260501 --ev-min 1.5 --top-n 12 --max-points 12
"""

import argparse
import sys

import numpy as np

import analyze as A
import predict as P

OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def pc(x):
    return f"{x*100:.1f}%"


def tbl(header, rows):
    if not rows:
        say("  (該当なし)")
        return
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    say("  " + "  ".join(str(h).rjust(w[i]) for i, h in enumerate(header)))
    for r in rows:
        say("  " + "  ".join(str(v).rjust(w[i]) for i, v in enumerate(r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="raw")
    ap.add_argument("--from", dest="dfrom", default="20260501")
    ap.add_argument("--to", dest="dto")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ex-from", dest="exf", default="20250501")
    ap.add_argument("--ex-to", dest="ext", default="20260430")
    ap.add_argument("--ev-min", type=float, default=P.EV_MIN)
    ap.add_argument("--top-n", type=int, default=P.TOP_N_PROB)
    ap.add_argument("--max-points", type=int, default=P.MAX_POINTS)
    ap.add_argument("--md", default="raceselect.md")
    args = ap.parse_args()

    races, combos = A.load(args.out, args.limit, args.dfrom, args.dto)
    if args.exf and args.ext:
        races = [r for r in races if not (args.exf <= r["date"] <= args.ext)]
    if not races:
        sys.exit("該当レースなし")
    cidx = {c: i for i, c in enumerate(combos)}
    lane1 = np.array([i for i, c in enumerate(combos) if c.startswith("1-")])

    say(f"対象 {len(races):,}レース  ({races[0]['date']} 〜 {races[-1]['date']})")
    say(f"買い方: EV≧{args.ev_min}  確率上位{args.top_n}点  最大{args.max_points}点")

    n = len(races)
    Pm = np.zeros((n, 120), dtype=np.float32)
    Od = np.zeros((n, 120), dtype=np.float32)
    Hit = np.zeros(n, dtype=np.int32)
    Pay = np.zeros(n, dtype=np.float32)
    Day = np.zeros(n, dtype=np.int64)
    Jcd = np.zeros(n, dtype=np.int16)
    Rno = np.zeros(n, dtype=np.int16)
    Mz = np.zeros(n, dtype=np.int8)
    Cls1 = np.zeros(n, dtype=np.int8)
    T1 = np.zeros(n, dtype=np.int8)

    say(f"\n確率を計算中… {n:,}レース")
    import time
    t0 = time.time()
    for i, r in enumerate(races):
        feats = P.make_race_features(A.race_rows(r))
        Pm[i] = A.combo_probs_fast(feats, r["jcd"], cidx)
        Od[i] = np.array(r["odds"], dtype=np.float32)
        Hit[i] = cidx[r["hit"]]
        Pay[i] = r["pay_3t"]
        Day[i] = int(r["date"])
        Jcd[i], Rno[i] = r["jcd"], r["rno"]
        es = r["entries"]
        Mz[i] = 1 if any(e["course_in"] != e["lane"] for e in es) else 0
        Cls1[i] = es[0]["cls_val"]
        tj = [e["tenji"] for e in es]
        T1[i] = 1 + sum(1 for t in tj if t > 0 and t < tj[0]) if tj[0] > 0 else 6
        if (i + 1) % 4000 == 0:
            say(f"  {i+1:,}/{n:,}  {time.time()-t0:.0f}秒")
    say(f"  完了 {time.time()-t0:.0f}秒")

    keep = np.abs(Od[np.arange(n), Hit] * 100 - Pay) <= 10
    say(f"返還等で除外 {int((~keep).sum()):,}件")
    Pm, Od, Hit, Pay, Day, Jcd, Rno, Mz, Cls1, T1 = [
        x[keep] for x in (Pm, Od, Hit, Pay, Day, Jcd, Rno, Mz, Cls1, T1)]
    n = len(Pm)

    # ---------------- 買い目を決める ----------------
    order = np.argsort(-Pm, axis=1)
    K = max(20, args.top_n)
    TOPI = order[:, :K]
    Ptop = np.take_along_axis(Pm, TOPI, 1)
    Otop = np.take_along_axis(Od, TOPI, 1)
    EVtop = Ptop * Otop
    HitTop = (TOPI == Hit[:, None])
    PayTop = HitTop * Pay[:, None]

    col = np.arange(K)[None, :]
    m = (col < args.top_n) & (EVtop >= args.ev_min)
    evm = np.where(m, EVtop, -1.0)
    sel = np.argsort(-evm, axis=1)[:, :args.max_points]
    ok = np.take_along_axis(m, sel, 1)
    cnt = ok.sum(1).astype(float)
    ret = (np.take_along_axis(PayTop, sel, 1) * ok).sum(1)
    played = cnt >= 1

    def stat(mask):
        mask = mask & played
        cost = cnt[mask] * 100.0
        if cost.sum() < 20000:
            return None
        r = float(ret[mask].sum() / cost.sum())
        se = float(np.sqrt(((ret[mask] - r * cost) ** 2).sum()) / cost.sum())
        hit = float((ret[mask] > 0).mean())
        return {"n": int(mask.sum()), "roi": r, "se": se, "hit": hit,
                "pl": float(ret[mask].sum() - cost.sum()), "pts": cost.sum() / 100}

    base = stat(np.ones(n, bool))
    say(f"\n買い目が出たレース {int(played.sum()):,}/{n:,} ({pc(played.mean())})")
    say(f"全体: 回収率 {pc(base['roi'])} ± {pc(base['se'])}  収支 {base['pl']:+,.0f}円")

    # ---------------- レース単位の指標 ----------------
    q = 1.0 / Od
    q = q / q.sum(1, keepdims=True)
    p_max = Ptop[:, 0]
    p_top12 = Ptop[:, :12].sum(1)
    max_ev = np.where(col[:, :args.top_n] >= 0, EVtop[:, :args.top_n], 0).max(1)
    n_ev15 = ((EVtop[:, :args.top_n] >= 1.5)).sum(1)
    edge1 = Pm[:, lane1].sum(1) - q[:, lane1].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(Pm > 0, Pm * np.log(np.maximum(Pm, 1e-12) / np.maximum(q, 1e-12)), 0).sum(1)
    fav_odds = Od.min(1)

    metrics = [
        ("p_max     モデル1位の確率", p_max, 5),
        ("p_top12   上位12点の確率合計", p_top12, 5),
        ("max_ev    上位内の最大EV", max_ev, 5),
        ("n_ev15    EV1.5以上の点数", n_ev15.astype(float), None),
        ("edge1     1号艇 モデル−市場", edge1, 5),
        ("kl        モデルと市場の食い違い", kl, 5),
        ("fav_odds  1番人気オッズ", fav_odds, 5),
        ("tenji1    1号艇の展示順位", T1.astype(float), None),
        ("maezuke   前付けあり=1", Mz.astype(float), None),
        ("cls1      1号艇の級別", Cls1.astype(float), None),
    ]

    say("\n" + "=" * 56)
    say("[1] レース単位の指標ごとの回収率")
    say("  誤差±を必ず見ること。誤差2つ分100%を超えていなければ意味なし")
    ranked = []
    for name, v, nq in metrics:
        say(f"\n  ◇ {name}")
        rows = []
        if nq:
            edges = np.unique(np.quantile(v[played], np.linspace(0, 1, nq + 1)))
            bins = np.clip(np.digitize(v, edges[1:-1]), 0, len(edges) - 2)
            labels = {k: f"{edges[k]:.3g}〜{edges[k+1]:.3g}" for k in range(len(edges) - 1)}
        else:
            bins = v.astype(int)
            labels = {k: str(k) for k in np.unique(bins)}
        best = None
        for k in sorted(set(bins.tolist())):
            s = stat(bins == k)
            if not s:
                continue
            rows.append([labels.get(k, k), f"{s['n']:,}", pc(s["roi"]), pc(s["se"]),
                         f"{s['pl']:+,.0f}"])
            z = (s["roi"] - 1) / s["se"] if s["se"] else 0
            if best is None or z > best[0]:
                best = (z, k, s)
        tbl(["区分", "レース", "回収率", "誤差±", "収支"], rows)
        if best:
            ranked.append((best[0], name, v, bins, best[1], best[2]))

    # ---------------- 前半/後半で確認 ----------------
    uniq = np.sort(np.unique(Day))
    cum = np.cumsum([(Day == u).sum() for u in uniq])
    mid = uniq[int(np.searchsorted(cum, cum[-1] / 2))]
    first, second = Day <= mid, Day > mid

    say("\n" + "=" * 56)
    say(f"[2] 有望な条件の前半/後半検証  (境目 {int(mid)})")
    ranked.sort(key=lambda x: -x[0])
    rows = []
    for z, name, v, bins, k, s in ranked[:8]:
        a = stat((bins == k) & first)
        b = stat((bins == k) & second)
        rows.append([name.split()[0], f"{s['n']:,}", pc(s["roi"]),
                     f"{z:+.1f}",
                     pc(a["roi"]) if a else "-", pc(b["roi"]) if b else "-"])
    tbl(["指標", "レース", "全体", "z値", "前半", "後半"], rows)
    say("  z値=100%から誤差いくつ分離れているか。2.0未満は偶然の範囲")
    say("  前半と後半のどちらかが100%を割る条件は使えない")

    # ---------------- 上位x%だけ買う ----------------
    say("\n" + "=" * 56)
    say("[3] 各指標の上位x%だけ買った場合")
    rows = []
    for name, v, _ in metrics:
        for frac in (0.1, 0.2, 0.3):
            for sign in (1, -1):
                thr = np.quantile(v[played], 1 - frac if sign > 0 else frac)
                mask = (v >= thr) if sign > 0 else (v <= thr)
                s = stat(mask)
                if not s or s["n"] < 300:
                    continue
                z = (s["roi"] - 1) / s["se"]
                if z < 1.0:
                    continue
                rows.append([name.split()[0], ("上位" if sign > 0 else "下位") + f"{frac:.0%}",
                             f"{s['n']:,}", pc(s["roi"]), pc(s["se"]), f"{z:+.1f}",
                             f"{s['pl']:+,.0f}", z])
    rows.sort(key=lambda r: -r[-1])
    tbl(["指標", "範囲", "レース", "回収率", "誤差±", "z値", "収支"],
        [r[:-1] for r in rows[:15]])
    if not rows:
        say("  100%を1シグマ以上うわまわる条件は1つもありません")

    say("\n" + "=" * 56)
    say("注意: ここは条件を多数試している。最大値は必ず上振れる。")
    say("ここで見つけた条件は仮説にすぎない。学習期間より前(2024-05〜2025-04)の")
    say("データで、条件を一切いじらずに当てはめて初めて答えが出る。")

    with open(args.md, "w", encoding="utf-8") as f:
        f.write("# 勝負レースの絞り込み\n\n```\n" + "\n".join(OUT) + "\n```\n")
    print(f"\n→ {args.md} に保存しました")


if __name__ == "__main__":
    main()
