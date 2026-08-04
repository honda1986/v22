#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calib.py -- モデル確率と市場確率の最適な配合を求める

背景
  raceselect.py で edge1(1号艇のモデル確率 − 市場確率)が単調に効いた。
    モデルが市場より1号艇を低く見ている  103.2%
    モデルが市場より1号艇を高く見ている   85.5%
  = モデルは1号艇を買いかぶっている。系統的な誤り。

  一部のレースを選んで避けるのではなく、確率そのものを直す。
  全レースを使えるので、バケット漁りより統計が桁違いに強い。

方法
  p ∝ (モデル確率)^a × (市場確率)^b   の2パラメータ
  前半で a,b を最尤推定し、後半で評価する。前半のデータは評価に使わない。

  a≈1,b≈0 → 市場は情報を足さない
  a≈0,b≈1 → モデルは市場を超えていない
  中間     → 混ぜると改善。どれだけかを実測する

使い方
  python calib.py --from 20260501
"""

import argparse
import sys
import time

import numpy as np

import analyze as A
import predict as P

OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def pc(x):
    return f"{x*100:.1f}%"


def zs(s):
    return f"{(s['roi']-1)/s['se']:+.1f}" if s["se"] > 1e-6 else "-"


def tbl(header, rows):
    if not rows:
        say("  (該当なし)")
        return
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    say("  " + "  ".join(str(h).rjust(w[i]) for i, h in enumerate(header)))
    for r in rows:
        say("  " + "  ".join(str(v).rjust(w[i]) for i, v in enumerate(r)))


def blend(lp, lq, a, b):
    """log確率から配合後の確率を作る。行ごとに正規化。"""
    s = a * lp + b * lq
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def nll(lp, lq, hit, a, b):
    """的中目の負の対数尤度(小さいほど良い)"""
    s = a * lp + b * lq
    s -= s.max(1, keepdims=True)
    lse = np.log(np.exp(s).sum(1))
    return float(-(s[np.arange(len(hit)), hit] - lse).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="raw")
    ap.add_argument("--from", dest="dfrom", default="20260501")
    ap.add_argument("--to", dest="dto")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ex-from", dest="exf", default="20250501")
    ap.add_argument("--ex-to", dest="ext", default="20260430")
    ap.add_argument("--split", help="推定に使う期間の終わり YYYYMMDD")
    ap.add_argument("--reverse", action="store_true",
                    help="推定と検証を入れ替える(後ろのブロックで推定し、前のブロックで検証)")
    ap.add_argument("--ev-min", type=float, default=P.EV_MIN)
    ap.add_argument("--top-n", type=int, default=P.TOP_N_PROB)
    ap.add_argument("--max-points", type=int, default=P.MAX_POINTS)
    ap.add_argument("--md", default="calib.md")
    args = ap.parse_args()

    races, combos = A.load(args.out, args.limit, args.dfrom, args.dto)
    if args.exf and args.ext:
        races = [r for r in races if not (args.exf <= r["date"] <= args.ext)]
    if not races:
        sys.exit("該当レースなし")
    cidx = {c: i for i, c in enumerate(combos)}
    lane1 = np.array([i for i, c in enumerate(combos) if c.startswith("1-")])

    say(f"対象 {len(races):,}レース  ({races[0]['date']} 〜 {races[-1]['date']})")
    say(f"学習期間 {args.exf}〜{args.ext} は除外済み")

    n = len(races)
    Pm = np.zeros((n, 120), dtype=np.float64)
    Od = np.zeros((n, 120), dtype=np.float64)
    Hit = np.zeros(n, dtype=np.int64)
    Pay = np.zeros(n, dtype=np.float64)
    Day = np.zeros(n, dtype=np.int64)

    say(f"\n確率を計算中… {n:,}レース")
    t0 = time.time()
    for i, r in enumerate(races):
        Pm[i] = A.combo_probs_fast(P.make_race_features(A.race_rows(r)), r["jcd"], cidx)
        Od[i] = r["odds"]
        Hit[i] = cidx[r["hit"]]
        Pay[i] = r["pay_3t"]
        Day[i] = int(r["date"])
        if (i + 1) % 4000 == 0:
            say(f"  {i+1:,}/{n:,}  {time.time()-t0:.0f}秒")
    say(f"  完了 {time.time()-t0:.0f}秒")

    keep = np.abs(Od[np.arange(n), Hit] * 100 - Pay) <= 10
    Pm, Od, Hit, Pay, Day = [x[keep] for x in (Pm, Od, Hit, Pay, Day)]
    n = len(Pm)
    say(f"返還等を除いて {n:,}レース")

    inv = 1.0 / Od
    Q = inv / inv.sum(1, keepdims=True)
    lp = np.log(np.maximum(Pm, 1e-12))
    lq = np.log(np.maximum(Q, 1e-12))

    if args.split:
        mid = int(args.split)
    else:
        uniq = np.sort(np.unique(Day))
        cum = np.cumsum([(Day == u).sum() for u in uniq])
        mid = uniq[int(np.searchsorted(cum, cum[-1] / 2))]
    tr, te = Day <= mid, Day > mid
    if args.reverse:
        tr, te = te, tr
        say(f"【逆回し】{int(tr.sum()):,}レース({int(mid)}より後)で推定 → "
            f"{int(te.sum()):,}レース({int(mid)}以前)で検証")
    else:
        say(f"前半 {int(tr.sum()):,}レース(〜{int(mid)})で推定 → "
            f"後半 {int(te.sum()):,}レースで評価")
    L_TR, L_TE = ("推定側", "検証側")

    # ---------------- 1) どちらが当てているか ----------------
    say("\n" + "=" * 54)
    say("[1] 的中目の対数損失(小さいほど当てている)")
    rows = []
    for name, (a, b) in [("モデルだけ", (1, 0)), ("市場だけ", (0, 1))]:
        rows.append([name, f"{nll(lp[tr], lq[tr], Hit[tr], a, b):.4f}",
                     f"{nll(lp[te], lq[te], Hit[te], a, b):.4f}"])
    tbl(["確率の作り方", L_TR, L_TE], rows)

    # ---------------- 2) 最適配合 ----------------
    say("\n" + "=" * 54)
    say("[2] 最適な配合を前半で推定")
    best, ga, gb = None, None, None
    for step, lo, hi in [(0.10, 0.0, 1.6), (0.02, None, None)]:
        if lo is None:
            lo_a, hi_a = max(0, ga - 0.12), ga + 0.12
            lo_b, hi_b = max(0, gb - 0.12), gb + 0.12
        else:
            lo_a = lo_b = lo
            hi_a = hi_b = hi
        for a in np.arange(lo_a, hi_a + 1e-9, step):
            for b in np.arange(lo_b, hi_b + 1e-9, step):
                v = nll(lp[tr], lq[tr], Hit[tr], a, b)
                if best is None or v < best:
                    best, ga, gb = v, float(a), float(b)
    say(f"  a(モデル) = {ga:.2f}   b(市場) = {gb:.2f}")
    say(f"  推定側の対数損失 {best:.4f}  →  検証側 {nll(lp[te], lq[te], Hit[te], ga, gb):.4f}")
    if gb > ga * 1.5:
        say("  → 市場の重みが勝っている。モデルは市場を超えていない。")
    elif ga > gb * 1.5:
        say("  → モデルの重みが勝っている。市場は情報を足していない。")
    else:
        say("  → 両方が効いている。混ぜる価値がある。")

    # ---------------- 3) 1号艇の確率 ----------------
    say("\n" + "=" * 54)
    say("[3] 1号艇1着の確率  モデル / 市場 / 実測")
    p1 = Pm[:, lane1].sum(1)
    q1 = Q[:, lane1].sum(1)
    won1 = np.isin(Hit, lane1)
    edges = np.quantile(p1, np.linspace(0, 1, 11))
    edges = np.unique(edges)
    bins = np.clip(np.digitize(p1, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for k in sorted(set(bins.tolist())):
        m = bins == k
        if m.sum() < 50:
            continue
        rows.append([f"{edges[k]:.2f}〜{edges[k+1]:.2f}", f"{int(m.sum()):,}",
                     pc(p1[m].mean()), pc(q1[m].mean()), pc(won1[m].mean()),
                     f"{p1[m].mean()-won1[m].mean():+.3f}"])
    tbl(["モデルの帯", "レース", "モデル", "市場", "実測", "モデル-実測"], rows)
    say(f"  全体: モデル {pc(p1.mean())}  市場 {pc(q1.mean())}  実測 {pc(won1.mean())}")
    say("  モデル-実測 がプラス = 買いかぶり")

    # ---------------- 4) 配合確率で買ってみる ----------------
    say("\n" + "=" * 54)
    say("[4] 閾値なしの比較  各レースでEV上位M点を買う")
    say("  市場確率×オッズは定義上0.75なので、EV閾値を固定すると比較にならない。")
    say("  ここは点数を揃えて、選び方の良し悪しだけを見る。")

    def buy(Pu, mask, m_pts, ev_min=None):
        order = np.argsort(-Pu, axis=1)[:, :args.top_n]
        pt = np.take_along_axis(Pu, order, 1)
        ot = np.take_along_axis(Od, order, 1)
        ev = pt * ot
        ht = (order == Hit[:, None])
        ok0 = np.ones_like(ev, dtype=bool) if ev_min is None else (ev >= ev_min)
        evm = np.where(ok0, ev, -1e9)
        sel = np.argsort(-evm, axis=1)[:, :m_pts]
        ok = np.take_along_axis(ok0, sel, 1)
        pay = np.take_along_axis(ht * Pay[:, None], sel, 1) * ok
        cnt = ok.sum(1).astype(float)[mask]
        ret = pay.sum(1)[mask]
        cost = cnt * 100.0
        if cost.sum() < 10000:
            return None
        r = float(ret.sum() / cost.sum())
        se = float(np.sqrt(((ret - r * cost) ** 2).sum()) / cost.sum())
        return {"races": int((cnt >= 1).sum()), "pts": int(cnt.sum()),
                "roi": r, "se": max(se, 1e-9), "pl": float(ret.sum() - cost.sum())}

    variants = [("モデルのみ", 1.0, 0.0), (f"配合 a={ga:.2f} b={gb:.2f}", ga, gb)]
    rows = []
    for name, a, b in variants:
        Pu = blend(lp, lq, a, b)
        for m_pts in (1, 2, 3, 5):
            r_tr = buy(Pu, tr, m_pts)
            r_te = buy(Pu, te, m_pts)
            if not r_te:
                continue
            rows.append([name, m_pts, f"{r_te['pts']:,}",
                         pc(r_tr["roi"]) if r_tr else "-",
                         pc(r_te["roi"]), pc(r_te["se"]), zs(r_te)])
    tbl(["確率", "点/R", "検証点数", L_TR, L_TE, "誤差±", "z値"], rows)

    # 参考: 市場の1番人気を1点買い
    fav = np.argmax(Q, axis=1)
    for label, mask in ((L_TR, tr), (L_TE, te)):
        w = (fav == Hit)[mask]
        r = float((w * Pay[mask]).sum() / (mask.sum() * 100))
        say(f"  参考 1番人気1点買い {label}: {pc(r)}")

    say("\n" + "=" * 54)
    say(f"[5] 現行ルール EV≧{args.ev_min} 上位{args.top_n}点 最大{args.max_points}点")
    rows = []
    for name, a, b in variants:
        Pu = blend(lp, lq, a, b)
        for label, mask in ((L_TR, tr), (L_TE, te)):
            s2 = buy(Pu, mask, args.max_points, ev_min=args.ev_min)
            if s2 is None:
                rows.append([name, label, "0", "0", "買い目なし", "-", "-", "-"])
                continue
            rows.append([name, label, f"{s2['races']:,}", f"{s2['pts']:,}",
                         pc(s2["roi"]), pc(s2["se"]), zs(s2), f"{s2['pl']:+,.0f}"])
    tbl(["確率", "期間", "レース", "点数", "回収率", "誤差±", "z値", "収支"], rows)

    say("\n  参考: EV閾値を変えたとき(配合確率)  ※本命は 1.10。他は後知恵になるので選ばない")
    Pu = blend(lp, lq, ga, gb)
    rows = []
    for ev in (1.05, 1.10, 1.15, 1.20, 1.30):
        a1 = buy(Pu, tr, args.max_points, ev_min=ev)
        b1 = buy(Pu, te, args.max_points, ev_min=ev)
        rows.append([f"{ev:.2f}",
                     f"{a1['pts']:,}" if a1 else "0", pc(a1["roi"]) if a1 else "-",
                     f"{b1['pts']:,}" if b1 else "0", pc(b1["roi"]) if b1 else "-",
                     pc(b1["se"]) if b1 else "-", zs(b1) if b1 else "-"])
    tbl(["EV", f"{L_TR}点数", L_TR, f"{L_TE}点数", L_TE, "誤差±", "z値"], rows)

    # ---------------- 6) 頑健性 ----------------
    say("\n" + "=" * 54)
    say("[6] 頑健性チェック  配合 × EV≧%.2f × 検証側" % args.ev_min)

    def detail(Pu, mask, m_pts, ev_min):
        order = np.argsort(-Pu, axis=1)[:, :args.top_n]
        pt = np.take_along_axis(Pu, order, 1)
        ot = np.take_along_axis(Od, order, 1)
        ev = pt * ot
        ht = (order == Hit[:, None])
        ok0 = ev >= ev_min
        evm = np.where(ok0, ev, -1e9)
        sel = np.argsort(-evm, axis=1)[:, :m_pts]
        ok = np.take_along_axis(ok0, sel, 1)
        pay = np.take_along_axis(ht * Pay[:, None], sel, 1) * ok
        return pay.sum(1)[mask], ok.sum(1).astype(float)[mask], Day[mask]

    Pu = blend(lp, lq, ga, gb)
    ret, cnt, dy = detail(Pu, te, args.max_points, args.ev_min)
    inplay = cnt >= 1
    ret, cnt, dy = ret[inplay], cnt[inplay], dy[inplay]
    cost = cnt.sum() * 100.0
    if cost < 10000:
        say("  買い目が少なすぎて評価できません")
    else:
        hits = ret[ret > 0]
        say(f"  レース {len(ret):,}  点数 {int(cnt.sum()):,}  的中 {len(hits):,}本"
            f"  (レース的中率 {pc(len(hits)/len(ret))})")
        say(f"  回収率 {pc(ret.sum()/cost)}  収支 {ret.sum()-cost:+,.0f}円")
        top = np.sort(hits)[::-1]
        say("  高額的中 上位8本: " + " ".join(f"{int(v):,}" for v in top[:8]))
        say("")
        rows = []
        for k in (0, 1, 3, 5, 10):
            r2 = (ret.sum() - top[:k].sum()) / cost
            rows.append([f"上位{k}本を除く" if k else "そのまま",
                         pc(r2), f"{ret.sum()-top[:k].sum()-cost:+,.0f}"])
        tbl(["条件", "回収率", "収支"], rows)
        say("  上位1本を抜いて100%を割るなら、まぐれ当たり1本で持っているだけ")

        say("")
        yr = (dy // 10000) - ((dy // 100) % 100 < 5).astype(int)
        rows = []
        for y in sorted(set(yr.tolist())):
            m2 = yr == y
            c2 = cnt[m2].sum() * 100.0
            if c2 < 5000:
                continue
            r2 = ret[m2].sum() / c2
            se2 = np.sqrt(((ret[m2] - r2 * cnt[m2] * 100) ** 2).sum()) / c2
            rows.append([f"{y}年度(5月〜)", f"{int(m2.sum()):,}",
                         f"{int(cnt[m2].sum()):,}", pc(r2), pc(se2),
                         f"{ret[m2].sum()-c2:+,.0f}"])
        tbl(["期間", "レース", "点数", "回収率", "誤差±", "収支"], rows)
        say("  どの年度も100%を超えていれば本物に近い。1年度だけなら偶然を疑う")

    say("\n" + "=" * 54)
    say("見るべきは『配合 × 検証側』の行だけ。a,bは推定側で決めたので、検証側は素の期間外。")
    say("そこが100%を誤差2つ分うわまわらなければ、この方向でも勝てない。")

    with open(args.md, "w", encoding="utf-8") as f:
        f.write("# 確率の配合\n\n```\n" + "\n".join(OUT) + "\n```\n")
    print(f"\n→ {args.md} に保存しました")


if __name__ == "__main__":
    main()
