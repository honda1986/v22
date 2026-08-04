#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v23_2t.py -- 3連単プールと2連単プールの食い違いを測る (Google Colab で実行)

■ これまでと何が違うか
  今まで4回失敗したのは全部「モデルが市場を超える」という賭けだった。
  v23の関門で、必要な優位の1%しか無いことが確定した。

  これは別の仕組み。
  3連単プールと2連単プールは、金の流れが別。買う人の層も違う。
  3連単プールが「1-2着は8%」と言っているのに、
  2連単プールが5%相当の値付けをしていたら、そこに歪みがある。

  モデルは要らない。2つの市場が互いに食い違っているかだけを見る。

■ 測ること
  1. 2連単30点を全部買った回収率(正常なら75%前後)
  2. 2つのプールの一致度
  3. EV = 3連単から作った確率 × 2連単オッズ  の帯別回収率  ← 本命
  4. 期間を半分に割って再現するか

■ 使い方 (Colab)
  !pip -q install lightgbm
  !rm -rf v22 && git clone --depth 1 https://github.com/honda1986/v22.git
  %run v22/v23_2t.py
"""

import glob
import gzip
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from bs4 import BeautifulSoup

RAW_DIR = "v22/raw" if os.path.isdir("v22/raw") else "raw"
OUT = "v23_out"
URL_OFF = "https://www.boatrace.jp/owpc/pc/race/odds2tf"
URL_FUN = "https://info.kyotei.fun/info-{d}-{jcd:02d}-{rno}.html"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

N_RACES = 8000          # 取得するレース数(遅い取得元なら自動で減らす)
WORKERS = 10            # 同時接続数
MINUTES = 60            # 打ち切り時間
SAMPLE_FROM = "20240501"   # この日以降から抽出

# 2連単30点の並び (1着, 2着)
PAIRS = [(a, b) for a in range(1, 7) for b in range(1, 7) if b != a]
PAIR_IX = {p: i for i, p in enumerate(PAIRS)}


# ============================================================ 取得
def parse_odds2t(html):
    """2連単オッズ30点。表は 1着ごとに6ブロック × 5行。
    2着の数字は読まず、並び順から組番を復元する。"""
    soup = BeautifulSoup(html, "html.parser")
    head = None
    for h in soup.find_all(["h2", "h3", "h4"]):
        if "2連単" in h.get_text():
            head = h
            break
    tbl = head.find_next("table") if head else None
    if tbl is None:
        return None

    rows = []
    for tr in tbl.find_all("tr"):
        vals = [td.get_text(strip=True) for td in tr.find_all("td")]
        f = [float(v) for v in vals if re.fullmatch(r"\d+\.\d+", v)]
        if len(f) == 6:
            rows.append(f)
    if len(rows) != 5:
        return None

    out = np.zeros(30, dtype=np.float32)
    for r, vals in enumerate(rows):
        for g, v in enumerate(vals):
            first = g + 1
            others = [b for b in range(1, 7) if b != first]
            out[PAIR_IX[(first, others[r])]] = v
    return out if (out > 0).all() else None


def parse_odds2t_fun(html, dump=False):
    """info.kyotei.fun の『2連単 人気順オッズ』から30点。
    艇番は数字ではなく色付きアイコンのCSSクラス(ng2r1〜ng2r6)で表される。"""
    soup = BeautifulSoup(html, "html.parser")
    head = None
    for h in soup.find_all(["h2", "h3", "h4"]):
        if "2連単" in h.get_text():
            head = h
            break
    if head is None:
        if dump:
            hs = [h.get_text(strip=True)[:30] for h in soup.find_all(["h2", "h3"])]
            print("  このページの見出し:", " / ".join(hs[:15]))
        return None

    out = np.zeros(30, dtype=np.float32)
    for el in head.find_all_next():
        if el.name in ("h2", "h3", "h4") and el is not head:
            break
        if el.name != "tr":
            continue
        nums = []
        for dv in el.find_all("div"):
            m = re.search(r"ng2r(\d)", " ".join(dv.get("class") or []))
            if m:
                nums.append(int(m.group(1)))
        if len(nums) != 2 or nums[0] == nums[1]:
            continue
        tds = el.find_all("td")
        if len(tds) < 2:
            continue
        try:
            v = float(tds[-1].get_text(strip=True).replace(",", ""))
        except ValueError:
            continue
        out[PAIR_IX[(nums[0], nums[1])]] = v
    return out if (out > 0).all() else None


def fetch_fun(sess, d, jcd, rno, dump=False):
    try:
        r = sess.get(URL_FUN.format(d=d, jcd=jcd, rno=rno), timeout=20)
        if r.status_code != 200:
            return None, f"http_{r.status_code}"
        if not r.encoding or r.encoding.lower() == "iso-8859-1":
            r.encoding = r.apparent_encoding or "utf-8"
        o = parse_odds2t_fun(r.text, dump)
        return o, (None if o is not None else "parse")
    except requests.RequestException as e:
        return None, type(e).__name__


def fetch(sess, d, jcd, rno):
    try:
        r = sess.get(URL_OFF, params={"rno": rno, "jcd": f"{jcd:02d}", "hd": d},
                     timeout=25)
        if r.status_code != 200:
            return None, f"http_{r.status_code}"
        r.encoding = "utf-8"
        o = parse_odds2t(r.text)
        return o, (None if o is not None else "parse")
    except requests.RequestException as e:
        return None, type(e).__name__


# ============================================================ raw
def load_raw():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json.gz")))
    if not files:
        sys.exit(f"{RAW_DIR} がありません")
    combos = None
    out = []
    for p in files:
        if os.path.basename(p)[:8] < SAMPLE_FROM:
            continue
        with gzip.open(p, "rt", encoding="utf-8") as f:
            d = json.load(f)
        if combos is None:
            combos = d["combos"]
            M = np.zeros((120, 30), dtype=np.float32)
            for i, c in enumerate(combos):
                a, b, _ = (int(x) for x in c.split("-"))
                M[i, PAIR_IX[(a, b)]] = 1.0
        for r in d["races"]:
            if "error" in r or r.get("n_odds") != 120:
                continue
            if not r.get("hit") or not r.get("pay_3t"):
                continue
            o3 = np.array(r["odds"], dtype=np.float32)
            if abs(o3[combos.index(r["hit"])] * 100 - r["pay_3t"]) > 10:
                continue          # 返還レース
            a, b, _ = (int(x) for x in r["hit"].split("-"))
            out.append((r["date"], r["jcd"], r["rno"], o3, PAIR_IX[(a, b)]))
    return out, M


# ============================================================ 分析
def pc(x):
    return f"{x*100:.1f}%"


def stat(ret, cnt):
    cost = cnt.sum() * 100.0
    if cost < 10000:
        return None
    r = float(ret.sum() / cost)
    se = float(np.sqrt(((ret - r * cnt * 100) ** 2).sum()) / cost)
    return r, se, int(cnt.sum()), float(ret.sum() - cost)


def main():
    races, M = load_raw()
    print(f"raw から {len(races):,}レース ({SAMPLE_FROM}以降)")
    random.seed(42)
    sample = random.sample(races, min(N_RACES, len(races)))
    sample.sort(key=lambda x: (x[0], x[1], x[2]))

    sess = requests.Session()
    sess.headers.update(UA)
    sess.mount("https://", requests.adapters.HTTPAdapter(
        pool_connections=WORKERS * 2, pool_maxsize=WORKERS * 2))

    # --- 取得元を自動で決める ---
    print("\n取得元を判定します")
    ok = 0
    t0 = time.time()
    for j, (d, jcd, rno, _, _) in enumerate(sample[:5]):
        o, err = fetch_fun(sess, d, jcd, rno, dump=(j == 0))
        ok += (o is not None)
    per_fun = (time.time() - t0) / 5
    print(f"  info.kyotei.fun  成功{ok}/5  1件{per_fun:.1f}秒")

    if ok >= 3:
        source, per = "fun", per_fun
        getter = lambda s: fetch_fun(sess, s[0], s[1], s[2])
        print("  → info.kyotei.fun を使います")
    else:
        t0 = time.time()
        for d, jcd, rno, _, _ in sample[:5]:
            fetch(sess, d, jcd, rno)
        per = (time.time() - t0) / 5
        source = "official"
        getter = lambda s: fetch(sess, s[0], s[1], s[2])
        print(f"  boatrace.jp 公式  1件{per:.1f}秒")
        print("  → 公式を使います(遅いので件数を減らします)")

    can = int(MINUTES * 60 / max(per, 0.01) * WORKERS)
    if can < len(sample):
        print(f"  {MINUTES}分で取れるのは約{can:,}件。そこまでに絞ります。")
        sample = sample[:max(can, 1500)]

    # --- 取得 ---
    print(f"\n{len(sample):,}レースを取得(同時{WORKERS}, 上限{MINUTES}分, {source})")
    O2, IDX = [], []
    fail = {}
    t0 = time.time()
    stop = False
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i in range(0, len(sample), 200):
            if (time.time() - t0) / 60 > MINUTES:
                print("  時間切れ。ここまでで集計します。")
                break
            chunk = sample[i:i + 200]
            res = list(ex.map(getter, chunk))
            for (d, jcd, rno, o3, hi), (o2, err) in zip(chunk, res):
                if o2 is None:
                    fail[err] = fail.get(err, 0) + 1
                    continue
                O2.append(o2)
                IDX.append((o3, hi))
            el = time.time() - t0
            print(f"  {i+len(chunk):,}/{len(sample):,}  取得{len(O2):,}  "
                  f"{el/60:.1f}分", flush=True)

    if len(O2) < 500:
        print(f"\n取得 {len(O2)}件。少なすぎて判定できません。 {fail}")
        return
    if fail:
        print("  取れなかったもの:", fail)

    O2 = np.stack(O2)
    O3 = np.stack([x[0] for x in IDX])
    HIT = np.array([x[1] for x in IDX])
    n = len(O2)
    print(f"\n集計対象 {n:,}レース")

    # 3連単プールから作った2連単確率
    inv3 = 1.0 / O3
    Q3 = inv3 / inv3.sum(1, keepdims=True)
    P = Q3 @ M                       # レース×30
    # 2連単プール自身の確率
    inv2 = 1.0 / O2
    Q2 = inv2 / inv2.sum(1, keepdims=True)

    HM = np.zeros((n, 30), dtype=bool)
    HM[np.arange(n), HIT] = True
    PAY = O2[np.arange(n), HIT] * 100.0

    print("\n" + "=" * 54)
    print("[1] 対照実験")
    all_cnt = np.full(n, 30.0)
    s = stat(PAY, all_cnt)
    print(f"  2連単30点を全部買った回収率 {pc(s[0])} ± {pc(s[1])}  (75%前後なら正常)")
    print(f"  2連単の控除率 {1-1/(inv2.sum(1)).mean():.1%}")
    print(f"  3連単の控除率 {1-1/(inv3.sum(1)).mean():.1%}")

    print("\n[2] 2つのプールの食い違い")
    print(f"  確率の相関 {np.corrcoef(P.ravel(), Q2.ravel())[0,1]:.4f}")
    print(f"  平均絶対差 {np.abs(P-Q2).mean()*100:.3f}ポイント")
    print(f"  どちらが当てているか(的中組の対数損失、小さいほど良い)")
    print(f"    3連単プール由来 {-np.log(np.clip(P[HM],1e-9,None)).mean():.4f}")
    print(f"    2連単プール自身 {-np.log(np.clip(Q2[HM],1e-9,None)).mean():.4f}")

    print("\n" + "=" * 54)
    print("[3] EV = 3連単から作った確率 × 2連単オッズ")
    EV = P * O2
    edges = [0, .7, .85, 1.0, 1.05, 1.1, 1.2, 1.4, 1.7, 99]
    print("      EV帯      点数     的中率     回収率     誤差±")
    for a, b in zip(edges[:-1], edges[1:]):
        m = (EV >= a) & (EV < b)
        c = int(m.sum())
        if c < 200:
            continue
        got = (HM & m) * PAY[:, None]
        r = got.sum() / (c * 100)
        se = np.sqrt(((got.sum(1) - r * m.sum(1) * 100) ** 2).sum()) / (c * 100)
        print(f"  {a:5.2f}〜{b:5.2f}  {c:8,}  {pc((HM&m).sum()/c):>8}  "
              f"{pc(r):>8}  {pc(se):>7}")

    print("\n[4] EV上位だけ買う")
    print("     EV下限   レース      点数     回収率     誤差±      z値      収支")
    for ev in (1.0, 1.05, 1.10, 1.20, 1.40):
        m = EV >= ev
        cnt = m.sum(1).astype(float)
        ret = ((HM & m) * PAY[:, None]).sum(1)
        s = stat(ret, cnt)
        if not s:
            print(f"  {ev:7.2f}   買い目なし")
            continue
        r, se, pts, pl = s
        print(f"  {ev:7.2f}  {int((cnt>=1).sum()):7,}  {pts:8,}  "
              f"{pc(r):>8}  {pc(se):>7}  {(r-1)/se:+6.1f}  {pl:+10,.0f}")

    print("\n[5] 期間を半分に割って再現するか(取得順=日付順)")
    print("     EV下限        前半        後半")
    half = n // 2
    for ev in (1.05, 1.10, 1.20):
        out = []
        for sl in (slice(0, half), slice(half, n)):
            m = EV[sl] >= ev
            cnt = m.sum(1).astype(float)
            ret = ((HM[sl] & m) * PAY[sl, None]).sum(1)
            s = stat(ret, cnt)
            out.append(pc(s[0]) if s else "なし")
        print(f"  {ev:7.2f}  {out[0]:>10}  {out[1]:>10}")

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f"{OUT}/odds2t.npz", O2=O2, O3=O3, HIT=HIT)
    print(f"\n{OUT}/odds2t.npz に保存しました({n:,}レース)")
    print("\n注意: 締切時オッズです。実運用は必ずこれより落ちます。")


if __name__ == "__main__":
    main()
