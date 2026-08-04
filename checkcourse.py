#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
checkcourse.py -- raw/ の course_in の正体を突き止める

分かっていること
  実運用(v22_app.py)は boatrace.jp の直前情報のスタート展示から
  「行の並び=コース、アイコンの数字=艇番」で course_in[艇番]=コース を作る。これは正しい。

分かっていないこと
  raw/ は info.kyotei.fun の「コースIN」行を、Colabのコードが
  「列=艇番、中身=コース」と逆向きに解釈して読んでいる。
  さらに、その行が展示進入なのか本番進入なのかも不明。

やること
  4通り全部を実物と突き合わせる。
    そのまま  vs 展示進入
    入れ替え  vs 展示進入
    そのまま  vs 本番進入
    入れ替え  vs 本番進入
  100%一致した組み合わせが正解。

使い方
  python checkcourse.py --n 200          # 約7分
  python checkcourse.py --n 100 --wait 3 # 弾かれる場合
"""

import argparse, glob, gzip, json, os, random, re, sys, time
from collections import Counter
import requests
from bs4 import BeautifulSoup

BASE = "https://www.boatrace.jp/owpc/pc/race"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept-Language": "ja,en;q=0.8"}


def parse_course(html):
    """スタート展示/スタート情報から {艇番: コース} を作る。
    行の並び順がコース番号、艇番アイコンの数字が艇番。"""
    soup = BeautifulSoup(html, "html.parser")

    # 方法1: 艇番アイコンの画像ファイル名(スタート欄だけに出る)
    boats = [int(m.group(1)) for m in
             (re.search(r"img_boat2_(\d)\.png", i.get("src", ""))
              for i in soup.find_all("img")) if m]
    if sorted(boats) == [1, 2, 3, 4, 5, 6]:
        return {b: i + 1 for i, b in enumerate(boats)}

    # 方法2: 艇番テキスト(実運用と同じ経路)
    nums = []
    for el in soup.find_all(class_=re.compile("boatImage1Number")):
        t = el.get_text(strip=True)
        if t.isdigit():
            nums.append(int(t))
    for grp in (nums[:6], nums[-6:]):
        if sorted(grp) == [1, 2, 3, 4, 5, 6]:
            return {b: i + 1 for i, b in enumerate(grp)}
    return None


def get(sess, page, d, jcd, rno):
    r = sess.get(f"{BASE}/{page}", params={"rno": rno, "jcd": f"{jcd:02d}", "hd": d},
                 timeout=20)
    if r.status_code != 200:
        return None, f"http_{r.status_code}"
    r.encoding = "utf-8"
    c = parse_course(r.text)
    return c, (None if c else "parse_failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="raw")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--wait", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.out, "*.json.gz")))
    if not files:
        sys.exit("raw/ がありません")

    mz, no = [], []
    for p in files:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            d = json.load(f)
        for r in d["races"]:
            if "error" in r:
                continue
            m = {e["lane"]: e["course_in"] for e in r["entries"]}
            (mz if any(l != c for l, c in m.items()) else no).append(
                (r["date"], r["jcd"], r["rno"], m))

    random.seed(args.seed)
    half = args.n // 2
    sample = random.sample(mz, min(half, len(mz))) + \
             random.sample(no, min(args.n - half, len(no)))
    random.shuffle(sample)
    print(f"raw内の前付けレース {len(mz):,}/{len(mz)+len(no):,} "
          f"({len(mz)/(len(mz)+len(no))*100:.1f}%)")
    print(f"照合 {len(sample)}レース × 2ページ  間隔{args.wait}秒\n")

    sess = requests.Session(); sess.headers.update(UA)
    hit = Counter(); fail = Counter(); done = 0
    tenji_vs_honban = [0, 0]
    ex = []

    t0 = time.time()
    for i, (d, jcd, rno, raw) in enumerate(sample):
        try:
            tenji, e1 = get(sess, "beforeinfo", d, jcd, rno)
            time.sleep(args.wait)
            honban, e2 = get(sess, "raceresult", d, jcd, rno)
            time.sleep(args.wait)
        except requests.RequestException as e:
            fail[type(e).__name__] += 1
            continue
        if e1: fail["展示_" + e1] += 1
        if e2: fail["本番_" + e2] += 1
        if not tenji or not honban:
            continue

        done += 1
        asis = raw                                    # 列=艇番, 中身=コース
        swap = {c: l for l, c in raw.items()}         # 列=コース, 中身=艇番

        for name, mine in (("そのまま", asis), ("入れ替え", swap)):
            for src, ref in (("展示", tenji), ("本番", honban)):
                if all(mine.get(b) == ref.get(b) for b in range(1, 7)):
                    hit[f"{name} vs {src}"] += 1

        tenji_vs_honban[0 if tenji == honban else 1] += 1
        if len(ex) < 6 and any(l != c for l, c in raw.items()):
            ex.append((d, jcd, rno,
                       [raw[l] for l in range(1, 7)],
                       [swap.get(l) for l in range(1, 7)],
                       [tenji[l] for l in range(1, 7)],
                       [honban[l] for l in range(1, 7)]))

        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(sample)}  {time.time()-t0:.0f}秒", flush=True)

    print("\n" + "=" * 54)
    if fail:
        print("取得できなかったもの:", dict(fail))
    if done == 0:
        print("★1件も照合できませんでした。--wait 3 で再試行してください。")
        return

    print(f"照合できたレース {done}\n")
    print("レース完全一致の割合")
    for k in ("そのまま vs 展示", "入れ替え vs 展示",
              "そのまま vs 本番", "入れ替え vs 本番"):
        v = hit[k]
        print(f"  {k}   {v:>4}/{done}  {v/done*100:5.1f}%")

    a, b = tenji_vs_honban
    print(f"\n展示進入と本番進入が同じだったレース {a}/{a+b} ({a/(a+b)*100:.1f}%)")

    if ex:
        print("\n前付けレースの実例 (艇1〜6のコース)")
        for d, jcd, rno, r1, r2, t, h in ex:
            print(f"  {d} {jcd:02d}場 {rno}R")
            print(f"    rawそのまま {r1}")
            print(f"    raw入れ替え {r2}")
            print(f"    展示        {t}")
            print(f"    本番        {h}")

    print("\n" + "=" * 54)
    best = max(hit, key=lambda k: hit[k]) if hit else None
    if best and hit[best] / done > 0.98:
        print(f"判定: raw の course_in は「{best}」")
        if "本番" in best:
            print("  → 未来の情報。前付け108.4%は無効。展示で作り直しが必要。")
        elif "入れ替え" in best:
            print("  → 向きが逆。前付けレースの course_in が壊れている。")
            print("     学習データも同じコードで作っているので、モデル自体に影響。")
        else:
            print("  → 展示進入かつ向きも正しい。バックテストは有効。")
    else:
        print("判定: どれとも一致しません。raw の course_in は別物の可能性。")
        print("  上の実例を見て、何を表しているか確認してください。")


if __name__ == "__main__":
    main()
