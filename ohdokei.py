#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ohdokei.py -- 「大時計」アプリのデータを作る

■ 考え方
  135,327レースの検証で分かったこと:
    ・自作モデルは市場を超えない(必要な優位の1%しか無い)
    ・したがって最良の確率推定は市場そのもの
    ・ただし市場には偏りがあり、本命は過小評価・穴は買われすぎ

  なので予想はしない。人気順の上位N点を買い、
  実測したオッズ帯別の回収率から、買う前に正直な数字を出す。

■ 較正表(26,957レースの実測)
  オッズ帯ごとの実測回収率。理論値は控除率25%なので75%。
  ここから 実確率 = 実測回収率 / オッズ が出る。
  期待回収率は、選んだ買い目の帯別回収率の平均そのもの。

■ 出力
  ohdokei/data.json  (静的ページが読む)
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup

JST = timezone(timedelta(hours=9))
BASE = "https://www.boatrace.jp/owpc/pc/race"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept-Language": "ja"}

VENUE = {1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
         7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
         13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
         19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村"}

COMBOS = [f"{a}-{b}-{c}"
          for a in range(1, 7)
          for b in range(1, 7) if b != a
          for c in range(1, 7) if c != a and c != b]
CIX = {c: i for i, c in enumerate(COMBOS)}

# オッズ帯ごとの実測回収率 (下限, 上限, 実測回収率)
# 2026-01〜08 の26,957レース、320万点から測定
CALIB = [(1, 5, 0.869), (5, 10, 0.797), (10, 20, 0.799), (20, 40, 0.746),
         (40, 80, 0.755), (80, 160, 0.735), (160, 400, 0.669), (400, 1e9, 0.274)]


def band_roi(o):
    for lo, hi, v in CALIB:
        if lo <= o < hi:
            return v
    return 0.274


# ---------------------------------------------------------------- 取得
def make_session():
    s = requests.Session()
    s.headers.update(UA)
    return s


def get(sess, page, **params):
    r = sess.get(f"{BASE}/{page}", params=params, timeout=25)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text


def parse_odds3t(html):
    """3連単120点。1着ごとに6ブロック×20行。並び順から組番を復元する。"""
    soup = BeautifulSoup(html, "html.parser")
    best, tbl = 0, None
    for t in soup.find_all("table"):
        n = len(re.findall(r">\s*\d+\.\d\s*<", str(t)))
        if n > best:
            best, tbl = n, t
    if tbl is None:
        return None
    rows = []
    for tr in tbl.find_all("tr"):
        v = [float(td.get_text(strip=True)) for td in tr.find_all("td")
             if re.fullmatch(r"\d+\.\d+", td.get_text(strip=True))]
        if len(v) == 6:
            rows.append(v)
    if len(rows) != 20:
        return None
    out = [None] * 120
    for r, vals in enumerate(rows):
        for g, v in enumerate(vals):
            first = g + 1
            others = [b for b in range(1, 7) if b != first]
            second = others[r // 4]
            third = [b for b in others if b != second][r % 4]
            out[CIX[f"{first}-{second}-{third}"]] = v
    return None if any(x is None for x in out) else out


def parse_schedule(html):
    """締切予定時刻の行から12レース分の時刻"""
    soup = BeautifulSoup(html, "html.parser")
    for tr in soup.find_all("tr"):
        if "締切予定時刻" not in tr.get_text():
            continue
        t = re.findall(r"\b(\d{1,2}:\d{2})\b", tr.get_text(" "))
        if len(t) >= 12:
            return t[:12]
    return None


def parse_resultlist(html):
    """場×日の結果一覧から3連単の組番と払戻"""
    soup = BeautifulSoup(html, "html.parser")
    out = {}
    for tr in soup.find_all("tr"):
        a = tr.find("a", href=re.compile(r"raceresult\?rno=(\d+)"))
        if not a:
            continue
        txt = tr.get_text(" ", strip=True)
        if "¥" not in txt:
            continue
        rno = int(re.search(r"raceresult\?rno=(\d+)", a["href"]).group(1))
        if rno in out:
            continue
        m = re.search(r"(?<!\d)([1-6])\s*-\s*([1-6])\s*-\s*([1-6])(?!\d)", txt)
        if not m:
            continue
        pays = re.findall(r"¥\s*([\d,]+)", txt)
        out[rno] = {"hit": f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
                    "pay": int(pays[0].replace(",", "")) if pays else None,
                    "henkan": "返還" in txt}
    return out


# ---------------------------------------------------------------- 買い目
def build_picks(odds, n_points):
    """人気順(オッズの安い順)に n_points 点。較正済みの確率と期待回収率を付ける。"""
    order = sorted(range(120), key=lambda i: odds[i])[:n_points]
    pts, hit, roi = [], 0.0, 0.0
    for i in order:
        o = odds[i]
        r = band_roi(o)
        p = r / o                      # 実測回収率 ÷ オッズ = 実確率
        pts.append({"c": COMBOS[i], "o": round(o, 1), "p": round(p, 4)})
        hit += p
        roi += r
    pays = [p["o"] * 100 for p in pts]
    return {"points": pts,
            "hit_rate": round(hit, 4),
            "exp_roi": round(roi / len(pts), 4),
            "pay_lo": int(min(pays)), "pay_hi": int(max(pays))}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ohdokei/data.json")
    ap.add_argument("--points", type=int, default=6)
    ap.add_argument("--window", type=int, default=40, help="締切まで何分先まで拾うか")
    ap.add_argument("--max-odds", type=int, default=10, help="1回で取るオッズの上限件数")
    args = ap.parse_args()

    now = datetime.now(JST)
    today = now.strftime("%Y%m%d")
    sess = make_session()

    # 既存データ(同じ日なら引き継ぐ)
    data = {"date": today, "points": args.points, "venues": {}, "races": {}}
    if os.path.exists(args.out):
        try:
            old = json.load(open(args.out, encoding="utf-8"))
            if old.get("date") == today:
                data = old
                data["points"] = args.points
        except Exception:
            pass

    # 1) 今日の開催と締切時刻(1日1回だけ)
    if not data["venues"]:
        print("今日の開催を調べます")
        for jcd in range(1, 25):
            try:
                sc = parse_schedule(get(sess, "odds3t", rno=1,
                                        jcd=f"{jcd:02d}", hd=today))
            except Exception:
                sc = None
            if sc:
                data["venues"][str(jcd)] = sc
                print(f"  {VENUE[jcd]} {sc[0]}〜{sc[11]}", flush=True)
        if not data["venues"]:
            print("本日の開催はありません")

    # 2) 締切が近いレースのオッズを取る
    targets = []
    for jcd_s, times in data["venues"].items():
        jcd = int(jcd_s)
        for rno, t in enumerate(times, 1):
            key = f"{today}-{jcd:02d}-{rno}"
            hh, mm = (int(x) for x in t.split(":"))
            close = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            mins = (close - now).total_seconds() / 60
            if 2 < mins <= args.window:
                targets.append((mins, jcd, rno, key, t))
    targets.sort()
    print(f"\n締切{args.window}分以内のレース {len(targets)}件")

    for mins, jcd, rno, key, t in targets[:args.max_odds]:
        try:
            odds = parse_odds3t(get(sess, "odds3t", rno=rno,
                                    jcd=f"{jcd:02d}", hd=today))
        except Exception as e:
            print(f"  {VENUE[jcd]}{rno}R 取得失敗 {type(e).__name__}")
            continue
        if not odds:
            continue
        rec = build_picks(odds, args.points)
        rec.update({"jcd": jcd, "venue": VENUE[jcd], "rno": rno, "close": t,
                    "odds_at": now.strftime("%H:%M"),
                    "result": data["races"].get(key, {}).get("result")})
        data["races"][key] = rec
        print(f"  {VENUE[jcd]}{rno}R {t}締切  的中率{rec['hit_rate']*100:.0f}%  "
              f"回収率{rec['exp_roi']*100:.0f}%", flush=True)

    # 3) 終わったレースの結果を入れる
    need = {}
    for key, r in data["races"].items():
        if r.get("result"):
            continue
        hh, mm = (int(x) for x in r["close"].split(":"))
        close = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if (now - close).total_seconds() / 60 > 12:
            need.setdefault(r["jcd"], []).append((key, r["rno"]))
    for jcd, items in list(need.items())[:8]:
        try:
            res = parse_resultlist(get(sess, "resultlist",
                                       jcd=f"{jcd:02d}", hd=today))
        except Exception:
            continue
        for key, rno in items:
            if rno not in res:
                continue
            rr = res[rno]
            picks = [p["c"] for p in data["races"][key]["points"]]
            data["races"][key]["result"] = {
                "hit": rr["hit"], "pay": rr["pay"],
                "won": rr["hit"] in picks, "henkan": rr["henkan"]}
            mark = "的中" if rr["hit"] in picks else "不的中"
            print(f"  結果 {VENUE[jcd]}{rno}R {rr['hit']} {mark}")

    data["updated"] = now.strftime("%Y-%m-%d %H:%M")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"\n{args.out} を更新しました (レース{len(data['races'])}件)")


if __name__ == "__main__":
    main()
