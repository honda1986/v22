# -*- coding: utf-8 -*-
"""
results.py — 本日のレース結果を取得し、予想(races.json)と突き合わせて成績を集計する

出力:
  history.json … 1レース1件の記録(累積)
  stats.json   … 本日/累計/日別の集計 + 本日の結果(アプリの的中表示用)

使い方: python results.py [YYYYMMDD]
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup

JST = timezone(timedelta(hours=+9), "JST")
DIR = os.path.dirname(os.path.abspath(__file__))

sess = requests.Session()
sess.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
try:
    from urllib3.util.retry import Retry
except Exception:
    from requests.packages.urllib3.util.retry import Retry  # type: ignore
try:
    _r = Retry(total=3, connect=3, read=3, status=3, backoff_factor=0.6,
               status_forcelist=(429, 500, 502, 503, 504),
               allowed_methods=frozenset(["GET"]), raise_on_status=False)
except TypeError:
    _r = Retry(total=3, backoff_factor=0.6, status_forcelist=(429, 500, 502, 503, 504),
               method_whitelist=frozenset(["GET"]), raise_on_status=False)
sess.mount("https://", HTTPAdapter(max_retries=_r, pool_connections=16, pool_maxsize=16))


def load(name, default):
    p = os.path.join(DIR, name)
    if not os.path.exists(p):
        return default
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save(name, obj):
    with open(os.path.join(DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def fetch_result(date: str, jcd: int, rno: int) -> Optional[Dict]:
    """公式のレース結果から3連単の的中組番と払戻金を取得する。

    払戻テーブルは賭式ごとに行があり、組番は数字セル、払戻は「¥1,234」形式。
    「3連単」の行だけを厳密に拾う(3連複と取り違えないよう文字列一致で判定)。
    """
    url = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd:02d}&hd={date}"
    try:
        r = sess.get(url, timeout=12)
        r.encoding = r.apparent_encoding
        if r.status_code != 200 or len(r.text) < 3000:
            return None
    except requests.RequestException:
        return None
    soup = BeautifulSoup(r.text, "html.parser")

    for td in soup.find_all(["td", "th"]):
        if td.get_text(strip=True) != "3連単":
            continue
        row = td.find_parent("tr")
        if row is None:
            continue
        # 同じ行(または続く行)から組番と払戻を拾う
        scope = [row]
        nxt = row.find_next_sibling("tr")
        if nxt is not None:
            scope.append(nxt)
        nums, pay = [], None
        for tr in scope:
            for el in tr.find_all(["span", "td"]):
                t = el.get_text(strip=True)
                if re.fullmatch(r"[1-6]", t) and len(nums) < 3:
                    nums.append(t)
                m = re.fullmatch(r"[¥￥]([\d,]+)", t)
                if m and pay is None:
                    pay = int(m.group(1).replace(",", ""))
            if len(nums) >= 3 and pay is not None:
                break
        if len(nums) >= 3 and len(set(nums)) == 3:
            return {"combo": "-".join(nums[:3]), "pay": pay or 0}
    return None


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now(JST).strftime("%Y%m%d")
    races_doc = load("races.json", {"races": []})
    races = [r for r in races_doc.get("races", []) if r.get("date") == date]
    if not races:
        print("races.json に", date, "のレースがありません。予想を先に実行してください。")
        return
    print("対象:", len(races), "レース(", date, ")")

    hist = load("history.json", {"entries": []})
    done = {e["key"] for e in hist["entries"]}

    targets = [r for r in races if r["key"] not in done]
    print("未取得:", len(targets), "レース")
    added = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch_result, date, r["jcd"], r["rno"]): r for r in targets}
        for f in as_completed(futs):
            r = futs[f]
            try:
                res = f.result()
            except Exception:
                res = None
            if not res:
                continue
            buys = [b["combo"] for b in (r.get("buys") or [])]
            hit = res["combo"] in buys
            hist["entries"].append({
                "key": r["key"], "date": date, "jcd": r["jcd"], "place": r["place"], "rno": r["rno"],
                "close": r.get("close", ""), "buys": buys, "points": len(buys),
                "combo": res["combo"], "pay": res["pay"],
                "hit": hit, "cost": len(buys) * 100, "ret": res["pay"] if hit else 0,
                "hitProb": r.get("hitProb"), "evTotal": r.get("evTotal"),
            })
            added += 1
    print("追加:", added, "件 / 累計", len(hist["entries"]), "件")
    save("history.json", hist)

    # ---- 集計(買い目があったレースのみを対象にする) ----
    E = [e for e in hist["entries"] if e["points"] > 0]

    def summarize(arr):
        if not arr:
            return None
        cost = sum(e["cost"] for e in arr)
        ret = sum(e["ret"] for e in arr)
        hits = sum(1 for e in arr if e["hit"])
        return {
            "races": len(arr),
            "hit": round(hits / len(arr) * 100, 1),
            "roi": round(ret / cost * 100, 1) if cost else 0,
            "profit": ret - cost,
        }

    by_date = {}
    for e in E:
        by_date.setdefault(e["date"], []).append(e)
    recent = [dict(date=d, **(summarize(by_date[d]) or {}))
              for d in sorted(by_date, reverse=True)[:7]]

    stats = {
        "updatedAt": datetime.now(JST).isoformat(),
        "todayDate": date,
        "today": summarize(by_date.get(date, [])),
        "overall": summarize(E),
        "recentDays": recent,
        # アプリが「的中/不的中」を表示するための本日分の結果
        "todayRaces": [{"key": e["key"], "combo": e["combo"], "pay": e["pay"]}
                       for e in hist["entries"] if e["date"] == date],
    }
    save("stats.json", stats)

    t, o = stats["today"], stats["overall"]
    if t:
        print(f"本日: {t['races']}R / 的中{t['hit']}% / 回収率{t['roi']}% / 収支{t['profit']:+,}円")
    if o:
        print(f"累計: {o['races']}R / 的中{o['hit']}% / 回収率{o['roi']}% / 収支{o['profit']:+,}円")


if __name__ == "__main__":
    main()
