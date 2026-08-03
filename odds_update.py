# -*- coding: utf-8 -*-
"""
odds_update.py — races.json のオッズだけを取り直し、EV と買い目を再計算する

予想(確率)は predict.py が計算済みで変わらない。締切まで動き続けるのは
オッズだけなので、そこだけを短い間隔で更新して期待値の鮮度を保つ。

1レース1リクエストなので軽く、10分おきに回せる。

使い方:
  python odds_update.py            締切前のレースだけ更新(通常運用)
  python odds_update.py --all      締切済みも含めて全レース更新
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Dict

import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup

JST = timezone(timedelta(hours=+9), "JST")
DIR = os.path.dirname(os.path.abspath(__file__))

# predict.py と同じ選定条件(ここを変えるときは両方そろえること)
EV_MIN = 1.10
TOP_N_PROB = 12
MAX_POINTS = 8

# 締切の何分後まで更新対象に含めるか(締切直後は結果待ちなので少しだけ猶予)
GRACE_MIN = 2

sess = requests.Session()
sess.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
try:
    from urllib3.util.retry import Retry
except Exception:
    from requests.packages.urllib3.util.retry import Retry  # type: ignore
try:
    _r = Retry(total=2, connect=2, read=2, status=2, backoff_factor=0.4,
               status_forcelist=(429, 500, 502, 503, 504),
               allowed_methods=frozenset(["GET"]), raise_on_status=False)
except TypeError:
    _r = Retry(total=2, backoff_factor=0.4, status_forcelist=(429, 500, 502, 503, 504),
               method_whitelist=frozenset(["GET"]), raise_on_status=False)
sess.mount("https://", HTTPAdapter(max_retries=_r, pool_connections=16, pool_maxsize=16))


def fetch_odds3t(date: str, jcd: int, rno: int) -> Dict[str, float]:
    """公式の3連単オッズ。表構造から組番を厳密に復元する(predict.py と同じ実装)。"""
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={date}"
    try:
        r = sess.get(url, timeout=10)
        r.encoding = r.apparent_encoding
        if r.status_code != 200 or len(r.text) < 3000:
            return {}
        soup = BeautifulSoup(r.text, "html.parser")
    except requests.RequestException:
        return {}
    table = None
    for tbl in soup.find_all("table"):
        if tbl.select("td.oddsPoint"):
            table = tbl
            break
    if table is None:
        return {}
    heads = []
    thead = table.find("thead")
    if thead:
        for th in thead.find_all("th"):
            t = th.get_text(strip=True)
            if t.isdigit() and 1 <= int(t) <= 6:
                heads.append(int(t))
    if not (2 <= len(heads) <= 6):
        heads = [1, 2, 3, 4, 5, 6]
    out, cur2 = {}, [None] * len(heads)
    for tr in table.select("tbody > tr"):
        tds = tr.find_all("td", recursive=False)
        twos = [td for td in tds if td.has_attr("rowspan") and "oddsPoint" not in (td.get("class") or [])]
        if len(twos) == len(heads):
            for gi, td in enumerate(twos):
                tv = td.get_text(strip=True)
                if tv.isdigit():
                    cur2[gi] = int(tv)
        gi, last = 0, None
        for td in tds:
            cls = td.get("class") or []
            txt = td.get_text(strip=True)
            if "oddsPoint" in cls:
                if gi < len(heads):
                    a, b, c = heads[gi], cur2[gi], last
                    if a and b and c and len({a, b, c}) == 3:
                        try:
                            v = float(txt.replace(",", ""))
                        except ValueError:
                            v = 0.0
                        if v > 0:
                            out[f"{a}-{b}-{c}"] = v
                gi += 1
            elif txt.isdigit():
                last = int(txt)
    return out


def pick_buys(probs: Dict[str, float], odds: Dict[str, float]):
    """確率上位から絞り、EV が基準以上のものを採用(predict.py と同じ)。"""
    if not probs:
        return []
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:TOP_N_PROB]
    rows = []
    for combo, p in top:
        o = odds.get(combo)
        if not o:
            continue
        ev = p * o
        if ev >= EV_MIN:
            rows.append({"combo": combo, "p": round(p * 100, 2), "odds": round(o, 1), "ev": round(ev, 2)})
    rows.sort(key=lambda r: -r["ev"])
    return rows[:MAX_POINTS]


def main():
    all_mode = "--all" in sys.argv
    path = os.path.join(DIR, "races.json")
    if not os.path.exists(path):
        print("races.json がありません。先に predict.py を実行してください。")
        return
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    races = doc.get("races", [])
    if not races:
        print("レースがありません。")
        return

    now = datetime.now(JST)
    hm = now.strftime("%H:%M")
    date = doc.get("date") or now.strftime("%Y%m%d")

    def still_open(r):
        c = r.get("close")
        if not c:
            return True
        try:
            dt = datetime.strptime(f"{date}{c}", "%Y%m%d%H:%M").replace(tzinfo=JST)
        except ValueError:
            return True
        return now <= dt + timedelta(minutes=GRACE_MIN)

    targets = races if all_mode else [r for r in races if still_open(r)]
    print(f"現在 {hm} / 全{len(races)}レース中 {len(targets)}レースを更新対象にします")
    if not targets:
        print("締切前のレースがありません。更新不要。")
        return

    # 確率は predict.py が計算済み。ここでは買い目候補の確率だけを使って
    # EV を計算し直す(120点すべての確率は races.json に持たないため、
    # 直近の買い目候補＋確率上位の情報から再構成する)。
    updated, changed, failed = 0, 0, 0

    def work(r):
        return r, fetch_odds3t(date, r["jcd"], r["rno"])

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(work, r) for r in targets]
        for fu in as_completed(futs):
            try:
                r, odds = fu.result()
            except Exception:
                failed += 1
                continue
            if len(odds) < 50:
                failed += 1
                continue
            probs = r.get("probs") or {}
            if not probs:
                # 旧形式(確率を保存していない)の場合は、既存の買い目の確率だけで再計算する
                probs = {b["combo"]: b["p"] / 100 for b in (r.get("buys") or [])}
            if not probs:
                failed += 1
                continue
            before = [b["combo"] for b in (r.get("buys") or [])]
            buys = pick_buys(probs, odds)
            r["buys"] = buys
            r["points"] = len(buys)
            r["hitProb"] = round(sum(b["p"] for b in buys), 1)
            r["evTotal"] = round(sum(b["p"] / 100 * b["odds"] for b in buys) / len(buys), 2) if buys else 0
            r["oddsCount"] = len(odds)
            r["oddsAt"] = now.isoformat()
            updated += 1
            if [b["combo"] for b in buys] != before:
                changed += 1

    doc["oddsUpdatedAt"] = now.isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)

    buy_now = [r for r in races if (r.get("buys") or []) and still_open(r)]
    print(f"完了: {updated}レース更新 / 買い目が変わった {changed}レース / 取得失敗 {failed}")
    print(f"締切前で買い目があるレース: {len(buy_now)}")
    for r in sorted(buy_now, key=lambda x: x.get("close") or "99:99")[:8]:
        print(f"  {r.get('close')} {r['place']}{r['rno']}R  {r['points']}点 "
              f"平均EV{r['evTotal']} 的中率{r['hitProb']}%  " + " ".join(b["combo"] for b in r["buys"][:4]))


if __name__ == "__main__":
    main()
