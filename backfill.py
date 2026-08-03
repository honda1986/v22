#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backfill.py -- 過去レースを info.kyotei.fun から一括抽出する

【v2 変更点】
  取得元を boatrace.jp から info.kyotei.fun に全面変更。
  GitHub Actions から公式サイトを叩くと弾かれる(0/144で全滅した原因)。
  info.kyotei.fun は1ページに
      出走表 / 平均ST / F数 / 展示タイム / コースIN / 3連単オッズ / 着順 / 払戻
  が全部入っているので、1レース1リクエストで済む。

  パーサは collect_v19_data.py(学習データを作ったコード)をそのまま流用。
  → 2連率が0〜1の小数、体重が整数、といった "学習時と同じ目盛り" が保証される。
     ここが1つでもズレるとモデルの確率が静かに壊れる。

■ 速度
  1日あたり 24場の1Rを叩いて開催場を判定 → 開催場だけ2〜12Rを取る = 約150リクエスト。
  workers=10 で 1日15〜25秒。90日で25〜35分。

■ 使い方
  python backfill.py --check                 # 1レースだけ取って中身を全部表示
  python backfill.py --days 90               # 直近90日
  python backfill.py --start 20260401 --end 20260630
"""

import argparse
import gzip
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import requests
from bs4 import BeautifulSoup

BASEURL = "https://info.kyotei.fun/info-{date}-{jcd:02d}-{rno}.html"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

COMBOS = [f"{a}-{b}-{c}"
          for a in range(1, 7)
          for b in range(1, 7) if b != a
          for c in range(1, 7) if c != a and c != b]
COMBO_INDEX = {c: i for i, c in enumerate(COMBOS)}

# ---- collect_v19_data.py と同じ定義(絶対に変えない) ----
RE_CLS = re.compile(r"([A12B]{2})")
RE_WEIGHT = re.compile(r"(\d+)kg", re.IGNORECASE)
RE_AGE = re.compile(r"\((\d{2})\)")
CLS_MAP = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}

DEFAULTS = {"age": 30, "cls": 1, "weight": 50, "f_count": 0, "avg_st": 0.17,
            "n_win": 0.0, "n_2ren": 0.0, "l_win": 0.0, "l_2ren": 0.0,
            "m_2ren": 0.0, "b_2ren": 0.0, "tenji": 0.0}


# ---------------------------------------------------------------- 取得

def make_session(pool):
    s = requests.Session()
    s.headers.update(UA)
    ad = requests.adapters.HTTPAdapter(pool_connections=pool, pool_maxsize=pool)
    s.mount("https://", ad)
    return s


def fetch(sess, d, jcd, rno, timeout=15):
    """(status, html) を返す。404 は再試行しない。"""
    url = BASEURL.format(date=d, jcd=jcd, rno=rno)
    last = 0
    for attempt in (1, 2):
        try:
            r = sess.get(url, timeout=timeout)
            if r.status_code == 200:
                if not r.encoding or r.encoding.lower() == "iso-8859-1":
                    r.encoding = r.apparent_encoding or "utf-8"
                return (200, r.text) if len(r.text) > 5000 else (204, "")
            if r.status_code == 404:
                return 404, ""
            last = r.status_code
        except requests.RequestException:
            last = -1
        if attempt == 1:
            time.sleep(1.5)
    return last, ""


# ---------------------------------------------------------------- パース

def _lane_from_class(td):
    div = td.find("div", class_=lambda c: c and "ng1r" in c)
    if not div:
        return None
    for cls in div.get("class", []):
        m = re.match(r"ng1r(\d)$", cls)
        if m:
            return int(m.group(1))
    return None


def parse_lane_to_rank(soup):
    lane_to_rank = {}
    jyuni = soup.find_all("div", class_="jyuni")
    if len(jyuni) >= 6:
        for i in range(6):
            t = jyuni[i].get_text(strip=True)
            if t.isdigit():
                lane_to_rank[i + 1] = int(t)
    return lane_to_rank


def parse_basic_table(soup, want_labels=False):
    base = {i + 1: dict(DEFAULTS, course_in=i + 1) for i in range(6)}
    seen = set()

    current_label = ""
    for tr in soup.find_all("tr"):
        tds = tr.find_all(["td", "th"])
        if not tds:
            continue
        if len(tds) >= 7:
            current_label = tds[0].get_text(strip=True).replace("\n", "") \
                .replace(" ", "").replace("\u3000", "")
            data_tds = tds[-6:]
        elif len(tds) == 6 and current_label:
            data_tds = tds
        else:
            current_label = ""
            continue
        seen.add(current_label)

        for i in range(6):
            td = data_tds[i]
            txt = td.get_text(" ", strip=True).replace(" ", "") \
                .replace("\u3000", "").replace("\n", "")
            lane = i + 1

            if "選手名" in current_label:
                m = RE_AGE.search(txt)
                if m:
                    base[lane]["age"] = int(m.group(1))
            elif "選手情報" in current_label or "支部" in current_label:
                m_cls = RE_CLS.search(txt)
                if m_cls:
                    base[lane]["cls"] = CLS_MAP.get(m_cls.group(1), 1)
                m_w = RE_WEIGHT.search(txt)
                if m_w:
                    base[lane]["weight"] = int(m_w.group(1))
            elif "級過去2期" in current_label:
                m_cls = RE_CLS.search(txt)
                if m_cls:
                    base[lane]["cls"] = CLS_MAP.get(m_cls.group(1), 1)
            elif "全国" in current_label and "勝率" in current_label:
                m2 = re.search(r"^([\d\.]+)", txt)
                mw = re.search(r"\(([\d\.]+)\)", txt)
                if m2:
                    v = float(m2.group(1))
                    base[lane]["n_2ren"] = v / 100.0 if v > 1.0 else v
                if mw:
                    base[lane]["n_win"] = float(mw.group(1))
            elif "当地" in current_label and "勝率" in current_label:
                m2 = re.search(r"^([\d\.]+)", txt)
                mw = re.search(r"\(([\d\.]+)\)", txt)
                if m2:
                    v = float(m2.group(1))
                    base[lane]["l_2ren"] = v / 100.0 if v > 1.0 else v
                if mw:
                    base[lane]["l_win"] = float(mw.group(1))
            elif "モータ" in current_label and "2連率" in current_label:
                m = re.search(r"^([\d\.]+)", txt)
                if m:
                    v = float(m.group(1))
                    base[lane]["m_2ren"] = v / 100.0 if v > 1.0 else v
            elif "ボート" in current_label and "2連率" in current_label:
                m = re.search(r"^([\d\.]+)", txt)
                if m:
                    v = float(m.group(1))
                    base[lane]["b_2ren"] = v / 100.0 if v > 1.0 else v
            elif "平均ST" in current_label:
                try:
                    base[lane]["avg_st"] = float(txt)
                except (ValueError, TypeError):
                    pass
            elif "フライング" in current_label:
                try:
                    base[lane]["f_count"] = int(txt)
                except (ValueError, TypeError):
                    pass
            elif current_label == "展示":
                try:
                    base[lane]["tenji"] = float(txt)
                except (ValueError, TypeError):
                    pass
            elif current_label == "コースIN":
                c = _lane_from_class(td)
                if c:
                    base[lane]["course_in"] = c

    ok = not (sum(b["n_win"] for b in base.values()) == 0 and
              sum(b["m_2ren"] for b in base.values()) == 0)
    res = base if ok else None
    return (res, seen) if want_labels else res


def parse_odds_3t(soup):
    odds = {}
    h3_target = None
    for h3 in soup.find_all("h3"):
        t = h3.get_text()
        if "3連単" in t and "人気" in t:
            h3_target = h3
            break
    if not h3_target:
        return {}
    container = h3_target.find_parent("div", id="raceData") \
        or h3_target.find_parent("div", class_="raceCard") or h3_target.parent

    for tbl in container.find_all("table", id="oddsTbl"):
        for tr in tbl.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) != 2:
                continue
            ng23 = tds[0].find("div", class_="ng23")
            if not ng23:
                continue
            divs = ng23.find_all("div", recursive=False)
            if len(divs) < 3:
                divs = ng23.find_all("div")
            nums = []
            for d in divs[:3]:
                m = re.search(r"ng2r(\d)", " ".join(d.get("class", [])))
                if m:
                    nums.append(int(m.group(1)))
            if len(nums) != 3 or len(set(nums)) != 3:
                continue
            try:
                v = float(tds[1].get_text(strip=True).replace(",", ""))
            except ValueError:
                continue
            odds[f"{nums[0]}-{nums[1]}-{nums[2]}"] = v
    return odds


def parse_payoff_3t(soup):
    for box in soup.find_all("div", class_="race_result_end_line"):
        label = box.find("div", class_="race_result_end_label")
        if label and label.get_text(strip=True) == "3連単":
            money = box.find("span", class_="race_result_end_money_num")
            if money:
                t = money.get_text(strip=True).replace(",", "")
                if t.isdigit():
                    return int(t)
    return None


def parse_weather(soup):
    """あれば拾う。無くても致命的ではないので best effort。"""
    page = soup.get_text(" ", strip=True)

    def pick(pat):
        m = re.search(pat, page)
        try:
            return float(m.group(1)) if m else None
        except ValueError:
            return None
    return {"temp": pick(r"気温[^\d\-]{0,4}(-?\d+(?:\.\d+)?)"),
            "wind": pick(r"風速[^\d\-]{0,4}(\d+(?:\.\d+)?)"),
            "wave": pick(r"波高[^\d\-]{0,4}(\d+(?:\.\d+)?)"),
            "water_temp": pick(r"水温[^\d\-]{0,4}(-?\d+(?:\.\d+)?)")}


# ---------------------------------------------------------------- 1レース

def race_record(d, jcd, rno, html):
    soup = BeautifulSoup(html, "html.parser")

    lane_to_rank = parse_lane_to_rank(soup)
    if not lane_to_rank:
        return {"date": d, "jcd": jcd, "rno": rno, "error": "no_result"}

    base = parse_basic_table(soup)
    if not base:
        return {"date": d, "jcd": jcd, "rno": rno, "error": "no_basic"}

    odds_map = parse_odds_3t(soup)
    odds = [odds_map.get(c) for c in COMBOS]

    r1 = next((l for l, rk in lane_to_rank.items() if rk == 1), None)
    r2 = next((l for l, rk in lane_to_rank.items() if rk == 2), None)
    r3 = next((l for l, rk in lane_to_rank.items() if rk == 3), None)
    hit = f"{r1}-{r2}-{r3}" if (r1 and r2 and r3) else None

    entries = []
    for lane in range(1, 7):
        b = base[lane]
        entries.append({
            "lane": lane, "cls_val": b["cls"], "age": b["age"],
            "weight": b["weight"], "f_count": b["f_count"], "avg_st": b["avg_st"],
            "n_win": b["n_win"], "n_2ren": b["n_2ren"],
            "l_win": b["l_win"], "l_2ren": b["l_2ren"],
            "m_2ren": b["m_2ren"], "b_2ren": b["b_2ren"],
            "tenji": b["tenji"], "course_in": b["course_in"],
            "rank": lane_to_rank.get(lane, 6),
        })

    return {"date": d, "jcd": jcd, "rno": rno,
            "entries": entries, "weather": parse_weather(soup),
            "odds": odds, "n_odds": sum(1 for o in odds if o),
            "hit": hit, "pay_3t": parse_payoff_3t(soup)}


def get_race(sess, d, jcd, rno):
    st, html = fetch(sess, d, jcd, rno)
    if st != 200:
        return {"date": d, "jcd": jcd, "rno": rno, "error": f"http_{st}"}
    try:
        return race_record(d, jcd, rno, html)
    except Exception as e:
        return {"date": d, "jcd": jcd, "rno": rno, "error": f"exc_{type(e).__name__}"}


# ---------------------------------------------------------------- 1日

def fetch_day(sess, d, workers):
    # まず各場の1Rを叩いて開催場を判定(24リクエスト)
    with ThreadPoolExecutor(max_workers=min(workers, 12)) as ex:
        first = list(ex.map(lambda j: get_race(sess, d, j, 1), range(1, 25)))
    live = [r["jcd"] for r in first if "error" not in r]
    races = [r for r in first if "error" not in r]

    if not live:
        return []

    tasks = [(j, rno) for j in live for rno in range(2, 13)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(lambda t: get_race(sess, d, t[0], t[1]), tasks):
            races.append(r)

    races.sort(key=lambda x: (x["jcd"], x["rno"]))
    return races


def save_day(outdir, d, races):
    os.makedirs(outdir, exist_ok=True)
    with gzip.open(os.path.join(outdir, f"{d}.json.gz"), "wt", encoding="utf-8") as f:
        json.dump({"date": d, "combos": COMBOS, "races": races},
                  f, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------- check

def do_check(sess, d, jcd, rno):
    url = BASEURL.format(date=d, jcd=jcd, rno=rno)
    print(f"=== {url} ===")
    st, html = fetch(sess, d, jcd, rno)
    print(f"HTTP {st}  本文 {len(html)}文字\n")
    if st != 200:
        print("取得できていません。日付/場/Rを変えて再試行してください。")
        return
    soup = BeautifulSoup(html, "html.parser")

    base, labels = parse_basic_table(soup, want_labels=True)
    print("[見つかった行ラベル]")
    print("  " + " / ".join(sorted(x for x in labels if x)[:40]) + "\n")

    print("[着順]", parse_lane_to_rank(soup), "\n")
    print("[出走表]")
    if base:
        for lane in range(1, 7):
            print("  ", json.dumps(dict(base[lane], lane=lane), ensure_ascii=False))
    else:
        print("  取得失敗 ← ラベル名が変わっている可能性")
    print()

    om = parse_odds_3t(soup)
    print(f"[3連単オッズ] {len(om)}/120点")
    pay = parse_payoff_3t(soup)
    rk = parse_lane_to_rank(soup)
    r1 = next((l for l, v in rk.items() if v == 1), None)
    r2 = next((l for l, v in rk.items() if v == 2), None)
    r3 = next((l for l, v in rk.items() if v == 3), None)
    if r1 and r2 and r3:
        hit = f"{r1}-{r2}-{r3}"
        o = om.get(hit)
        print(f"  的中 {hit}  オッズ {o}  払戻 {pay}円")
        if o and pay:
            print(f"  → オッズ×100 = {o*100:.0f} / 払戻 {pay}  "
                  f"{'一致 OK' if abs(o*100-pay) < 11 else '★不一致 要確認'}")
    if om:
        s = sum(1 / v for v in om.values() if v)
        print(f"  控除率の目安 {1-1/s:.1%}  (25%前後なら正常)")
    print("\n[気象]", json.dumps(parse_weather(soup), ensure_ascii=False))
    print("\n--- 上が正しければ本番実行してOK ---")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out", default="raw")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--minutes", type=float, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-date")
    ap.add_argument("--check-jcd", type=int, default=12)
    ap.add_argument("--check-rno", type=int, default=12)
    args = ap.parse_args()

    sess = make_session(args.workers * 2)

    if args.check:
        d = args.check_date or (date.today() - timedelta(days=7)).strftime("%Y%m%d")
        do_check(sess, d, args.check_jcd, args.check_rno)
        return

    if args.start and args.end:
        d0 = datetime.strptime(args.start, "%Y%m%d").date()
        d1 = datetime.strptime(args.end, "%Y%m%d").date()
    else:
        d1 = date.today() - timedelta(days=1)
        d0 = d1 - timedelta(days=args.days - 1)

    days, d = [], d1
    while d >= d0:
        days.append(d.strftime("%Y%m%d"))
        d -= timedelta(days=1)

    t0 = time.time()
    total_ok = 0
    for hd in days:
        if os.path.exists(os.path.join(args.out, f"{hd}.json.gz")):
            continue
        if args.minutes and (time.time() - t0) / 60 > args.minutes:
            print(f"[stop] 時間切れ。再実行で {hd} から再開します。", flush=True)
            break
        t = time.time()
        races = fetch_day(sess, hd, args.workers)
        good = [r for r in races if "error" not in r]
        save_day(args.out, hd, races)
        total_ok += len(good)
        if races:
            errs = {}
            for r in races:
                if "error" in r:
                    errs[r["error"]] = errs.get(r["error"], 0) + 1
            tail = ("  " + " ".join(f"{k}={v}" for k, v in sorted(errs.items()))) if errs else ""
            print(f"{hd}: {len(good)}/{len(races)}  {time.time()-t:.0f}秒  "
                  f"累計{total_ok}{tail}", flush=True)
        else:
            print(f"{hd}: 開催なし", flush=True)

    print(f"\n完了: {total_ok}レース / {(time.time()-t0)/60:.1f}分")


if __name__ == "__main__":
    main()
