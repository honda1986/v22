#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backfill.py  --  過去レースを boatrace.jp から一括抽出する

history.json が貯まるのを待たずに、勝負レースの絞り込みを検証するための
データを過去から作る。1日分ずつ raw/YYYYMMDD.json.gz に保存する。

■ 重要な発見
  公式サイトは「締切時オッズ」を過去レース分もずっと残している。
  例: /owpc/pc/race/odds3t?rno=7&jcd=03&hd=20170329 → 2017年のレースでも120点全部出る。
  つまり EV(確率×オッズ)ベースの検証が過去データでそのまま出来る。

■ 1レースあたりのリクエスト数
  racelist(出走表) + beforeinfo(直前) + odds3t(オッズ) = 3
  結果は resultlist(場×日で1枚に12レース分) なので 1/12 で済む。
  → 1日 約470リクエスト(開催12場想定)。90日で約4.2万。
    workers=8 で 30〜40分ほど。

■ 使い方
  python backfill.py --check                    # 1レースだけ取って中身を表示(最初に必ずこれ)
  python backfill.py --days 90                  # 直近90日(今日は含めない)を取得
  python backfill.py --start 20260401 --end 20260630
  python backfill.py --days 90 --minutes 300    # 5時間で打ち切り(続きは再実行で再開)

  既に raw/ にある日はスキップするので、何回でも再実行して続きから進められる。
"""

import argparse
import gzip
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://www.boatrace.jp/owpc/pc/race"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " \
     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# 3連単120点の並び順(この順で odds を配列として保存する)
COMBOS = [f"{a}-{b}-{c}"
          for a in range(1, 7)
          for b in range(1, 7) if b != a
          for c in range(1, 7) if c != a and c != b]
COMBO_INDEX = {c: i for i, c in enumerate(COMBOS)}

CLS_VAL = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}


# ---------------------------------------------------------------- HTTP

def make_session(pool: int):
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.6,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"])
    ad = HTTPAdapter(max_retries=retry, pool_connections=pool, pool_maxsize=pool)
    s.mount("https://", ad)
    s.mount("http://", ad)
    s.headers.update({"User-Agent": UA, "Accept-Language": "ja,en;q=0.8"})
    return s


def get(session, path, **params):
    url = f"{BASE}/{path}"
    r = session.get(url, params=params, timeout=25)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text


# ---------------------------------------------------------------- 小道具

def _floats_after(text):
    """テキストから数値トークンを出現順に取り出す"""
    return re.findall(r"-?\d+(?:\.\d+)?", text)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _racer_tbodies(soup):
    """選手プロフィールへのリンクを含む tbody(=1艇分)を枠番付きで返す"""
    out = []
    for tb in soup.find_all("tbody"):
        if not tb.find("a", href=re.compile(r"toban=\d+")):
            continue
        tds = tb.find_all("td")
        if not tds:
            continue
        head = tds[0].get_text(strip=True)
        m = re.match(r"^([1-6])$", head)
        if not m:
            continue
        out.append((int(m.group(1)), tb))
    # 枠番の重複を除去(念のため)
    seen, uniq = set(), []
    for lane, tb in out:
        if lane in seen:
            continue
        seen.add(lane)
        uniq.append((lane, tb))
    return sorted(uniq, key=lambda x: x[0])


# ---------------------------------------------------------------- 出走表

def parse_racelist(html):
    """出走表: 級別・年齢・体重・F数・平均ST・全国/当地勝率・モーター/ボート2連率"""
    soup = BeautifulSoup(html, "html.parser")
    entries = []
    for lane, tb in _racer_tbodies(soup):
        full = tb.get_text("\n", strip=True)

        m_toban = re.search(r"(\d{4})\s*/\s*(A1|A2|B1|B2)", full)
        toban = int(m_toban.group(1)) if m_toban else None
        cls = m_toban.group(2) if m_toban else None
        if cls is None:
            m_cls = re.search(r"\b(A1|A2|B1|B2)\b", full)
            cls = m_cls.group(1) if m_cls else None

        m_age = re.search(r"(\d+)歳", full)
        m_wt = re.search(r"(\d+(?:\.\d+)?)kg", full)

        # F数 / L数 / 平均ST は縦に並ぶ("F0","L0","0.18")
        m_fl = re.search(r"F\s*(\d+)\s*[\n\s]*L\s*(\d+)\s*[\n\s]*(\d+\.\d+)", full)
        if m_fl:
            f_count, l_count, avg_st = int(m_fl.group(1)), int(m_fl.group(2)), float(m_fl.group(3))
            tail = full[m_fl.end():]
        else:
            f_count = l_count = avg_st = None
            tail = full

        # F/L/STの後ろは 全国3・当地3・モーター3・ボート3 の順に12個並ぶ
        nums = _floats_after(tail)[:12]
        g = (lambda i: _f(nums[i]) if i < len(nums) else None)

        name_a = tb.find("a", href=re.compile(r"toban=\d+"))
        entries.append({
            "lane": lane,
            "toban": toban,
            "name": name_a.get_text(strip=True) if name_a else None,
            "cls": cls,
            "cls_val": CLS_VAL.get(cls),
            "age": int(m_age.group(1)) if m_age else None,
            "weight_list": _f(m_wt.group(1)) if m_wt else None,
            "f_count": f_count,
            "l_count": l_count,
            "avg_st": avg_st,
            "n_win": g(0), "n_2ren": g(1), "n_3ren": g(2),
            "l_win": g(3), "l_2ren": g(4), "l_3ren": g(5),
            "motor_no": g(6), "m_2ren": g(7), "m_3ren": g(8),
            "boat_no": g(9), "b_2ren": g(10), "b_3ren": g(11),
        })
    return entries if len(entries) == 6 else None


# ---------------------------------------------------------------- 直前情報

def parse_beforeinfo(html):
    """直前情報: 展示タイム・チルト・当日体重・スタート展示(進入コースと展示ST)・気象"""
    soup = BeautifulSoup(html, "html.parser")

    boats = {}
    for lane, tb in _racer_tbodies(soup):
        full = tb.get_text("\n", strip=True)
        m_wt = re.search(r"(\d+(?:\.\d+)?)kg", full)
        # 展示タイムだけが小数2桁("6.80")。チルト/調整重量は1桁("0.5","0.0")
        m_tenji = re.search(r"(?<![\d.])(\d\.\d\d)(?![\d])", full)
        m_tilt = re.search(r"(?<![\d.])(-?\d\.\d)(?![\d])", full)
        boats[lane] = {
            "lane": lane,
            "weight": _f(m_wt.group(1)) if m_wt else None,
            "tenji": _f(m_tenji.group(1)) if m_tenji else None,
            "tilt": _f(m_tilt.group(1)) if m_tilt else None,
        }

    # スタート展示: コース順に img_boat2_N.png(N=艇番)と ST(".04" / "F.05")
    start = []
    for tb in soup.find_all(["tbody", "table"]):
        imgs = tb.find_all("img", src=re.compile(r"img_boat2_\d\.png"))
        if len(imgs) < 3:
            continue
        rows = []
        for tr in tb.find_all("tr"):
            img = tr.find("img", src=re.compile(r"img_boat2_(\d)\.png"))
            if not img:
                continue
            boat = int(re.search(r"img_boat2_(\d)\.png", img["src"]).group(1))
            txt = tr.get_text(" ", strip=True)
            m_st = re.search(r"([FL]?)\s*\.(\d{2})", txt)
            st, flag = None, ""
            if m_st:
                flag = m_st.group(1) or ""
                st = float("0." + m_st.group(2))
                if flag == "F":
                    st = -st          # フライングはマイナス表記にする
            rows.append({"course": len(rows) + 1, "lane": boat,
                         "st": st, "st_flag": flag})
        if len(rows) >= 3:
            start = rows
            break

    for r in start:
        if r["lane"] in boats:
            boats[r["lane"]]["course_in"] = r["course"]
            boats[r["lane"]]["tenji_st"] = r["st"]
            boats[r["lane"]]["tenji_st_flag"] = r["st_flag"]

    for lane, b in boats.items():
        b.setdefault("course_in", lane)      # 展示が取れない場合は枠なり
        b.setdefault("tenji_st", None)
        b.setdefault("tenji_st_flag", "")
        b["maezuke"] = 1 if b["course_in"] != lane else 0
        b["course_diff"] = b["course_in"] - lane

    # 水面気象
    page = soup.get_text(" ", strip=True)
    def pick(pat):
        m = re.search(pat, page)
        return _f(m.group(1)) if m else None
    m_wind = re.search(r"is-wind(\d+)", html)
    weather = {
        "temp": pick(r"気温\s*(-?\d+(?:\.\d+)?)"),
        "wind": pick(r"風速\s*(\d+(?:\.\d+)?)"),
        "water_temp": pick(r"水温\s*(-?\d+(?:\.\d+)?)"),
        "wave": pick(r"波高\s*(\d+(?:\.\d+)?)"),
        "wind_dir": int(m_wind.group(1)) if m_wind else None,
    }

    if len(boats) != 6:
        return None
    return {"boats": [boats[i] for i in sorted(boats)], "weather": weather}


# ---------------------------------------------------------------- オッズ

def parse_odds3t(html):
    """
    3連単オッズ120点。
    ページは「1着艇ごとに6ブロック × 20行」で、各行は左から1着=1..6の順にオッズが並ぶ。
    2着/3着の数字は rowspan で崩れるので読まず、並び順から組番を復元する。
    """
    soup = BeautifulSoup(html, "html.parser")
    target, best = None, 0
    for tbl in soup.find_all("table"):
        n = len(re.findall(r">\s*\d+\.\d\s*<", str(tbl)))
        if n > best:
            best, target = n, tbl
    if target is None:
        return None

    rows = []
    for tr in target.find_all("tr"):
        vals = []
        for td in tr.find_all("td"):
            t = td.get_text(strip=True)
            if re.fullmatch(r"\d+\.\d+", t):
                vals.append(float(t))
            elif t in ("欠場", "-", "―", ""):
                pass
        if len(vals) == 6:
            rows.append(vals)
    if len(rows) != 20:
        return None

    odds = [None] * 120
    for r, vals in enumerate(rows):
        for g, v in enumerate(vals):
            first = g + 1
            others = [b for b in range(1, 7) if b != first]
            second = others[r // 4]
            thirds = [b for b in others if b != second]
            third = thirds[r % 4]
            odds[COMBO_INDEX[f"{first}-{second}-{third}"]] = v
    if any(o is None for o in odds):
        return None
    return odds


# ---------------------------------------------------------------- 結果(場×日で1枚)

def parse_resultlist(html):
    """1場1日分(最大12レース)の 3連単組番・払戻金・返還有無・決まり手"""
    soup = BeautifulSoup(html, "html.parser")
    grade = "一般"
    title = ""
    for h in soup.find_all(["h2", "h3"]):
        t = h.get_text(strip=True)
        if t and "BOAT RACE" not in t:
            title = t
            break
    for src, dst in (("ＳＧ", "SG"), ("Ｇ１", "G1"), ("Ｇ２", "G2"), ("Ｇ３", "G3")):
        if title.startswith(src):
            grade = dst
            break

    out = {}
    for tr in soup.find_all("tr"):
        a = tr.find("a", href=re.compile(r"raceresult\?rno=(\d+)"))
        if not a:
            continue
        txt = tr.get_text(" ", strip=True)
        if "¥" not in txt:
            continue                      # 着順結果テーブル側の行は飛ばす
        rno = int(re.search(r"raceresult\?rno=(\d+)", a["href"]).group(1))
        if rno in out:
            continue
        m = re.search(r"(?<!\d)([1-6])\s*-\s*([1-6])\s*-\s*([1-6])(?!\d)", txt)
        if not m:
            continue                      # 不成立・中止
        pays = re.findall(r"¥\s*([\d,]+)", txt)
        out[rno] = {
            "rno": rno,
            "hit": f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
            "pay_3t": int(pays[0].replace(",", "")) if pays else None,
            "pay_2t": int(pays[1].replace(",", "")) if len(pays) > 1 else None,
            "henkan": ("返還" in txt),
            "grade": grade,
        }
    return out


# ---------------------------------------------------------------- 1レース取得

def fetch_race(session, hd, jcd, rno, res):
    try:
        rl = parse_racelist(get(session, "racelist", rno=rno, jcd=f"{jcd:02d}", hd=hd))
        bi = parse_beforeinfo(get(session, "beforeinfo", rno=rno, jcd=f"{jcd:02d}", hd=hd))
        od = parse_odds3t(get(session, "odds3t", rno=rno, jcd=f"{jcd:02d}", hd=hd))
    except Exception as e:
        return {"date": hd, "jcd": jcd, "rno": rno, "error": f"{type(e).__name__}: {e}"}

    if not rl or not bi or not od:
        miss = [n for n, v in (("racelist", rl), ("beforeinfo", bi), ("odds3t", od)) if not v]
        return {"date": hd, "jcd": jcd, "rno": rno, "error": "parse_failed:" + ",".join(miss)}

    bmap = {b["lane"]: b for b in bi["boats"]}
    entries = []
    for e in rl:
        b = bmap.get(e["lane"], {})
        e = dict(e)
        e.update({
            "weight": b.get("weight") or e.get("weight_list"),
            "tenji": b.get("tenji"),
            "tilt": b.get("tilt"),
            "course_in": b.get("course_in", e["lane"]),
            "tenji_st": b.get("tenji_st"),
            "maezuke": b.get("maezuke", 0),
            "course_diff": b.get("course_diff", 0),
        })
        entries.append(e)

    return {
        "date": hd, "jcd": jcd, "rno": rno,
        "grade": res.get("grade"),
        "entries": entries,
        "weather": bi["weather"],
        "odds": od,
        "hit": res.get("hit"),
        "pay_3t": res.get("pay_3t"),
        "pay_2t": res.get("pay_2t"),
        "henkan": res.get("henkan", False),
    }


# ---------------------------------------------------------------- 1日分

def fetch_day(session, hd, workers):
    # まず24場ぶんの結果一覧を引いて「開催していた場とレース」を確定させる
    day_res = {}
    with ThreadPoolExecutor(max_workers=min(workers, 12)) as ex:
        def one(jcd):
            try:
                return jcd, parse_resultlist(get(session, "resultlist", jcd=f"{jcd:02d}", hd=hd))
            except Exception:
                return jcd, {}
        for jcd, r in ex.map(one, range(1, 25)):
            if r:
                day_res[jcd] = r

    tasks = [(jcd, rno, r) for jcd, races in day_res.items() for rno, r in sorted(races.items())]
    if not tasks:
        return []

    races = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch_race, session, hd, jcd, rno, r) for jcd, rno, r in tasks]
        for f in futs:
            races.append(f.result())
    races.sort(key=lambda x: (x["jcd"], x["rno"]))
    return races


def save_day(outdir, hd, races):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{hd}.json.gz")
    payload = {"date": hd, "combos": COMBOS, "races": races}
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    return path


# ---------------------------------------------------------------- check

def do_check(session, hd, jcd, rno):
    print(f"=== check {hd} {jcd:02d}場 {rno}R ===\n")
    rlist = parse_resultlist(get(session, "resultlist", jcd=f"{jcd:02d}", hd=hd))
    print("[resultlist]", json.dumps(rlist.get(rno), ensure_ascii=False), "\n")
    rl = parse_racelist(get(session, "racelist", rno=rno, jcd=f"{jcd:02d}", hd=hd))
    print("[racelist]")
    for e in (rl or []):
        print("  ", json.dumps(e, ensure_ascii=False))
    print()
    bi = parse_beforeinfo(get(session, "beforeinfo", rno=rno, jcd=f"{jcd:02d}", hd=hd))
    print("[beforeinfo]")
    if bi:
        for b in bi["boats"]:
            print("  ", json.dumps(b, ensure_ascii=False))
        print("   weather:", json.dumps(bi["weather"], ensure_ascii=False))
    print()
    od = parse_odds3t(get(session, "odds3t", rno=rno, jcd=f"{jcd:02d}", hd=hd))
    if od:
        print(f"[odds3t] {len(od)}点  合計逆数(控除率の目安)={sum(1/o for o in od):.3f}")
        for c in ("1-2-3", "1-2-4", "1-3-2", "6-5-4"):
            print(f"   {c}: {od[COMBO_INDEX[c]]}")
        if rlist.get(rno, {}).get("hit"):
            hit = rlist[rno]["hit"]
            print(f"   的中 {hit}: オッズ {od[COMBO_INDEX[hit]]}  払戻 {rlist[rno]['pay_3t']}円"
                  f"  (オッズ×100 と一致すればOK)")
    else:
        print("[odds3t] 取得失敗")
    print("\n--- 上が正しければ本番実行してOK ---")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90, help="直近何日ぶん(既定90)")
    ap.add_argument("--start", help="YYYYMMDD")
    ap.add_argument("--end", help="YYYYMMDD")
    ap.add_argument("--out", default="raw")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--minutes", type=float, default=0, help="この分数で打ち切り(0=無制限)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-date"); ap.add_argument("--check-jcd", type=int, default=12)
    ap.add_argument("--check-rno", type=int, default=12)
    args = ap.parse_args()

    session = make_session(args.workers * 2)

    if args.check:
        hd = args.check_date or (date.today() - timedelta(days=7)).strftime("%Y%m%d")
        do_check(session, hd, args.check_jcd, args.check_rno)
        return

    if args.start and args.end:
        d0 = datetime.strptime(args.start, "%Y%m%d").date()
        d1 = datetime.strptime(args.end, "%Y%m%d").date()
    else:
        d1 = date.today() - timedelta(days=1)
        d0 = d1 - timedelta(days=args.days - 1)

    days = []
    d = d1
    while d >= d0:                       # 新しい日から取る(途中で止めても直近が揃う)
        days.append(d.strftime("%Y%m%d"))
        d -= timedelta(days=1)

    t0 = time.time()
    done = ok = 0
    for hd in days:
        path = os.path.join(args.out, f"{hd}.json.gz")
        if os.path.exists(path):
            continue
        if args.minutes and (time.time() - t0) / 60 > args.minutes:
            print(f"[stop] 時間切れ。次回実行で {hd} から再開します。", flush=True)
            break
        t = time.time()
        races = fetch_day(session, hd, args.workers)
        if not races:
            save_day(args.out, hd, [])   # 開催なしの日も置いて再取得を防ぐ
            print(f"{hd}: 開催なし", flush=True)
            continue
        good = [r for r in races if "error" not in r]
        save_day(args.out, hd, races)
        done += 1
        ok += len(good)
        print(f"{hd}: {len(good)}/{len(races)}レース  {time.time()-t:.0f}秒  累計{ok}レース",
              flush=True)

    print(f"\n完了: {done}日 / {ok}レース / {(time.time()-t0)/60:.1f}分")


if __name__ == "__main__":
    main()
