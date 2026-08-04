#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ohdokei.py -- 「大時計」アプリのデータを作る

■ 買い方 (132,219レースの実測で決定)
    オッズ6倍以下の組が「ちょうど2点」あるレースだけ買う。その2点を買う。

    2点  8,071レース  回収率 86.7% ± 1.3%
         年度別 84.1 / 87.2 / 90.6 / 89.8%  (4年度とも土台84.0%を上回る)
    1点 21,233レース  82.3%
    3点    407レース  77.2%

    買えるのは全体の6.1%。1日8〜9レース。1レース200円。

  予想はしない。自作モデルは市場を超えないと13.5万レースで確認済み。
  市場が2つの決着に金を集中させているレースだけを拾う、という買い方。

■ 出力
  ohdokei/data.json

■ 通知 (任意)
  環境変数 NTFY_TOPIC を設定すると、確定した買い目を ntfy で通知する。
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

# 132,219レースの実測。オッズ帯ごとの回収率。
CALIB = [(1, 3.5, 0.824), (3.5, 4, 0.890), (4, 5, 0.839), (5, 6, 0.841),
         (6, 8, 0.809), (8, 10, 0.785), (10, 13, 0.769), (13, 16, 0.789),
         (16, 20, 0.800), (20, 30, 0.765), (30, 50, 0.764), (50, 100, 0.731),
         (100, 1e9, 0.481)]

MAX_ODDS = 6.0        # この倍率以下を数える
NEED = 2              # ちょうど何点あれば買うか

# 公式サイトの「締切予定時刻」は本場の締切。
# ネット投票(テレボート)は本場より2〜3分早く締め切られるので、
# 表示時刻から NET_LEAD 分を引いた時刻を、こちらの実質的な締切として扱う。
NET_LEAD = 3

# 1レースにつきオッズは1回だけ取る。実質締切の7〜22分前の窓で判定する。
#   5分間隔なら取りこぼし0%、10分に遅れても1%(シミュレーション実測)。
#   窓を逃したレースだけ3〜7分前に拾い直す。
WIN_HI = 22
WIN_LO = 7
LATE_LO = 3

# 締切予定時刻は遅延する。オッズを取るたびに、そのページに載っている
# 最新の予定表で上書きする(追加のリクエストは要らない)。


def deadline(now, hhmm):
    """表示の締切予定時刻から、ネット投票の実質締切を返す"""
    hh, mm = (int(x) for x in hhmm.split(":"))
    d = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return d - timedelta(minutes=NET_LEAD)


def band_roi(o):
    for lo, hi, v in CALIB:
        if lo <= o < hi:
            return v
    return 0.481


# ---------------------------------------------------------------- 取得
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
    soup = BeautifulSoup(html, "html.parser")
    for tr in soup.find_all("tr"):
        if "締切予定時刻" not in tr.get_text():
            continue
        t = re.findall(r"\b(\d{1,2}:\d{2})\b", tr.get_text(" "))
        if len(t) >= 12:
            return t[:12]
    return None


def parse_resultlist(html):
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
def build_picks(odds):
    """6倍以下がちょうど2点のときだけ買う。それ以外は見送り。"""
    sel = [i for i in range(120) if odds[i] <= MAX_ODDS]
    if len(sel) != NEED:
        return None, len(sel)
    sel.sort(key=lambda i: odds[i])
    pts, hit, roi = [], 0.0, 0.0
    for i in sel:
        o = odds[i]
        r = band_roi(o)
        p = r / o
        pts.append({"c": COMBOS[i], "o": round(o, 1), "p": round(p, 4)})
        hit += p
        roi += r
    pays = [p["o"] * 100 for p in pts]
    return {"points": pts, "hit_rate": round(hit, 4),
            "exp_roi": round(roi / len(pts), 4), "stake": len(pts) * 100,
            "pay_lo": int(min(pays)), "pay_hi": int(max(pays))}, len(sel)


# ---------------------------------------------------------------- 通知
def notify(topic, rec, app_url, urgent=False):
    if not topic:
        return False
    buys = "  ".join(f"{p['c']} {p['o']:.1f}倍" for p in rec["points"])
    body = (f"{buys}\n"
            f"本場締切 {rec['close']}\n"
            f"的中率 {rec['hit_rate']*100:.0f}%  "
            f"期待回収率 {rec['exp_roi']*100:.0f}%\n"
            f"{rec['stake']}円 → 平均 {int(rec['stake']*rec['exp_roi'])}円")
    payload = {"topic": topic,
               "title": ("まもなく締切 " if urgent else "") +
                        f"{rec['venue']} {rec['rno']}R  "
                        f"ネット{rec.get('net', rec['close'])}締切",
               "message": body, "priority": 5 if urgent else 4,
               "tags": ["rotating_light" if urgent else "speedboat"]}
    if app_url:
        payload["click"] = app_url
    try:
        r = requests.post("https://ntfy.sh", json=payload, timeout=10)
        return r.status_code < 300
    except requests.RequestException:
        return False


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ohdokei/data.json")
    ap.add_argument("--scan", type=int, default=20, help="1回で調べる上限")
    ap.add_argument("--app-url", default="", help="通知タップで開くURL")
    ap.add_argument("--test-notify", action="store_true",
                    help="ダミーの買い目を1件だけ通知して終了する(設定確認用)")
    args = ap.parse_args()

    now = datetime.now(JST)
    today = now.strftime("%Y%m%d")
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    sess = requests.Session()
    sess.headers.update(UA)

    if args.test_notify:
        if not topic:
            print("★ NTFY_TOPIC が設定されていません。")
            print("  GitHub → Settings → Secrets and variables → Actions で")
            print("  NTFY_TOPIC という名前のシークレットを登録してください。")
            raise SystemExit(1)
        print(f"トピック名の長さ {len(topic)}文字 (中身は伏せます)")
        dummy = {"venue": "テスト", "rno": 7,
                 "close": (now + timedelta(minutes=16)).strftime("%H:%M"),
                 "points": [{"c": "1-2-3", "o": 3.4}, {"c": "1-2-4", "o": 5.1}],
                 "hit_rate": 0.41, "exp_roi": 0.83, "stake": 200}
        ok = notify(topic, dummy, args.app_url)
        print("通知を送りました。スマホを確認してください。" if ok
              else "★ 送信に失敗しました。トピック名を確認してください。")
        raise SystemExit(0 if ok else 1)

    data = {"date": today, "rule": {"max_odds": MAX_ODDS, "need": NEED},
            "venues": {}, "races": {}}
    if os.path.exists(args.out):
        try:
            old = json.load(open(args.out, encoding="utf-8"))
            if old.get("date") == today:
                data = old
                data["rule"] = {"max_odds": MAX_ODDS, "need": NEED}
        except Exception:
            pass

    # 1) 今日の開催と締切時刻
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

    # 2) 判定するレースを選ぶ (1レース1回)
    sched, targets, late = [], [], []
    for jcd_s, times in data["venues"].items():
        jcd = int(jcd_s)
        for rno, t in enumerate(times, 1):
            key = f"{today}-{jcd:02d}-{rno}"
            close = deadline(now, t)
            mins = (close - now).total_seconds() / 60
            sched.append({"key": key, "jcd": jcd, "venue": VENUE[jcd],
                          "rno": rno, "close": t,
                          "net": close.strftime("%H:%M")})
            if key in data["races"]:
                continue
            if WIN_LO <= mins <= WIN_HI:
                targets.append((mins, jcd, rno, key, t, False))
            elif LATE_LO <= mins < WIN_LO:
                late.append((mins, jcd, rno, key, t, True))
    data["schedule"] = sched
    targets.sort()
    late.sort()
    queue = (late + targets)[:args.scan]      # 拾い直しを最優先
    print(f"\n判定 {len(targets)}件 / 拾い直し {len(late)}件 → 処理 {len(queue)}件")
    if len(targets) + len(late) > len(queue):
        print(f"  ★{len(targets)+len(late)-len(queue)}件あふれました。scanを増やしてください。")

    sent = 0
    for mins, jcd, rno, key, t, is_late in queue:
        try:
            html = get(sess, "odds3t", rno=rno, jcd=f"{jcd:02d}", hd=today)
            odds = parse_odds3t(html)
        except Exception as e:
            print(f"  {VENUE[jcd]}{rno}R 取得失敗 {type(e).__name__}")
            continue
        if not odds:
            continue

        # 遅延に追随する: このページの予定表で上書き
        sc = parse_schedule(html)
        if sc and sc != data["venues"].get(str(jcd)):
            before = data["venues"].get(str(jcd), [])
            data["venues"][str(jcd)] = sc
            if before:
                ch = [f"{i+1}R {a}→{b}" for i, (a, b) in
                      enumerate(zip(before, sc)) if a != b]
                if ch:
                    print(f"  ※{VENUE[jcd]} 締切時刻が変わりました: "
                          + " ".join(ch[:4]), flush=True)
            t = sc[rno - 1]
            mins = (deadline(now, t) - now).total_seconds() / 60
            if mins < 1:
                print(f"  {VENUE[jcd]}{rno}R は締切済みでした。見送ります。")
                continue

        rec, n_cheap = build_picks(odds)
        base = {"jcd": jcd, "venue": VENUE[jcd], "rno": rno, "close": t,
                "net": deadline(now, t).strftime("%H:%M"),
                "odds_at": now.strftime("%H:%M"), "mins_at": round(mins),
                "late": is_late, "n_cheap": n_cheap,
                "notified": False, "final": None, "result": None}

        if rec is None:
            base["status"] = "skip"
            data["races"][key] = base
            print(f"  {VENUE[jcd]}{rno}R {t}締切  見送り(6倍以下が{n_cheap}点)",
                  flush=True)
            continue

        rec.update(base)
        rec["status"] = "buy"
        data["races"][key] = rec
        print(f"  {VENUE[jcd]}{rno}R {t}締切 {round(mins)}分前"
              f"{'(拾い直し)' if is_late else ''}  "
              f"{rec['points'][0]['c']} {rec['points'][0]['o']}倍 / "
              f"{rec['points'][1]['c']} {rec['points'][1]['o']}倍  "
              f"的中率{rec['hit_rate']*100:.0f}%", flush=True)

        if notify(topic, rec, args.app_url, is_late):
            rec["notified"] = True
            sent += 1
    if topic:
        print(f"通知 {sent}件")

    # 2.5) 締切後にもう一度オッズを取り、判定時とのズレを測る
    chk = [(k, r) for k, r in data["races"].items()
           if r.get("status") == "buy" and not r.get("final")]
    done_chk = 0
    for key, r in chk:
        if done_chk >= 6:
            break
        if (now - deadline(now, r["close"])).total_seconds() / 60 < 6:
            continue
        try:
            odds = parse_odds3t(get(sess, "odds3t", rno=r["rno"],
                                    jcd=f"{r['jcd']:02d}", hd=today))
        except Exception:
            continue
        if not odds:
            continue
        fo = [round(odds[CIX[p["c"]]], 1) for p in r["points"]]
        n_cheap_final = sum(1 for o in odds if o <= MAX_ODDS)
        r["final"] = {"odds": fo, "n_cheap": n_cheap_final,
                      "still": n_cheap_final == NEED}
        done_chk += 1
        drift = [f"{p['o']}→{f}" for p, f in zip(r["points"], fo)]
        print(f"  締切時 {r['venue']}{r['rno']}R  {' / '.join(drift)}  "
              f"6倍以下{n_cheap_final}点 {'条件維持' if r['final']['still'] else '条件外れ'}")

    # 3) 結果を入れる
    need = {}
    for key, r in data["races"].items():
        if r.get("result") or r.get("status") != "buy":
            continue
        if (now - deadline(now, r["close"])).total_seconds() / 60 > 15:
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
            print(f"  結果 {VENUE[jcd]}{rno}R {rr['hit']} "
                  f"{'的中' if rr['hit'] in picks else '不的中'}")

    data["updated"] = now.strftime("%Y-%m-%d %H:%M")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"\n{args.out} を更新 (レース{len(data['races'])}件)")


if __name__ == "__main__":
    main()
