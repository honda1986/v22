#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kfile.py -- 競走成績ファイル(Kファイル)を取得して JSON に整形する

  http://www1.mbrace.or.jp/od2/K/YYYYMM/kYYMMDD.lzh
  LZH圧縮 / Shift-JIS の固定長テキスト

出力
  kfile/YYYYMMDD.json.gz
  {"date": "20260815",
   "races": [{"jcd":24, "rno":1, "grade":"予選", "dist":1800,
              "weather":"曇り", "wind_dir":"南西", "wind":1, "wave":1,
              "kimari":"逃げ", "hit":"1-5-2", "pay_3t":2730,
              "entries":[{"lane":1, "chaku":"01", "toban":5117,
                          "name":"冨名腰桃奈", "motor":47, "boat":67,
                          "tenji":6.91, "course":1, "st":0.01,
                          "st_flag":""}, ...]}, ...]}

  raw/ tokuten/ と同じ 1日1ファイルの形式。

使い方
  python kfile.py --from 20240301 --to 20260815
  python kfile.py --days 1                # 昨日ぶんだけ
  python kfile.py --probe                 # 接続確認のみ
"""

import argparse
import datetime as dt
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import requests

URL = "http://www1.mbrace.or.jp/od2/K/{ym}/k{ymd}.lzh"
OUTDIR = "kfile"
JST = dt.timezone(dt.timedelta(hours=9))

SESS = requests.Session()
SESS.headers["User-Agent"] = (
    "Mozilla/5.0 (compatible; boatrace-research/1.0)")


# ================================================================
# LZH 展開。lhafile が使えなければ lha コマンドに落とす
# ================================================================
def _lhafile():
    try:
        from lhafile import LhaFile
        return LhaFile
    except ImportError:
        return None


LHAFILE = _lhafile()
HAVE_LHA = shutil.which("lha") is not None


def unlzh(blob):
    """LZH のバイト列から、中身のテキスト(bytes)を返す"""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "a.lzh")
        with open(p, "wb") as f:
            f.write(blob)
        if LHAFILE is not None:
            try:
                a = LHAFILE(p)
                return a.read(a.infolist()[0].filename)
            except Exception:
                pass
        if HAVE_LHA:
            subprocess.run(["lha", "-xw=" + td, p],
                           capture_output=True, check=False)
            for fn in os.listdir(td):
                if fn.lower().endswith(".txt"):
                    with open(os.path.join(td, fn), "rb") as f:
                        return f.read()
    return None


# ================================================================
# パーサー(実データで検算済み)
# ================================================================
RE_BGN = re.compile(r"^(\d{2})KBGN")
RE_PAY = re.compile(r"^\s*(\d{1,2})R\s+([1-6]-[1-6]-[1-6])\s+(\d+)")
RE_HEAD = re.compile(r"^\s*(\d{1,2})R\s{2,}(\S+)?.*?H(\d+)m")
RE_COL = re.compile(r"ﾚｰｽﾀｲﾑ(.*)$")
RE_ROW = re.compile(r"^\s*(\S{1,3})\s+([1-6])\s+(\d{4})\s(.*)$")
RE_WIND = re.compile(r"風\s*(\S+?)\s*(\d+)m")
RE_WAVE = re.compile(r"波\s*(\d+)cm")
RE_TENKI = re.compile(r"H\d+m\s+(\S+)")


def to_f(t):
    try:
        return float(t)
    except (TypeError, ValueError):
        return None


def to_i(t):
    return int(t) if t and t.isdigit() else None


def parse_st(t):
    """ST。F(フライング)は負値、L(出遅れ)はフラグだけ残す"""
    t = (t or "").strip()
    if not t or set(t) <= {".", " "}:
        return None, ""
    flag = ""
    if t[0] in "FLＦＬ":
        flag = "F" if t[0] in "FＦ" else "L"
        t = t[1:].strip()
    v = to_f(t)
    if v is None:
        return None, flag
    return (-v if flag == "F" else v), flag


def split_name(rest):
    """選手名は全角。最後の全角文字までが名前、その後がASCIIの数値列"""
    idx = -1
    for i, ch in enumerate(rest):
        if ord(ch) > 0x7F:
            idx = i
    if idx < 0:
        return "", rest
    return rest[:idx + 1].replace("\u3000", "").strip(), rest[idx + 1:]


def parse_k(text, date):
    races = {}
    pay = {}
    jcd = rno = None
    cur = None

    for line in text.split("\n"):
        m = RE_BGN.match(line)
        if m:
            jcd, rno, cur = int(m.group(1)), None, None
            continue
        if jcd is None:
            continue

        m = RE_PAY.match(line)
        if m and line.strip().startswith(m.group(1)):
            pay[(jcd, int(m.group(1)))] = (m.group(2), int(m.group(3)))
            continue

        m = RE_HEAD.match(line)
        if m:
            rno = int(m.group(1))
            w = RE_TENKI.search(line)
            wd = RE_WIND.search(line)
            wv = RE_WAVE.search(line)
            cur = {
                "jcd": jcd, "rno": rno,
                "grade": (m.group(2) or "").replace("\u3000", "").strip()
                         or None,
                "dist": int(m.group(3)),
                "weather": w.group(1) if w else None,
                "wind_dir": wd.group(1) if wd else None,
                "wind": int(wd.group(2)) if wd else None,
                "wave": int(wv.group(1)) if wv else None,
                "kimari": None, "entries": [],
            }
            races[(jcd, rno)] = cur
            continue

        if "ﾚｰｽﾀｲﾑ" in line:
            if cur is not None:
                m = RE_COL.search(line)
                k = (m.group(1).replace("\u3000", "").strip()
                     if m else "")
                cur["kimari"] = k or None
            continue

        if cur is None:
            continue
        m = RE_ROW.match(line)
        if not m:
            continue
        chaku, lane, toban, rest = m.groups()
        name, tail = split_name(rest)
        if not name:
            continue
        tk = tail.split()
        if len(tk) < 4:
            continue
        st, flag = parse_st(tk[4]) if len(tk) > 4 else (None, "")
        cur["entries"].append({
            "lane": int(lane), "chaku": chaku.strip(),
            "toban": int(toban), "name": name,
            "motor": to_i(tk[0]), "boat": to_i(tk[1]),
            "tenji": to_f(tk[2]), "course": to_i(tk[3]),
            "st": st, "st_flag": flag,
        })

    out = []
    for key in sorted(races):
        r = races[key]
        h, p = pay.get(key, (None, None))
        r["hit"], r["pay_3t"] = h, p
        out.append(r)
    return out


# ================================================================
# 取得
# ================================================================
def fetch(date, retry=3):
    url = URL.format(ym=date[:6], ymd=date[2:])
    for i in range(retry):
        try:
            r = SESS.get(url, timeout=40)
        except requests.RequestException as e:
            if i == retry - 1:
                return None, f"{type(e).__name__}: {e}"
            time.sleep(3 * (i + 1))
            continue
        if r.status_code == 404:
            return None, "404(開催なし)"
        if r.status_code != 200:
            if i == retry - 1:
                return None, f"HTTP {r.status_code}"
            time.sleep(3 * (i + 1))
            continue
        blob = unlzh(r.content)
        if blob is None:
            return None, "LZH展開に失敗"
        return blob.decode("cp932", errors="replace"), None
    return None, "リトライ上限"


def save(date, races, outdir):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, f"{date}.json.gz")
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump({"date": date, "races": races}, f,
                  ensure_ascii=False, separators=(",", ":"))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="dfrom")
    ap.add_argument("--to", dest="dto")
    ap.add_argument("--days", type=int,
                    help="今日から遡って何日ぶんか(1なら昨日)")
    ap.add_argument("--out", default=OUTDIR)
    ap.add_argument("--sleep", type=float, default=1.5)
    ap.add_argument("--probe", action="store_true",
                    help="接続確認だけして終わる")
    ap.add_argument("--force", action="store_true",
                    help="既存ファイルも取り直す")
    a = ap.parse_args()

    print(f"LZH展開: lhafile={LHAFILE is not None} lha={HAVE_LHA}")
    if LHAFILE is None and not HAVE_LHA:
        print("どちらも使えません。lhasa か lhafile を入れてください。")
        return 1

    if a.probe:
        d = (dt.datetime.now(JST) - dt.timedelta(days=1)).strftime("%Y%m%d")
        print(f"接続確認 {d} ...")
        txt, err = fetch(d)
        if err:
            print(f"  失敗: {err}")
            return 1
        races = parse_k(txt, d)
        ent = sum(len(r["entries"]) for r in races)
        tenji = sum(1 for r in races for e in r["entries"]
                    if e["tenji"] is not None)
        st = sum(1 for r in races for e in r["entries"]
                 if e["st"] is not None)
        print(f"  成功: {len(txt):,}文字 / {len(races)}レース / {ent}艇")
        print(f"  展示 {tenji}/{ent}  ST {st}/{ent}  "
              f"決まり手 {sum(1 for r in races if r['kimari'])}/{len(races)}")
        return 0

    today = dt.datetime.now(JST).date()
    if a.days:
        dates = [(today - dt.timedelta(days=i)).strftime("%Y%m%d")
                 for i in range(1, a.days + 1)][::-1]
    else:
        if not (a.dfrom and a.dto):
            print("--from と --to、または --days を指定してください")
            return 1
        s = dt.date(int(a.dfrom[:4]), int(a.dfrom[4:6]), int(a.dfrom[6:]))
        e = dt.date(int(a.dto[:4]), int(a.dto[4:6]), int(a.dto[6:]))
        dates = [(s + dt.timedelta(days=i)).strftime("%Y%m%d")
                 for i in range((e - s).days + 1)]

    print(f"対象 {len(dates)}日 ({dates[0]}〜{dates[-1]})")
    ok = skip = ng = 0
    errs = []
    t0 = time.time()
    for i, d in enumerate(dates, 1):
        p = os.path.join(a.out, f"{d}.json.gz")
        if os.path.exists(p) and not a.force:
            skip += 1
            continue
        txt, err = fetch(d)
        if err:
            ng += 1
            errs.append((d, err))
        else:
            races = parse_k(txt, d)
            save(d, races, a.out)
            ok += 1
        if i % 25 == 0 or i == len(dates):
            print(f"  {i}/{len(dates)}  新規{ok} 既存{skip} 失敗{ng}  "
                  f"{time.time()-t0:.0f}秒", flush=True)
        time.sleep(a.sleep)

    print(f"\n完了 新規{ok} 既存{skip} 失敗{ng} / {time.time()-t0:.0f}秒")
    if errs:
        print("失敗した日(先頭10件):")
        for d, e in errs[:10]:
            print(f"  {d}  {e}")
        real = [x for x in errs if "404" not in x[1]]
        if real:
            print(f"404以外の失敗 {len(real)}件 ← 要確認")
    return 0


if __name__ == "__main__":
    sys.exit(main())
