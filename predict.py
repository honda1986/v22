# -*- coding: utf-8 -*-
"""
predict.py — 本日の全レースを予想して races.json に保存する(GitHub Actions で実行)

v22_app.py(Streamlit版)から「取得・特徴量・モデル推論・買い目選定」だけを取り出したもの。
画面表示は index.html(GitHub Pages)が races.json を読んで行うため、ここでは出力に専念する。

依存: lgb_p1_v22.txt / lgb_p2_v22.txt / lgb_p3_v22.txt と各 *_features.json
使い方: python predict.py [YYYYMMDD]   (省略時は本日 JST)
"""
import json
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
import lightgbm as lgb

JST = timezone(timedelta(hours=+9), "JST")
DIR = os.path.dirname(os.path.abspath(__file__))

# ---- 買い目の選び方(Streamlit版のサイドバー設定に相当。ここで固定する) ----
EV_MIN = 1.10        # 期待値(確率×オッズ)がこの値以上の買い目だけ採用
TOP_N_PROB = 12      # まず確率上位この点数に絞る
MAX_POINTS = 8       # 1レースの上限点数
MIN_POINTS = 1       # これ未満なら「見送り」扱い

JCD_NAME = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村",
}

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
sess = requests.Session()
sess.headers.update(UA)
try:
    from urllib3.util.retry import Retry
except Exception:
    from requests.packages.urllib3.util.retry import Retry  # type: ignore


def _retry():
    kw = dict(total=3, connect=3, read=3, status=3, backoff_factor=0.6,
              status_forcelist=(429, 500, 502, 503, 504),
              respect_retry_after_header=True, raise_on_status=False)
    try:
        return Retry(allowed_methods=frozenset(["GET"]), **kw)
    except TypeError:
        return Retry(method_whitelist=frozenset(["GET"]), **kw)


_ad = HTTPAdapter(max_retries=_retry(), pool_connections=16, pool_maxsize=16)
sess.mount("https://", _ad)
sess.mount("http://", _ad)


# ============================================================
# モデル読み込み
# ============================================================
def _load_model(name):
    p = os.path.join(DIR, name)
    return lgb.Booster(model_file=p) if os.path.exists(p) else None


def _load_features(name):
    p = os.path.join(DIR, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


m_p1 = _load_model("lgb_p1_v22.txt")
m_p2 = _load_model("lgb_p2_v22.txt")
m_p3 = _load_model("lgb_p3_v22.txt")
features_p1 = _load_features("lgb_p1_v22_features.json")
features_p2 = _load_features("lgb_p2_v22_features.json")
features_p3 = _load_features("lgb_p3_v22_features.json")


# ============================================================
# 特徴量生成(学習時と同じロジック)
# ============================================================
def make_race_features(racer_rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(racer_rows).sort_values("lane").reset_index(drop=True)
    df["win_dev"] = df["n_win"] - df["n_win"].mean()
    df["motor_dev"] = df["m_2ren"] - df["m_2ren"].mean()
    df["st_dev"] = df["avg_st"].mean() - df["avg_st"]
    df["tenji_dev"] = df["tenji"].mean() - df["tenji"]
    df["win_rank"] = df["n_win"].rank(ascending=False, method="min").astype(int)
    df["motor_rank"] = df["m_2ren"].rank(ascending=False, method="min").astype(int)
    df["st_rank"] = df["avg_st"].rank(ascending=True, method="min").astype(int)
    df["tenji_rank"] = df["tenji"].rank(ascending=True, method="min").astype(int)
    df["maezuke"] = (df["lane"] != df["course_in"]).astype(int)
    df["course_diff"] = df["course_in"] - df["lane"]
    for col in ["avg_st", "n_win", "tenji"]:
        for direction, shift in [("in", 1), ("out", -1)]:
            vals = []
            for i in range(len(df)):
                j = i - shift
                vals.append(df.loc[i, col] - df.loc[j, col] if 0 <= j < len(df) else 0.0)
            df[f"{col}_diff_{direction}"] = vals
    return df


def predict_combo_probs(features_df: pd.DataFrame, race_jcd: int) -> Dict[str, float]:
    """6艇の特徴量から120点の3連単確率を返す。"""
    if not (m_p1 and m_p2 and m_p3):
        return {}
    df = features_df.copy()
    df["jcd"] = race_jcd
    base_cols = features_p1

    p1 = {}
    for _, row in df.iterrows():
        x = row[base_cols].values.reshape(1, -1).astype(float)
        p1[int(row["lane"])] = float(m_p1.predict(x)[0])
    s = sum(p1.values())
    if s > 0:
        p1 = {k: v / s for k, v in p1.items()}

    combos = {}
    for w1 in range(1, 7):
        w1_row = df[df["lane"] == w1].iloc[0]
        p2_raw = {}
        for cand in range(1, 7):
            if cand == w1:
                continue
            cand_row = df[df["lane"] == cand].iloc[0]
            feat = {f: cand_row[f] for f in base_cols if f in cand_row.index}
            for f in ["lane", "cls_val", "avg_st", "n_win", "m_2ren", "tenji", "course_in", "maezuke"]:
                feat[f"w1_{f}"] = w1_row[f]
            feat["w1_lane_diff"] = cand_row["lane"] - w1_row["lane"]
            feat["w1_course_diff"] = cand_row["course_in"] - w1_row["course_in"]
            x = np.array([feat.get(c, 0.0) for c in features_p2]).reshape(1, -1).astype(float)
            p2_raw[cand] = float(m_p2.predict(x)[0])
        s2 = sum(p2_raw.values())
        p2 = {k: (v / s2 if s2 > 0 else 0) for k, v in p2_raw.items()}

        for w2 in range(1, 7):
            if w2 == w1:
                continue
            w2_row = df[df["lane"] == w2].iloc[0]
            p3_raw = {}
            for cand in range(1, 7):
                if cand in (w1, w2):
                    continue
                cand_row = df[df["lane"] == cand].iloc[0]
                feat = {f: cand_row[f] for f in base_cols if f in cand_row.index}
                for f in ["lane", "cls_val", "avg_st", "n_win", "m_2ren", "tenji", "course_in", "maezuke"]:
                    feat[f"w1_{f}"] = w1_row[f]
                feat["w1_lane_diff"] = cand_row["lane"] - w1_row["lane"]
                feat["w1_course_diff"] = cand_row["course_in"] - w1_row["course_in"]
                for f in ["lane", "cls_val", "avg_st", "n_win", "m_2ren", "tenji", "course_in"]:
                    feat[f"w2_{f}"] = w2_row[f]
                feat["w2_lane_diff"] = cand_row["lane"] - w2_row["lane"]
                x = np.array([feat.get(c, 0.0) for c in features_p3]).reshape(1, -1).astype(float)
                p3_raw[cand] = float(m_p3.predict(x)[0])
            s3 = sum(p3_raw.values())
            p3 = {k: (v / s3 if s3 > 0 else 0) for k, v in p3_raw.items()}
            for w3 in range(1, 7):
                if w3 in (w1, w2):
                    continue
                combos[f"{w1}-{w2}-{w3}"] = p1[w1] * p2[w2] * p3[w3]
    return combos


def boat_eval(combo_probs: Dict[str, float]) -> List[Dict]:
    """120点の確率から各艇の 1着率/2連対率/3着内率/評価 を集計。"""
    p1 = {i: 0.0 for i in range(1, 7)}
    p2 = {i: 0.0 for i in range(1, 7)}
    p3 = {i: 0.0 for i in range(1, 7)}
    for combo, p in combo_probs.items():
        try:
            a, b, c = (int(x) for x in combo.split("-"))
        except ValueError:
            continue
        p1[a] += p
        p2[b] += p
        p3[c] += p
    rows = []
    for lane in range(1, 7):
        w, s2, s3 = p1[lane], p2[lane], p3[lane]
        rows.append({
            "lane": lane,
            "win": round(w * 100, 1),
            "ren2": round((w + s2) * 100, 1),
            "ren3": round((w + s2 + s3) * 100, 1),
            "score": round((3 * w + 2 * s2 + s3) / 3 * 100, 1),
        })
    return sorted(rows, key=lambda r: -r["score"])


# ============================================================
# 取得
# ============================================================
RE_CLS = re.compile(r"([A12B]{2})")
RE_WEIGHT = re.compile(r"(\d+)kg", re.IGNORECASE)
RE_AGE = re.compile(r"\((\d{2})\)")
CLS_MAP = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
RE_HHMM = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").replace(" ", "").replace("\u3000", "")


def _soup(url, timeout=12, min_len=3000, tries=1):
    for _ in range(max(1, tries)):
        try:
            r = sess.get(url, timeout=timeout)
            r.encoding = r.apparent_encoding
            if r.status_code == 200 and len(r.text) >= min_len:
                return BeautifulSoup(r.text, "html.parser")
        except requests.RequestException:
            pass
    return None


def _lane_from_class(td):
    div = td.find("div", class_=lambda c: c and "ng1r" in c)
    if not div:
        return None
    for cls in div.get("class", []):
        m = re.match(r"ng1r(\d)$", cls)
        if m:
            return int(m.group(1))
    return None


def fetch_close_times(date: str, jcd: int) -> Dict[int, str]:
    """その場・その日の各レースの締切予定時刻 {rno: 'HH:MM'}。開催が無ければ空。"""
    url = f"https://www.boatrace.jp/owpc/pc/race/raceindex?jcd={jcd:02d}&hd={date}"
    soup = _soup(url, min_len=2000)
    out: Dict[int, str] = {}
    if soup is None:
        return out
    nfkc = lambda s: unicodedata.normalize("NFKC", s)
    for a in soup.find_all("a", href=True):
        m = re.search(r"[?&]rno=(\d{1,2})(?!\d)", a["href"])
        if not m:
            continue
        rno = int(m.group(1))
        if not (1 <= rno <= 12) or rno in out:
            continue
        row = a.find_parent("tr") or a.parent
        if row is None:
            continue
        tm = RE_HHMM.search(nfkc(row.get_text(" ")))
        if tm:
            h, mi = int(tm.group(1)), int(tm.group(2))
            if 6 <= h <= 23 and 0 <= mi <= 59:
                out[rno] = f"{h:02d}:{mi:02d}"
    if len(out) >= 6:
        return out
    text = nfkc(soup.get_text(" "))
    for m in re.finditer(r"(?<!\d)(\d{1,2})\s*R(?![0-9A-Za-z])[^0-9]{0,40}?(\d{1,2}):(\d{2})(?!\d)", text):
        rno, h, mi = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= rno <= 12 and 6 <= h <= 23 and rno not in out:
            out[rno] = f"{h:02d}:{mi:02d}"
    return out


def fetch_racecard(date: str, jcd: int, rno: int):
    """kyotei.fun の結合ページから選手データを取得。"""
    url = f"https://info.kyotei.fun/info-{date}-{jcd:02d}-{rno}.html"
    try:
        r = sess.get(url, timeout=15)
        r.encoding = r.apparent_encoding
        if r.status_code != 200 or len(r.text) < 5000:
            return None
    except requests.RequestException:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    base = {i + 1: {"lane": i + 1, "age": 30, "cls_val": 1, "weight": 50, "f_count": 0,
                    "avg_st": 0.17, "n_win": 0.0, "n_2ren": 0.0, "l_win": 0.0, "l_2ren": 0.0,
                    "m_2ren": 0.0, "b_2ren": 0.0, "tenji": 6.80, "course_in": i + 1,
                    "name": "", "tenji_st": ""} for i in range(6)}
    label = ""
    for tr in soup.find_all("tr"):
        tds = tr.find_all(["td", "th"])
        if not tds:
            continue
        if len(tds) >= 7:
            label = tds[0].get_text(strip=True).replace("\n", "").replace(" ", "").replace("\u3000", "")
            data = tds[-6:]
        elif len(tds) == 6 and label:
            data = tds
        else:
            label = ""
            continue
        for i in range(6):
            td = data[i]
            txt = td.get_text(" ", strip=True).replace(" ", "").replace("\u3000", "").replace("\n", "")
            lane = i + 1
            if "選手名" in label:
                nm = re.sub(r"\(.*", "", txt).strip()
                if nm:
                    base[lane]["name"] = nm[:8]
                m = RE_AGE.search(txt)
                if m:
                    base[lane]["age"] = int(m.group(1))
            elif "選手情報" in label or "支部" in label:
                mc = RE_CLS.search(txt)
                if mc:
                    base[lane]["cls_val"] = CLS_MAP.get(mc.group(1), 1)
                mw = RE_WEIGHT.search(txt)
                if mw:
                    base[lane]["weight"] = int(mw.group(1))
            elif "級過去2期" in label:
                mc = RE_CLS.search(txt)
                if mc:
                    base[lane]["cls_val"] = CLS_MAP.get(mc.group(1), 1)
            elif "全国" in label and "勝率" in label:
                m2 = re.search(r"^([\d\.]+)", txt)
                mw = re.search(r"\(([\d\.]+)\)", txt)
                if m2:
                    v = float(m2.group(1))
                    base[lane]["n_2ren"] = v / 100.0 if v > 1.0 else v
                if mw:
                    base[lane]["n_win"] = float(mw.group(1))
            elif "当地" in label and "勝率" in label:
                m2 = re.search(r"^([\d\.]+)", txt)
                mw = re.search(r"\(([\d\.]+)\)", txt)
                if m2:
                    v = float(m2.group(1))
                    base[lane]["l_2ren"] = v / 100.0 if v > 1.0 else v
                if mw:
                    base[lane]["l_win"] = float(mw.group(1))
            elif "モータ" in label and "2連率" in label:
                m = re.search(r"^([\d\.]+)", txt)
                if m:
                    v = float(m.group(1))
                    base[lane]["m_2ren"] = v / 100.0 if v > 1.0 else v
            elif "ボート" in label and "2連率" in label:
                m = re.search(r"^([\d\.]+)", txt)
                if m:
                    v = float(m.group(1))
                    base[lane]["b_2ren"] = v / 100.0 if v > 1.0 else v
            elif "平均ST" in label:
                try:
                    base[lane]["avg_st"] = float(txt)
                except ValueError:
                    pass
            elif "フライング" in label:
                try:
                    base[lane]["f_count"] = int(txt)
                except ValueError:
                    pass
            elif label == "展示":
                try:
                    base[lane]["tenji"] = float(txt)
                except ValueError:
                    pass
            elif label == "コースIN":
                c = _lane_from_class(td)
                if c:
                    base[lane]["course_in"] = c
    return [base[i + 1] for i in range(6)]


def _closest_class(node, cls):
    p = node
    while p is not None and getattr(p, "get", None) is not None:
        if cls in (p.get("class") or []):
            return p
        p = p.parent
    return None


def _start_timing(row):
    if row is None:
        return None
    txt = None
    for el in row.find_all(True):
        if any("Time" in c for c in (el.get("class") or [])):
            t = _norm(el.get_text())
            if t:
                txt = t
                break
    if not txt:
        txt = _norm(row.get_text())
    m = re.search(r"([FL])?\.(\d{2})", txt)
    if m:
        return f"{m.group(1) or ''}.{m.group(2)}"
    return "F" if "F" in txt else ("L" if "L" in txt else None)


def fetch_beforeinfo(date: str, jcd: int, rno: int, tries: int = 2) -> Dict:
    """公式の直前情報から展示タイム・進入コース・展示ST・体重・気象。"""
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd:02d}&hd={date}"
    best = {"tenji": {}, "course_in": {}, "exhibition_st": {}, "weight": {}, "weather": {}}
    for attempt in range(max(1, tries)):
        soup = _soup(url)
        if soup is None:
            continue
        out = {"tenji": {}, "course_in": {}, "exhibition_st": {}, "weight": {}, "weather": {}}
        card = soup.find("table", class_=lambda c: c and "is-w748" in c)
        for tb in (card.find_all("tbody") if card else []):
            tr = tb.find("tr")
            if not tr:
                continue
            tds = tr.find_all("td", recursive=False)
            if not tds:
                continue
            lane = None
            for c in tds[0].get("class", []):
                m = re.match(r"is-boatColor(\d)", c)
                if m:
                    lane = int(m.group(1))
                    break
            if lane is None:
                t0 = tds[0].get_text(strip=True)
                lane = int(t0) if t0.isdigit() else None
            if lane is None or not (1 <= lane <= 6):
                continue
            for td in tds:
                t = _norm(td.get_text())
                if lane not in out["tenji"] and re.fullmatch(r"[4-9]\.\d{2}", t):
                    out["tenji"][lane] = float(t)
                mw = re.fullmatch(r"(\d{2}\.\d)kg", t)
                if mw and lane not in out["weight"]:
                    out["weight"][lane] = float(mw.group(1))
        for course, sp in enumerate(soup.select(".table1_boatImage1 .table1_boatImage1Number")[:6], start=1):
            t = sp.get_text(strip=True)
            if not (t.isdigit() and 1 <= int(t) <= 6):
                continue
            boat = int(t)
            out["course_in"][boat] = course
            st = _start_timing(_closest_class(sp, "table1_boatImage1"))
            if st:
                out["exhibition_st"][boat] = st
        wbox = soup.find("div", class_="weather1_body") or soup
        for key, title in [("風速", "風速"), ("気温", "気温"), ("水温", "水温"), ("波高", "波高")]:
            for unit in wbox.find_all("div", class_="weather1_bodyUnitLabel"):
                tt = unit.find("span", class_="weather1_bodyUnitLabelTitle")
                dd = unit.find("span", class_="weather1_bodyUnitLabelData")
                if tt and dd and title in _norm(tt.get_text()):
                    m = re.search(r"([\d.]+)", dd.get_text(strip=True))
                    if m:
                        out["weather"][key] = float(m.group(1))
                    break
        if (len(out["tenji"]) + len(out["course_in"])) >= (len(best["tenji"]) + len(best["course_in"])):
            best = out
        if len(out["tenji"]) >= 6:
            return out
        if attempt < tries - 1:
            time.sleep(0.5 * (attempt + 1))
    return best


def fetch_odds3t(date: str, jcd: int, rno: int) -> Dict[str, float]:
    """公式の3連単オッズ。表構造から厳密に組番を復元する。"""
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={date}"
    soup = _soup(url, timeout=10)
    if soup is None:
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


# ============================================================
# 買い目の選定
# ============================================================
def pick_buys(combos: Dict[str, float], odds: Dict[str, float]) -> List[Dict]:
    """確率上位から絞り、期待値(確率×オッズ)がEV_MIN以上のものを最大MAX_POINTSまで採用。"""
    if not combos:
        return []
    top = sorted(combos.items(), key=lambda kv: -kv[1])[:TOP_N_PROB]
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


def process_race(date: str, jcd: int, rno: int, close: str) -> Optional[Dict]:
    rows = fetch_racecard(date, jcd, rno)
    if not rows:
        return None
    bi = fetch_beforeinfo(date, jcd, rno)
    src = {"tenji": False, "course": False, "st": False}
    for r in rows:
        ln = r["lane"]
        if ln in bi.get("tenji", {}):
            r["tenji"] = bi["tenji"][ln]
            src["tenji"] = True
        if ln in bi.get("course_in", {}):
            r["course_in"] = bi["course_in"][ln]
            src["course"] = True
        if ln in bi.get("exhibition_st", {}):
            r["tenji_st"] = bi["exhibition_st"][ln]
            src["st"] = True
        if ln in bi.get("weight", {}):
            r["weight"] = bi["weight"][ln]
    feats = make_race_features(rows)
    combos = predict_combo_probs(feats, jcd)
    if not combos:
        return None
    odds = fetch_odds3t(date, jcd, rno)
    buys = pick_buys(combos, odds)
    ev_sum = sum(b["p"] / 100 * b["odds"] for b in buys)
    hit_p = sum(b["p"] for b in buys)
    return {
        "key": f"{date}_{jcd}_{rno}",
        "date": date, "jcd": jcd, "place": JCD_NAME.get(jcd, str(jcd)), "rno": rno,
        "close": close,
        "boats": [{"lane": r["lane"], "name": r.get("name", ""), "cls": r["cls_val"],
                   "nwin": round(r["n_win"], 2), "motor": round(r["m_2ren"] * 100, 1),
                   "tenji": r["tenji"], "course": r["course_in"], "st": r.get("tenji_st", "")}
                  for r in rows],
        "eval": boat_eval(combos),
        # オッズ更新時に買い目を選び直せるよう、確率上位の組番を保存しておく。
        # (オッズが動くと、前回は外れていた組番が条件を満たすことがあるため)
        "probs": {c: round(p, 6) for c, p in sorted(combos.items(), key=lambda kv: -kv[1])[:TOP_N_PROB * 2]},
        "buys": buys,
        "points": len(buys),
        "hitProb": round(hit_p, 1),          # 買い目のどれかが当たる確率(%)
        "evTotal": round(ev_sum / len(buys), 2) if buys else 0,
        "oddsCount": len(odds),
        "oddsAt": datetime.now(JST).isoformat(),
        "weather": bi.get("weather", {}),
        "src": src,
    }


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now(JST).strftime("%Y%m%d")
    print("対象日:", date)

    # 1) 開催している場と締切時刻を調べる
    active = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_close_times, date, j): j for j in range(1, 25)}
        for f in as_completed(futs):
            j = futs[f]
            try:
                ct = f.result()
            except Exception:
                ct = {}
            if ct:
                active[j] = ct
    print("開催中:", len(active), "場 →", " ".join(JCD_NAME[j] for j in sorted(active)))
    if not active:
        json.dump({"date": date, "updatedAt": datetime.now(JST).isoformat(), "races": []},
                  open(os.path.join(DIR, "races.json"), "w", encoding="utf-8"), ensure_ascii=False)
        print("開催なし。races.json を空で保存しました。")
        return

    # 2) 各レースを予想
    tasks = [(j, r, c) for j, ct in active.items() for r, c in ct.items()]
    print("対象レース:", len(tasks))
    races, done, failed = [], 0, 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(process_race, date, j, r, c): (j, r) for j, r, c in tasks}
        for f in as_completed(futs):
            j, r = futs[f]
            try:
                res = f.result()
            except Exception as e:
                res = None
                print("  失敗:", JCD_NAME.get(j), f"{r}R", type(e).__name__)
            done += 1
            if res:
                races.append(res)
            else:
                failed += 1
            if done % 20 == 0:
                print(f"  {done}/{len(tasks)} 完了")

    races.sort(key=lambda x: (x.get("close") or "99:99", x["jcd"], x["rno"]))
    out = {
        "date": date,
        "updatedAt": datetime.now(JST).isoformat(),
        "settings": {"evMin": EV_MIN, "topN": TOP_N_PROB, "maxPoints": MAX_POINTS},
        "races": races,
    }
    with open(os.path.join(DIR, "races.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    buy = [r for r in races if r["points"] >= MIN_POINTS]
    print(f"\n完了: {len(races)}レース保存 / 失敗{failed} / 買い目ありは {len(buy)} レース")
    for r in buy[:10]:
        print(f"  {r['close']} {r['place']}{r['rno']}R  {r['points']}点 "
              f"的中率{r['hitProb']}% 平均EV{r['evTotal']}  " + " ".join(b["combo"] for b in r["buys"][:4]))


if __name__ == "__main__":
    main()
