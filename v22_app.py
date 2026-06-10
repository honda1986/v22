# -*- coding: utf-8 -*-
"""
v22_app.py
==========
v22 モデル(半年学習・キャリブレーション無し)用 予想アプリ
バックテスト＋当日予想の両機能。

【UI最新仕様】
  - サイドバー: EV閾値 / 確率上位N点 / 上限点数(最大20)
  - 確率上位N点に絞ってからEVで二次フィルタする選定ロジック

【依存ファイル (同じフォルダ)】
  - lgb_p1_v22.txt / lgb_p2_v22.txt / lgb_p3_v22.txt
  - lgb_p1_v22_features.json / lgb_p2_v22_features.json / lgb_p3_v22_features.json
"""

import os
import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import lightgbm as lgb
from bs4 import BeautifulSoup

JST = timezone(timedelta(hours=+9), 'JST')
MODEL_DIR = "."   # アプリと同じフォルダ。Colab/Drive ならパスを書き換える。

# ============================================================
# モデル＆特徴量読込
# ============================================================
@st.cache_resource
def load_model(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path): return None
    try:
        return lgb.Booster(model_file=path)
    except Exception as e:
        st.warning(f"モデル読み込み失敗 {filename}: {e}")
        return None

@st.cache_resource
def load_features(filename: str) -> Optional[List[str]]:
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path): return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

m_p1 = load_model("lgb_p1_v22.txt")
m_p2 = load_model("lgb_p2_v22.txt")
m_p3 = load_model("lgb_p3_v22.txt")
features_p1 = load_features("lgb_p1_v22_features.json")
features_p2 = load_features("lgb_p2_v22_features.json")
features_p3 = load_features("lgb_p3_v22_features.json")

JCD_NAME = {
    1:"桐生", 2:"戸田", 3:"江戸川", 4:"平和島", 5:"多摩川", 6:"浜名湖",
    7:"蒲郡", 8:"常滑", 9:"津", 10:"三国", 11:"びわこ", 12:"住之江",
    13:"尼崎", 14:"鳴門", 15:"丸亀", 16:"児島", 17:"宮島", 18:"徳山",
    19:"下関", 20:"若松", 21:"芦屋", 22:"福岡", 23:"唐津", 24:"大村"
}


# ============================================================
# 特徴量生成 (学習時と同じロジック)
# ============================================================
def make_race_features(racer_rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(racer_rows).sort_values("lane").reset_index(drop=True)
    df["win_dev"]    = df["n_win"]  - df["n_win"].mean()
    df["motor_dev"]  = df["m_2ren"] - df["m_2ren"].mean()
    df["st_dev"]     = df["avg_st"].mean() - df["avg_st"]
    df["tenji_dev"]  = df["tenji"].mean() - df["tenji"]
    df["win_rank"]   = df["n_win"].rank(ascending=False, method="min").astype(int)
    df["motor_rank"] = df["m_2ren"].rank(ascending=False, method="min").astype(int)
    df["st_rank"]    = df["avg_st"].rank(ascending=True,  method="min").astype(int)
    df["tenji_rank"] = df["tenji"].rank(ascending=True,   method="min").astype(int)
    df["maezuke"]    = (df["lane"] != df["course_in"]).astype(int)
    df["course_diff"] = df["course_in"] - df["lane"]
    for col in ["avg_st","n_win","tenji"]:
        for direction, shift in [("in",1),("out",-1)]:
            vals = []
            for i in range(len(df)):
                j = i - shift
                if 0 <= j < len(df):
                    vals.append(df.loc[i, col] - df.loc[j, col])
                else:
                    vals.append(0.0)
            df[f"{col}_diff_{direction}"] = vals
    return df


def predict_combo_probs(features_df: pd.DataFrame, race_jcd: int) -> Dict[str, float]:
    """6艇の特徴量から120点の3連単確率を返す。キャリブレーション補正なし。"""
    if not (m_p1 and m_p2 and m_p3):
        return {}
    df = features_df.copy()
    df["jcd"] = race_jcd
    base_cols = features_p1

    # p1: 各艇の1着率
    p1 = {}
    for _, row in df.iterrows():
        x = row[base_cols].values.reshape(1, -1).astype(float)
        p1[int(row["lane"])] = float(m_p1.predict(x)[0])
    s = sum(p1.values())
    if s > 0:
        p1 = {k: v/s for k, v in p1.items()}

    combos = {}
    for w1 in range(1, 7):
        w1_row = df[df["lane"]==w1].iloc[0]
        p2_raw = {}
        for cand in range(1, 7):
            if cand == w1: continue
            cand_row = df[df["lane"]==cand].iloc[0]
            feat = {f: cand_row[f] for f in base_cols if f in cand_row.index}
            for f in ["lane","cls_val","avg_st","n_win","m_2ren","tenji","course_in","maezuke"]:
                feat[f"w1_{f}"] = w1_row[f]
            feat["w1_lane_diff"]   = cand_row["lane"]      - w1_row["lane"]
            feat["w1_course_diff"] = cand_row["course_in"] - w1_row["course_in"]
            x = np.array([feat.get(c, 0.0) for c in features_p2]).reshape(1, -1).astype(float)
            p2_raw[cand] = float(m_p2.predict(x)[0])
        s2 = sum(p2_raw.values())
        p2 = {k: v/s2 if s2>0 else 0 for k, v in p2_raw.items()}

        for w2 in range(1, 7):
            if w2 == w1: continue
            w2_row = df[df["lane"]==w2].iloc[0]
            p3_raw = {}
            for cand in range(1, 7):
                if cand in (w1, w2): continue
                cand_row = df[df["lane"]==cand].iloc[0]
                feat = {f: cand_row[f] for f in base_cols if f in cand_row.index}
                for f in ["lane","cls_val","avg_st","n_win","m_2ren","tenji","course_in","maezuke"]:
                    feat[f"w1_{f}"] = w1_row[f]
                feat["w1_lane_diff"]   = cand_row["lane"]      - w1_row["lane"]
                feat["w1_course_diff"] = cand_row["course_in"] - w1_row["course_in"]
                for f in ["lane","cls_val","avg_st","n_win","m_2ren","tenji","course_in"]:
                    feat[f"w2_{f}"] = w2_row[f]
                feat["w2_lane_diff"] = cand_row["lane"] - w2_row["lane"]
                x = np.array([feat.get(c, 0.0) for c in features_p3]).reshape(1, -1).astype(float)
                p3_raw[cand] = float(m_p3.predict(x)[0])
            s3 = sum(p3_raw.values())
            p3 = {k: v/s3 if s3>0 else 0 for k, v in p3_raw.items()}

            for w3 in range(1, 7):
                if w3 in (w1, w2): continue
                combos[f"{w1}-{w2}-{w3}"] = p1[w1] * p2[w2] * p3[w3]
    return combos


def boat_eval_scores(combo_probs: Dict[str, float]) -> "pd.DataFrame":
    """120点の3連単確率から各艇の評価値を集計する。
    返り値の列: 枠 / 1着率(%) / 2連対率(%)=1〜2着 / 3着内率(%)=1〜3着 / 評価(0-100)。
    各確率は3連単確率の周辺化（その艇が各着になる確率の合計）。
    評価は着順重み(1着3点・2着2点・3着1点)の期待点を0-100換算した総合指標。"""
    p1 = {i: 0.0 for i in range(1, 7)}
    p2 = {i: 0.0 for i in range(1, 7)}
    p3 = {i: 0.0 for i in range(1, 7)}
    for combo, p in combo_probs.items():
        try:
            a, b, c = (int(x) for x in combo.split("-"))
        except (ValueError, AttributeError):
            continue
        p1[a] += p
        p2[b] += p
        p3[c] += p
    rows = []
    for lane in range(1, 7):
        win, second, third = p1[lane], p2[lane], p3[lane]
        score = (3 * win + 2 * second + third) / 3 * 100
        rows.append({
            "枠": lane,
            "1着率(%)": round(win * 100, 1),
            "2連対率(%)": round((win + second) * 100, 1),
            "3着内率(%)": round((win + second + third) * 100, 1),
            "評価": round(score, 1),
        })
    return pd.DataFrame(rows).sort_values("評価", ascending=False).reset_index(drop=True)


# ============================================================
# 当日データ取得
# ============================================================
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
req_sess = requests.Session()
req_sess.headers.update(UA)

RE_CLS    = re.compile(r"([A12B]{2})")
RE_WEIGHT = re.compile(r"(\d+)kg", re.IGNORECASE)
RE_AGE    = re.compile(r"\((\d{2})\)")
CLS_MAP   = {"A1":4, "A2":3, "B1":2, "B2":1}

def _lane_from_class(td) -> Optional[int]:
    div = td.find("div", class_=lambda c: c and "ng1r" in c)
    if not div: return None
    for cls in div.get("class", []):
        m = re.match(r"ng1r(\d)$", cls)
        if m: return int(m.group(1))
    return None


def _norm(s: str) -> str:
    """全角/半角・康熙部首などを正規化して見出し比較を安定させる。"""
    return unicodedata.normalize("NFKC", s or "").replace(" ", "").replace("　", "")


def _combo_from_cell(cell) -> Optional[str]:
    """オッズ行の先頭セル内の ng2r* / ng1r* クラスから艇番3つを出現順に取り出す。
    例: ng2r1 / ng2r3n / ng2r5n -> '1-3-5'
    （このサイトは艇番をテキストではなく CSS クラス名で表現している）"""
    nums = []
    for div in cell.find_all("div"):
        m = re.search(r"ng[12]r([1-6])", " ".join(div.get("class", [])))
        if m:
            nums.append(int(m.group(1)))
            if len(nums) == 3:
                break
    if len(nums) == 3 and len(set(nums)) == 3:
        return f"{nums[0]}-{nums[1]}-{nums[2]}"
    return None


def _parse_odds3t_kyotei(soup) -> Dict[str, float]:
    """kyotei.fun 結合ページから3連単オッズ(人気順)を取得する。
    締切前(レース前)/締切後どちらのレイアウトでも拾えるよう多段で探索する。"""
    def parse_container(container) -> Dict[str, float]:
        out = {}
        for tbl in container.find_all("table", id="oddsTbl"):
            for tr in tbl.find_all("tr"):
                tds = tr.find_all("td", recursive=False)
                if len(tds) != 2:
                    continue
                combo = _combo_from_cell(tds[0])
                if not combo:
                    continue
                txt = tds[1].get_text(strip=True).replace(",", "")
                try:
                    v = float(txt)
                except ValueError:
                    continue
                if v > 0:
                    out[combo] = v
        return out

    # 1) 見出しに「3連単」を含む raceData セクションを優先
    #    （「3連複」には“単”が無いので 2連単/3連複 を誤検出しない）
    for sec in soup.find_all("div", id="raceData"):
        h3 = sec.find("h3")
        if h3 and "3連単" in _norm(h3.get_text()):
            res = parse_container(sec)
            if res:
                return res

    # 2) フォールバック: 「3連単」を含む h3 の親コンテナで再探索
    for h3 in soup.find_all("h3"):
        if "3連単" in _norm(h3.get_text()):
            container = h3.find_parent("div", id="raceData") or h3.parent
            res = parse_container(container)
            if res:
                return res

    return {}


def fetch_race_data(date: str, jcd: int, rno: int):
    url = f"https://info.kyotei.fun/info-{date}-{jcd:02d}-{rno}.html"
    try:
        r = req_sess.get(url, timeout=15)
        r.encoding = r.apparent_encoding
        if r.status_code != 200 or len(r.text) < 5000: return None
    except requests.RequestException:
        return None
    soup = BeautifulSoup(r.text, "html.parser")

    lane_to_rank = {}
    for i, d in enumerate(soup.find_all("div", class_="jyuni")[:6]):
        t = d.get_text(strip=True)
        if t.isdigit(): lane_to_rank[i+1] = int(t)

    base = {i+1: {
        "lane": i+1, "age":30, "cls_val":1, "weight":50, "f_count":0, "avg_st":0.17,
        "n_win":0.0, "n_2ren":0.0, "l_win":0.0, "l_2ren":0.0, "m_2ren":0.0, "b_2ren":0.0,
        "tenji":6.80, "course_in": i+1,
    } for i in range(6)}
    current_label = ""
    for tr in soup.find_all("tr"):
        tds = tr.find_all(["td","th"])
        if not tds: continue
        if len(tds) >= 7:
            current_label = tds[0].get_text(strip=True).replace("\n","").replace(" ","").replace("　","")
            data_tds = tds[-6:]
        elif len(tds) == 6 and current_label:
            data_tds = tds
        else:
            current_label = ""
            continue
        for i in range(6):
            td = data_tds[i]
            txt = td.get_text(" ", strip=True).replace(" ","").replace("　","").replace("\n","")
            lane = i+1
            if "選手名" in current_label:
                m = RE_AGE.search(txt)
                if m: base[lane]["age"] = int(m.group(1))
            elif "選手情報" in current_label or "支部" in current_label:
                m_cls = RE_CLS.search(txt)
                if m_cls: base[lane]["cls_val"] = CLS_MAP.get(m_cls.group(1), 1)
                m_w = RE_WEIGHT.search(txt)
                if m_w: base[lane]["weight"] = int(m_w.group(1))
            elif "級過去2期" in current_label:
                m_cls = RE_CLS.search(txt)
                if m_cls: base[lane]["cls_val"] = CLS_MAP.get(m_cls.group(1), 1)
            elif "全国" in current_label and "勝率" in current_label:
                m2 = re.search(r"^([\d\.]+)", txt); mw = re.search(r"\(([\d\.]+)\)", txt)
                if m2: v=float(m2.group(1)); base[lane]["n_2ren"]=v/100.0 if v>1.0 else v
                if mw: base[lane]["n_win"] = float(mw.group(1))
            elif "当地" in current_label and "勝率" in current_label:
                m2 = re.search(r"^([\d\.]+)", txt); mw = re.search(r"\(([\d\.]+)\)", txt)
                if m2: v=float(m2.group(1)); base[lane]["l_2ren"]=v/100.0 if v>1.0 else v
                if mw: base[lane]["l_win"] = float(mw.group(1))
            elif "モータ" in current_label and "2連率" in current_label:
                m = re.search(r"^([\d\.]+)", txt)
                if m: v=float(m.group(1)); base[lane]["m_2ren"]=v/100.0 if v>1.0 else v
            elif "ボート" in current_label and "2連率" in current_label:
                m = re.search(r"^([\d\.]+)", txt)
                if m: v=float(m.group(1)); base[lane]["b_2ren"]=v/100.0 if v>1.0 else v
            elif "平均ST" in current_label:
                try: base[lane]["avg_st"] = float(txt)
                except: pass
            elif "フライング" in current_label:
                try: base[lane]["f_count"] = int(txt)
                except: pass
            elif current_label == "展示":
                try: base[lane]["tenji"] = float(txt)
                except: pass
            elif current_label == "コースIN":
                c = _lane_from_class(td)
                if c: base[lane]["course_in"] = c

    rows = [base[i+1] for i in range(6)]

    odds_map = _parse_odds3t_kyotei(soup)

    payoff = None
    for box in soup.find_all("div", class_="race_result_end_line"):
        label = box.find("div", class_="race_result_end_label")
        if label and label.get_text(strip=True) == "3連単":
            money = box.find("span", class_="race_result_end_money_num")
            if money:
                t = money.get_text(strip=True).replace(",","")
                if t.isdigit(): payoff = int(t)

    return rows, lane_to_rank, odds_map, payoff


# ============================================================
# 公式サイト(boatrace.jp)からの取得（直前情報・オッズ）
# ============================================================
def fetch_official_beforeinfo(date: str, jcd: int, rno: int) -> Dict:
    """公式の直前情報ページから展示タイム・進入コース・体重・気象を取得。
    返り値: {'tenji': {lane:float}, 'course_in': {boat:course}, 'weight': {lane:float}, 'weather': {...}}
    失敗時は空辞書を返す。

    実構造(2026時点):
      ・出走表テーブル class="is-w748"。各艇は <tbody> 1つ。
        先頭行の td に [枠(is-boatColorN) / 写真 / 名前 / 体重(NN.Nkg) / 展示タイム(N.NN) / チルト / ...]。
      ・スタート展示テーブル: 行の並び順=コース、span.table1_boatImage1Number のテキスト=艇番。
      ・気象は div.weather1_body 内の bodyUnitLabel(Title/Data)。"""
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd:02d}&hd={date}"
    out = {"tenji": {}, "course_in": {}, "weight": {}, "weather": {}}
    try:
        r = req_sess.get(url, timeout=10)
        r.encoding = r.apparent_encoding
        if r.status_code != 200 or len(r.text) < 3000:
            return {}
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception:
        return {}

    # --- 出走表(直前情報)テーブル: 展示タイム・体重 ---
    card = soup.find("table", class_=lambda c: c and "is-w748" in c)
    for tb in (card.find_all("tbody") if card else []):
        first_tr = tb.find("tr")
        if not first_tr:
            continue
        tds = first_tr.find_all("td", recursive=False)
        if not tds:
            continue
        # 枠番: td.is-boatColorN
        lane = None
        for c in tds[0].get("class", []):
            m = re.match(r"is-boatColor(\d)", c)
            if m:
                lane = int(m.group(1)); break
        if lane is None:
            t0 = tds[0].get_text(strip=True)
            if t0.isdigit():
                lane = int(t0)
        if lane is None or not (1 <= lane <= 6):
            continue
        # 同じ行から 展示タイム(例 6.77) と 体重(例 54.4kg) を拾う
        for td in tds:
            txt = _norm(td.get_text())
            if lane not in out["tenji"] and re.fullmatch(r"[4-9]\.\d{2}", txt):
                out["tenji"][lane] = float(txt)
            mw = re.fullmatch(r"(\d{2}\.\d)kg", txt)
            if mw and lane not in out["weight"]:
                out["weight"][lane] = float(mw.group(1))

    # --- スタート展示: 行順=コース / Numberのテキスト=艇番 → course_in[艇番]=コース ---
    spans = soup.select(".table1_boatImage1 .table1_boatImage1Number")
    for course, sp in enumerate(spans[:6], start=1):
        t = sp.get_text(strip=True)
        if t.isdigit() and 1 <= int(t) <= 6:
            out["course_in"][int(t)] = course

    # --- 気象 ---
    wbox = soup.find("div", class_="weather1_body") or soup
    def _wval(title: str) -> Optional[str]:
        for unit in wbox.find_all("div", class_="weather1_bodyUnitLabel"):
            tt = unit.find("span", class_="weather1_bodyUnitLabelTitle")
            dd = unit.find("span", class_="weather1_bodyUnitLabelData")
            if tt and dd and title in _norm(tt.get_text()):
                return dd.get_text(strip=True)
        return None
    for key, title in [("風速(m)", "風速"), ("気温", "気温"),
                       ("水温", "水温"), ("波高(cm)", "波高")]:
        v = _wval(title)
        if v:
            m = re.search(r"([\d.]+)", v)
            if m:
                try: out["weather"][key] = float(m.group(1))
                except ValueError: pass

    return out


# 公式 odds3t は表構造から組番を厳密に復元するため信頼できる。True で有効。
OFFICIAL_ODDS3T_ENABLED = True


def _parse_odds3t_official(soup) -> Dict[str, float]:
    """公式 odds3t の表から3連単オッズを復元する。
    表構造: 列グループ=1着(thead), 各グループ内が [2着(rowspan=4) / 3着 / オッズ]。
    DOM は行優先なので 1着 がインターリーブする。各オッズセルについて
      1着 = その行で何番目のオッズか → ヘッダの艇番
      2着 = その列グループで現在有効な rowspan セルの値（ブロック先頭行で更新）
      3着 = オッズ直前の艇番セル
    として厳密に組番を決める（並び順の決め打ちをしない）。"""
    # オッズセルを含むテーブルを特定
    odds_table = None
    for tbl in soup.find_all("table"):
        if tbl.select("td.oddsPoint"):
            odds_table = tbl
            break
    if odds_table is None:
        return {}

    # ヘッダから1着の艇番を列順に取得（通常 [1,2,3,4,5,6]）
    head_boats = []
    thead = odds_table.find("thead")
    if thead:
        for th in thead.find_all("th"):
            t = th.get_text(strip=True)
            if t.isdigit() and 1 <= int(t) <= 6:
                head_boats.append(int(t))
    if not (2 <= len(head_boats) <= 6):
        head_boats = [1, 2, 3, 4, 5, 6]

    out = {}
    cur_2 = [None] * len(head_boats)   # 列ごとの現在の2着
    for tr in odds_table.select("tbody > tr"):
        tds = tr.find_all("td", recursive=False)
        # ブロック先頭行: rowspan セル(=2着)が列数ぶん並ぶ
        twos = [td for td in tds
                if td.has_attr("rowspan") and "oddsPoint" not in (td.get("class") or [])]
        if len(twos) == len(head_boats):
            for gi, td in enumerate(twos):
                tv = td.get_text(strip=True)
                if tv.isdigit():
                    cur_2[gi] = int(tv)
        # 行内のセルを順に走査。直近の数字セル=3着、オッズの出現順=列(1着)。
        gi = 0
        last_num = None
        for td in tds:
            cls = td.get("class") or []
            txt = td.get_text(strip=True)
            if "oddsPoint" in cls:
                if gi < len(head_boats):
                    a, b, c = head_boats[gi], cur_2[gi], last_num
                    if a and b and c and len({a, b, c}) == 3:
                        try:
                            v = float(txt.replace(",", ""))
                        except ValueError:
                            v = 0.0
                        if v > 0:
                            out[f"{a}-{b}-{c}"] = v
                gi += 1
            elif txt.isdigit():
                last_num = int(txt)
    return out


def fetch_official_odds3t(date: str, jcd: int, rno: int) -> Dict[str, float]:
    """公式の3連単オッズページを取得し、構造ベースで {'1-2-3': 5.8, ...} を返す。
    取れなければ空辞書。"""
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={date}"
    try:
        r = req_sess.get(url, timeout=10)
        r.encoding = r.apparent_encoding
        if r.status_code != 200 or len(r.text) < 3000:
            return {}
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception:
        return {}
    return _parse_odds3t_official(soup)


def fetch_race_data_hybrid(date: str, jcd: int, rno: int):
    """ハイブリッド取得:
       1) kyotei.fun から出走表・選手成績・初期オッズ・結果を取得
       2) 公式の直前情報で展示・コースINを上書き (取れたら)
       3) 公式のオッズで上書き (取れたら)
       返り値: (racers, lane_to_rank, odds_map, payoff, sources)
        sources は何が公式から取れたかのフラグ辞書"""
    sources = {"odds_from_official": False, "tenji_from_official": False,
               "course_in_from_official": False, "weather": {}}

    base_res = fetch_race_data(date, jcd, rno)
    if not base_res:
        return None
    racers, lane_to_rank, odds_map, payoff = base_res

    # 公式直前情報
    time.sleep(0.5)   # 節度のための小休止
    bi = fetch_official_beforeinfo(date, jcd, rno)
    if bi.get("tenji"):
        for r in racers:
            if r["lane"] in bi["tenji"]:
                r["tenji"] = bi["tenji"][r["lane"]]
        sources["tenji_from_official"] = True
    if bi.get("course_in"):
        for r in racers:
            if r["lane"] in bi["course_in"]:
                r["course_in"] = bi["course_in"][r["lane"]]
        sources["course_in_from_official"] = True
    if bi.get("weight"):
        for r in racers:
            if r["lane"] in bi["weight"]:
                r["weight"] = bi["weight"][r["lane"]]
    sources["weather"] = bi.get("weather", {})

    # オッズ:
    #   kyotei.fun は各組番を CSS クラス(ng2r*)で明示しているため「組番→オッズ」の
    #   対応が一意で確実。これを最優先（診断で oddsTbl=14 / ng2r=510 を確認済み）。
    #   公式 odds3t は 120 セル取れるが「セルの並び順→組番」が未検証で、万一ズレると
    #   全EVが壊れるため、kyotei が空(=反映前)のときだけフォールバックとして使う。
    # オッズ: 公式 odds3t を最優先（表構造から厳密復元・最新/締切時オッズ）。
    #   公式が取れないとき(売上開始前など)は kyotei.fun の人気順オッズにフォールバック。
    if OFFICIAL_ODDS3T_ENABLED:
        time.sleep(0.5)
        official_odds = fetch_official_odds3t(date, jcd, rno)
        if len(official_odds) >= 100:
            odds_map = official_odds
            sources["odds_from_official"] = True

    return racers, lane_to_rank, odds_map, payoff, sources


def fetch_kyotei_odds_page(date: str, jcd: int, rno: int) -> Dict[str, float]:
    """kyotei.club 専用オッズページ(od-)から3連単オッズを取得（best-effort）。
    結合ページが JS でオッズを描画していて取れない場合の代替。
    同系列ページなら _parse_odds3t_kyotei がそのまま使える（構造が違えば空を返す）。"""
    url = f"https://odds.kyotei.club/od-{date}-{jcd:02d}-{rno}.html"
    try:
        r = req_sess.get(url, timeout=10)
        r.encoding = r.apparent_encoding
        if r.status_code != 200 or len(r.text) < 2000:
            return {}
        return _parse_odds3t_kyotei(BeautifulSoup(r.text, "html.parser"))
    except Exception:
        return {}


def debug_fetch_report(date: str, jcd: int, rno: int) -> str:
    """オッズが取れない原因の切り分け用。各ソースの“生HTML”に
    オッズの痕跡(マーカー)が含まれているかを報告する。
    ブラウザは JS を実行するので画面に見えるが、requests は実行しない。
    生HTMLにマーカーが無ければ JS 描画＝サーバー取得(requests)では拾えない、と判定できる。"""
    targets = {
        "kyotei.fun (結合)": f"https://info.kyotei.fun/info-{date}-{jcd:02d}-{rno}.html",
        "公式 直前情報":      f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd:02d}&hd={date}",
        "公式 odds3t":        f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={date}",
        "kyotei.club (od)":  f"https://odds.kyotei.club/od-{date}-{jcd:02d}-{rno}.html",
    }
    lines = []
    for name, url in targets.items():
        lines.append(f"■ {name}\n  {url}")
        try:
            r = req_sess.get(url, timeout=10)
            r.encoding = r.apparent_encoding
            html = r.text
            mk = {
                "status": r.status_code,
                "len": len(html),
                "'オッズ'": html.count("オッズ"),
                "'3連単'": html.count("3連単"),
                "oddsTbl": html.count("oddsTbl"),
                "oddsPoint": html.count("oddsPoint"),
                "ng2r": len(re.findall(r"ng2r[1-6]", html)),
                "is-w748": html.count("is-w748"),
            }
            lines.append("  " + " / ".join(f"{k}={v}" for k, v in mk.items()))
            decimals = re.findall(r">\s*\d{1,4}\.\d\s*<", html)
            lines.append(f"  生HTML内の小数値(>n.n<): {len(decimals)} 件  例: {decimals[:5]}")
        except Exception as e:
            lines.append(f"  取得エラー: {e}")
        lines.append("")
    return "\n".join(lines)


def select_bets_by_ev(combo_probs: Dict[str, float], odds_map: Dict[str, float],
                       ev_th: float, top_n_prob: int, max_n: int) -> List[Dict]:
    """確率上位N点に絞り、その中でEV>=ev_thをEV降順で最大max_n点採用。"""
    sorted_by_prob = sorted(combo_probs.items(), key=lambda x: x[1], reverse=True)
    candidates = sorted_by_prob[:top_n_prob]
    out = []
    for combo, p in candidates:
        o = odds_map.get(combo, 0.0)
        if o <= 0: continue
        ev = p * o
        if ev < ev_th: continue
        out.append({"bet": combo, "prob": p, "odds": o, "ev": ev})
    out.sort(key=lambda x: x["ev"], reverse=True)
    return out[:max_n]


# ============================================================
# コース別連対率: boatracer.kyotei.club の選手ページ(racerYm)から
# 各艇の進入コースの2連対率/3連対率を取得し、3連単確率を後段で補正する。
# 登録番号は kyotei.sakura.ne.jp の出走表(racelist)から取得する。
# （いずれも学習済み重みではないヒューリスティックな後段補正）
# ============================================================
JCD_ROMAJI = {
    1: "kiryu", 2: "toda", 3: "edogawa", 4: "heiwajima", 5: "tamagawa", 6: "hamanako",
    7: "gamagori", 8: "tokoname", 9: "tsu", 10: "mikuni", 11: "biwako", 12: "suminoe",
    13: "amagasaki", 14: "naruto", 15: "marugame", 16: "kojima", 17: "miyajima",
    18: "tokuyama", 19: "shimonoseki", 20: "wakamatsu", 21: "ashiya", 22: "fukuoka",
    23: "karatsu", 24: "omura",
}


def parse_racelist_regnos(html: str) -> Dict[int, Dict[int, int]]:
    """sakura の出走表HTMLから各レース・各艇の登録番号を取り出す。
    返り値: {rno: {lane: 登録番号}}。
    「登録番号」行に <a href=".../racer-XXXX.html"> が6艇ぶん（艇番順）入っている。
    表は <a name="1R">..<a name="12R"> の順なので、登録番号行を出現順に R1..R12 へ割当。"""
    rnos = [int(m.group(1)) for m in re.finditer(r'name="(\d+)R"', html)]
    soup = BeautifulSoup(html, "html.parser")
    groups = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        label = tds[0].get_text(strip=True).replace(" ", "").replace("\u3000", "")
        if label.startswith("登録番号"):
            regs = []
            for a in tr.find_all("a", href=re.compile(r"racer-(\d+)\.html")):
                m = re.search(r"racer-(\d+)\.html", a.get("href", ""))
                if m:
                    regs.append(int(m.group(1)))
            if len(regs) >= 6:
                groups.append(regs[:6])
    out: Dict[int, Dict[int, int]] = {}
    for idx, regs in enumerate(groups):
        rno = rnos[idx] if idx < len(rnos) else idx + 1
        out[rno] = {lane: regs[lane - 1] for lane in range(1, 7)}
    return out


def fetch_racelist_regnos(date: str, jcd: int) -> Dict[int, Dict[int, int]]:
    """指定日・場の全12R分の登録番号を sakura から取得（場ごとに1回）。失敗時は空。"""
    romaji = JCD_ROMAJI.get(jcd)
    if not romaji:
        return {}
    url = f"https://kyotei.sakura.ne.jp/racelist-{romaji}-{date}.html"
    try:
        r = req_sess.get(url, timeout=10)
        r.encoding = r.apparent_encoding
        if r.status_code != 200 or len(r.text) < 3000:
            return {}
        return parse_racelist_regnos(r.text)
    except Exception:
        return {}


def parse_course_rates(html: str) -> Dict[int, Dict[str, float]]:
    """選手ページ(racerYm)HTMLからコース別の2連対率/3連対率(%)を取り出す。
    返り値: {course(1-6): {'r2': 2連対率%, 'r3': 3連対率%}}。
    コース別テーブルは「Nコース」ラベル行の下に各指標が
    <td class="label">3連対率</td><td class="text">87.0% (20)</td> の形で並ぶ。"""
    soup = BeautifulSoup(html, "html.parser")
    out: Dict[int, Dict[str, float]] = {}
    cur = None
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        label = tds[0].get_text(strip=True)
        m = re.match(r"([1-6])コース", label)
        if m:
            cur = int(m.group(1))
            out.setdefault(cur, {})
            continue
        if cur and len(tds) >= 2:
            mm = re.match(r"([\d.]+)", tds[1].get_text(strip=True))
            if mm:
                v = float(mm.group(1))
                if label == "3連対率":
                    out[cur]["r3"] = v
                elif label == "2連対率":
                    out[cur]["r2"] = v
    return out


def fetch_course_rates(date: str, regno: int) -> Dict[int, Dict[str, float]]:
    """登録番号の選手ページ(racerYm)からコース別連対率を取得（選手ごとに1回）。失敗時は空。"""
    url = f"https://boatracer.kyotei.club/racerYm-{regno}-{date}.html"
    try:
        r = req_sess.get(url, timeout=10)
        r.encoding = r.apparent_encoding
        if r.status_code != 200 or len(r.text) < 3000:
            return {}
        return parse_course_rates(r.text)
    except Exception:
        return {}


def apply_course_adjustment(combo_probs: Dict[str, float],
                            r2_by_lane: Dict[int, Optional[float]],
                            r3_by_lane: Dict[int, Optional[float]],
                            k: float) -> Dict[str, float]:
    """各艇の進入コースの2連対率/3連対率(%)で3連単確率を補正し再正規化する（k=0で無補正）。
    1着・2着は2連対率、3着は3連対率の偏差を使い、率の高い艇の組番を増幅。
    有効データが2艇未満の場合は補正しない。"""
    if k <= 0 or not combo_probs:
        return combo_probs
    known2 = [v for v in r2_by_lane.values() if v is not None]
    known3 = [v for v in r3_by_lane.values() if v is not None]
    if len(known2) < 2 or len(known3) < 2:
        return combo_probs
    mean2 = sum(known2) / len(known2)
    mean3 = sum(known3) / len(known3)
    z2 = {lane: ((r2_by_lane.get(lane) if r2_by_lane.get(lane) is not None else mean2) - mean2) / 100.0
          for lane in range(1, 7)}
    z3 = {lane: ((r3_by_lane.get(lane) if r3_by_lane.get(lane) is not None else mean3) - mean3) / 100.0
          for lane in range(1, 7)}
    adj = {}
    for combo, p in combo_probs.items():
        try:
            a, b, c = (int(x) for x in combo.split("-"))
        except ValueError:
            adj[combo] = p
            continue
        adj[combo] = p * float(np.exp(k * (z2[a] + z2[b] + z3[c])))
    tot = sum(adj.values())
    if tot > 0:
        adj = {kk: v / tot for kk, v in adj.items()}
    return adj


def apply_weather_adjustment(combo_probs: Dict[str, float],
                              wind_m: Optional[float], wave_cm: Optional[float],
                              k: float,
                              course_by_boat: Optional[Dict[int, int]] = None) -> Dict[str, float]:
    """風速(m)・波高(cm)から水面の荒れ度を算出し、3連単確率を後段補正する（k=0で無効）。
    一般に荒れ水面ではイン(1コース)の1着信頼度が下がり、外コースの捲り・差しが増える
    傾向をヒューリスティックに反映する。穏やかな水面(風<4m かつ 波<4cm)では何もしない。
    course_by_boat: {艇番: 進入コース}（公式スタート展示の進入。無ければ枠なり想定）。"""
    if k <= 0 or not combo_probs:
        return combo_probs
    sev = (max(0.0, ((wind_m or 0.0) - 3.0) / 5.0)
           + max(0.0, ((wave_cm or 0.0) - 3.0) / 7.0))
    sev = min(sev, 1.5)
    if sev <= 0:
        return combo_probs
    cmap = course_by_boat or {i: i for i in range(1, 7)}
    # 進入コース別の効き（1着方向）。荒れるほどインを減衰し外を増幅。
    w1 = {1: -1.0, 2: -0.2, 3: 0.1, 4: 0.4, 5: 0.3, 6: 0.4}
    adj = {}
    for combo, p in combo_probs.items():
        try:
            a, b, c = (int(x) for x in combo.split("-"))
        except ValueError:
            adj[combo] = p
            continue
        ca = w1.get(cmap.get(a, a), 0.0)
        cb = w1.get(cmap.get(b, b), 0.0)
        adj[combo] = p * float(np.exp(k * sev * (ca + 0.3 * cb)))
    tot = sum(adj.values())
    return {kk: v / tot for kk, v in adj.items()} if tot > 0 else combo_probs


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="v22 半年学習版", layout="wide")
st.title("🚤 v22 EVバックテスト＆当日予想 (半年学習)")

model_ready = all([m_p1, m_p2, m_p3, features_p1, features_p2, features_p3])
if not model_ready:
    st.error("⚠️ v22モデルファイルが見つかりません。以下をアプリと同じフォルダに置いてください:")
    st.code("lgb_p1_v22.txt\nlgb_p2_v22.txt\nlgb_p3_v22.txt\n"
            "lgb_p1_v22_features.json\nlgb_p2_v22_features.json\nlgb_p3_v22_features.json")
    st.stop()

# サイドバー
st.sidebar.markdown("### ⚙️ EV判定設定")
ev_th      = st.sidebar.slider("EV閾値", 1.0, 3.0, 1.30, 0.05)
top_n_prob = st.sidebar.slider("予想確率上位 N 点に絞る", 3, 120, 15, 1,
                                 help="モデルが自信を持つ上位N点を候補にし、その中でEV判定する")
max_bets   = st.sidebar.slider("1レース上限点数", 1, 20, 4, 1)
bet_amt    = st.sidebar.number_input("1点の購入金額(円)", min_value=100, step=100, value=100)
st.sidebar.caption("💡 確率上位N点に絞ってからEV判定。Nを小さくすると厳選、大きくすると候補拡大。")

st.sidebar.markdown("### 🎯 1号艇フィルタ（全レーススキャン用）")
boat1_min = st.sidebar.slider("1号艇の勝率(1着率) 下限(%)", 0.0, 100.0, 0.0, 1.0,
                              help="全レーススキャン(タブ3)で、モデル予想の1号艇1着率がこの値以上のレースだけを抽出します。0で無効。")
st.sidebar.caption("💡 1号艇の1着率が高い＝堅いと予想されるレースだけに絞れます（0で全レース対象）。")

tab1, tab2, tab3 = st.tabs(["📊 バックテスト", "🎯 当日予想", "🔎 全レース勝率スキャン"])

# ----------------------------- Tab1
with tab1:
    st.markdown("##### CSVを読み、EV>閾値の買い目だけ買った場合の回収率を測定。")
    uploaded = st.file_uploader("v19_dataset.csv (半年分でも可)", type=["csv"])
    period = st.text_input("期間 (開始,終了 例: 20260425,20260430)", "")

    if uploaded and st.button("🚀 バックテスト実行", type="primary"):
        df = pd.read_csv(uploaded, dtype={"date":str, "result_combo":str, "odds_3t_json":str})
        df = df[df["tenji"] > 0]
        df = df[df["payoff_3t"] > 0]
        if period.strip():
            try:
                s, e = [x.strip() for x in period.split(",")]
                df = df[(df["date"] >= s) & (df["date"] <= e)]
                st.info(f"期間フィルタ {s}〜{e}: {len(df)//6:,}レース")
            except Exception:
                st.warning("期間フィルタの形式が不正。全期間で実行。")

        race_keys = df[["date","jcd","rno"]].drop_duplicates().values.tolist()
        st.write(f"対象 {len(race_keys):,} レース処理中 ...")
        prog = st.progress(0.0)

        records, bet_details = [], []
        for idx, (d, j, r) in enumerate(race_keys):
            sub = df[(df["date"]==d)&(df["jcd"]==j)&(df["rno"]==r)]
            if len(sub) != 6: continue
            racers = sub.to_dict("records")
            try:
                odds_map = json.loads(sub.iloc[0]["odds_3t_json"])
            except Exception:
                continue
            result_combo = sub.iloc[0]["result_combo"]
            payoff = int(sub.iloc[0]["payoff_3t"])

            feat_df = make_race_features(racers)
            combo_probs = predict_combo_probs(feat_df, int(j))
            chosen = select_bets_by_ev(combo_probs, odds_map, ev_th, top_n_prob, max_bets)

            buys = [c["bet"] for c in chosen]
            hit = result_combo in buys
            inv = len(buys) * bet_amt
            ret = payoff * (bet_amt/100.0) if hit else 0
            records.append({
                "date":d, "jcd":int(j), "rno":int(r),
                "n_bets": len(buys),
                "buys": ",".join(buys) if buys else "見",
                "result": result_combo,
                "hit": 1 if hit else 0,
                "investment": inv, "return": ret, "payoff": payoff,
                "sum_ev": round(sum(c["ev"] for c in chosen), 2),
            })
            for c in chosen:
                bet_details.append({
                    "prob": c["prob"], "odds": c["odds"], "ev": c["ev"],
                    "hit": 1 if c["bet"] == result_combo else 0,
                    "investment": bet_amt,
                    "return": payoff*(bet_amt/100.0) if c["bet"] == result_combo else 0,
                })
            if idx % 30 == 0 or idx == len(race_keys)-1:
                prog.progress((idx+1)/len(race_keys))

        if not records:
            st.error("結果が空でした。"); st.stop()
        res = pd.DataFrame(records)
        bet_races = res[res["n_bets"] > 0]
        skip_races = res[res["n_bets"] == 0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("対象", f"{len(res):,}")
        c2.metric("買った", f"{len(bet_races):,}", f"見送り {len(skip_races):,}")
        if len(bet_races) > 0:
            tot_inv = bet_races["investment"].sum()
            tot_ret = bet_races["return"].sum()
            hit_rate = bet_races["hit"].sum() / len(bet_races) * 100
            ret_rate = tot_ret/tot_inv*100 if tot_inv>0 else 0
            c3.metric("回収率", f"{ret_rate:.1f}%",
                      f"投資{int(tot_inv):,} / 回収{int(tot_ret):,}円")
            c4.metric("的中率", f"{hit_rate:.1f}%",
                      f"{int(bet_races['hit'].sum())}/{len(bet_races)}")
            if ret_rate >= 100:
                st.success(f"🎉 回収率 {ret_rate:.1f}% — 理論プラス。標本{len(bet_races)}本での結果なので追加検証必須。")
            elif ret_rate >= 85:
                st.info(f"回収率 {ret_rate:.1f}% — 控除率の壁は超えたが100%未満。設定を細かく動かして探索を。")
            else:
                st.warning(f"回収率 {ret_rate:.1f}% — 控除率の壁未満。EV閾値を上げる、Nを下げる等を試してください。")

        st.markdown("---")

        # EV帯別マトリクス
        if bet_details:
            st.subheader("📈 EV帯別の回収率（買い目1点ごと）")
            bd = pd.DataFrame(bet_details)
            ev_bins = [1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0, 99]
            bd["ev_band"] = pd.cut(bd["ev"], bins=ev_bins, right=False,
                                     labels=[f"{ev_bins[i]:.1f}-{ev_bins[i+1]:.1f}" for i in range(len(ev_bins)-1)])
            rows = []
            for band, g in bd.groupby("ev_band"):
                if len(g)==0: continue
                inv = g["investment"].sum(); ret = g["return"].sum()
                rows.append({
                    "EV帯": band,
                    "買い目数": len(g),
                    "的中": int(g["hit"].sum()),
                    "的中率(%)": round(g["hit"].mean()*100, 2),
                    "投資": int(inv),
                    "回収": int(ret),
                    "回収率(%)": round(ret/inv*100, 1) if inv>0 else 0,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.caption("単調にEVが高いほど回収率が高い帯域構造が理想。歪みがあれば確率が過大評価。")

        st.markdown("---")
        st.subheader("📋 レース別結果")
        st.dataframe(res, use_container_width=True)

# ----------------------------- Tab2
with tab2:
    st.markdown("##### 1レースを取得 → v22モデルでEV判定")
    cA, cB, cC = st.columns(3)
    with cA: d_input = st.date_input("日付", value=datetime.now(JST).date())
    with cB: v_idx = st.selectbox("場", options=list(JCD_NAME.keys()), format_func=lambda x: JCD_NAME[x])
    with cC: r_idx = st.selectbox("R", options=list(range(1, 13)))

    if st.button("🔍 取得して予想", type="primary", use_container_width=True):
        dstr = d_input.strftime("%Y%m%d")
        with st.spinner("取得中 (kyotei.fun → 公式で補完)..."):
            res = fetch_race_data_hybrid(dstr, v_idx, r_idx)
        if not res:
            st.error("取得失敗。日付・場・レース番号を確認してください。"); st.stop()
        racers, lane_to_rank, odds_map, payoff, sources = res

        st.subheader("出走表")
        df_show = pd.DataFrame(racers)[["lane","cls_val","age","avg_st","n_win","m_2ren","tenji","course_in"]]
        df_show.columns = ["枠","級","年齢","平均ST","勝率","M2連率","展示","コースIN"]
        st.dataframe(df_show.set_index("枠"), use_container_width=True)

        # --- オッズ取得状況 ---
        if sources.get("odds_from_official"):
            src_label = "公式 boatrace.jp"
        elif sources.get("odds_from_kyotei_odds_page"):
            src_label = "kyotei.club (od)"
        elif odds_map:
            src_label = "kyotei.fun"
        else:
            src_label = "なし"
        n_odds = len(odds_map)
        odds_msg = f"オッズ取得: {n_odds}/120 件 ・ 取得元: {src_label}"
        if n_odds >= 100:
            st.success(odds_msg)
        elif n_odds > 0:
            st.warning(odds_msg + " ⚠️ 一部のみ取得（締切直前で変動中／一部未確定の可能性）")
        else:
            st.error(odds_msg + "\n\n❌ オッズが取得できませんでした。"
                     "kyotei.fun にまだ3連単オッズが掲載されていません"
                     "（発走が近づくと掲載されます。少し待って再取得してください）。"
                     "公式オッズは並び順の検証が終わるまで一時停止中です"
                     "（誤った並びでEVが壊れるのを防ぐため）。")
        extra = []
        if sources.get("tenji_from_official"): extra.append("展示=公式")
        if sources.get("course_in_from_official"): extra.append("進入=公式")
        if sources.get("weather"):
            extra.append("気象: " + " / ".join(f"{k}{v}" for k, v in sources["weather"].items()))
        if extra:
            st.caption(" ・ ".join(extra))

        feat_df = make_race_features(racers)
        combo_probs = predict_combo_probs(feat_df, v_idx)

        if combo_probs:
            st.subheader("🛥️ 各艇の評価値")
            eval_df = boat_eval_scores(combo_probs)
            st.dataframe(eval_df.set_index("枠"), use_container_width=True)
            st.bar_chart(eval_df.set_index("枠")["1着率(%)"].sort_index())
            st.caption("1着率／2連対率(1〜2着)／3着内率(1〜3着) はモデルの3連単確率を集計したもの。"
                       "評価は着順重み(1着3・2着2・3着1)の期待点を0-100換算した総合指標で、強い順に並べています。")

        if odds_map:
            chosen = select_bets_by_ev(combo_probs, odds_map, ev_th, top_n_prob, max_bets)
            st.subheader(f"🎯 採用買い目 (EV≥{ev_th}, 確率上位{top_n_prob}点から選定, 最大{max_bets}点)")
            if chosen:
                df_b = pd.DataFrame([{
                    "買い目": c["bet"],
                    "予想確率(%)": round(c["prob"]*100, 2),
                    "オッズ": round(c["odds"], 1),
                    "EV": round(c["ev"], 3),
                } for c in chosen])
                st.dataframe(df_b.set_index("買い目"), use_container_width=True)
                st.code(",".join(c["bet"] for c in chosen))
                if payoff and lane_to_rank:
                    r1 = next((l for l,r in lane_to_rank.items() if r==1), None)
                    r2 = next((l for l,r in lane_to_rank.items() if r==2), None)
                    r3 = next((l for l,r in lane_to_rank.items() if r==3), None)
                    if r1 and r2 and r3:
                        result = f"{r1}-{r2}-{r3}"
                        buys = [c["bet"] for c in chosen]
                        hit = result in buys
                        inv = len(buys)*bet_amt
                        ret = payoff*(bet_amt/100.0) if hit else 0
                        st.success(f"結果: {result} ({payoff}円) — {'🎯 的中' if hit else '❌ 外れ'} "
                                   f"投資 {inv:,} / 回収 {int(ret):,}円")
            else:
                st.info("条件を満たす買い目なし → 見送り")

        st.subheader("📊 確率上位 (参考)")
        top = sorted(combo_probs.items(), key=lambda x: x[1], reverse=True)[:15]
        df_top = pd.DataFrame([{"買い目":k, "予想確率(%)":round(v*100,2),
                                 "オッズ":odds_map.get(k,0), "EV":round(v*odds_map.get(k,0),3)}
                                for k,v in top])
        st.dataframe(df_top.set_index("買い目"), use_container_width=True)

    with st.expander("🔧 オッズが取れない時の診断（生HTMLを確認）"):
        st.caption("ブラウザは JavaScript を実行するので画面にオッズが見えますが、"
                   "このアプリ(requests)は JS を実行しません。各ソースの“生HTML”に"
                   "オッズの痕跡が入っているかを確認します。痕跡が無ければ JS 描画で、"
                   "サーバー取得(requests)では拾えないことを意味します。")
        if st.button("診断を実行", key="diag"):
            dstr2 = d_input.strftime("%Y%m%d")
            with st.spinner("各ソースの生HTMLを取得して確認中..."):
                report = debug_fetch_report(dstr2, v_idx, r_idx)
            st.code(report)
            st.markdown("**公式 直前情報のパース結果（展示タイム・進入・気象）**")
            bi = fetch_official_beforeinfo(dstr2, v_idx, r_idx)
            if bi and (bi.get("tenji") or bi.get("course_in")):
                st.write({
                    "展示タイム (枠→秒)": bi.get("tenji"),
                    "進入コース (艇番→コース)": bi.get("course_in"),
                    "体重 (枠→kg)": bi.get("weight"),
                    "気象": bi.get("weather"),
                })
                st.caption("ここに展示タイムが6枠分出ていれば、当日予想でも公式の展示タイムが反映されます。")
            else:
                st.warning("公式 直前情報を取得/解析できませんでした。"
                           "上の『公式 直前情報』が status≠200 や len 極小なら、"
                           "ホスティング先IPが bot 保護(Akamai)でブロックされている可能性が高いです。")

            st.markdown("**オッズ照合（kyotei.fun と 公式 odds3t を同じ組番で比較）**")
            try:
                _ko = fetch_race_data(dstr2, v_idx, r_idx)
                kmap = _ko[2] if _ko else {}
            except Exception:
                kmap = {}
            oo = fetch_official_odds3t(dstr2, v_idx, r_idx)
            st.write(f"kyotei.fun: {len(kmap)} 件 / 公式 odds3t: {len(oo)} 件")
            checks = ["1-2-3", "1-2-4", "1-3-2", "2-1-3", "3-1-2", "6-5-4"]
            st.write({c: {"kyotei": kmap.get(c), "公式": oo.get(c)} for c in checks})
            st.caption("同じ組番で kyotei と 公式 の値が一致していれば両方の対応付けが正しい。"
                       "食い違う場合は公式 odds3t の並び順の仮定が誤り（その時は odds3t の view-source をください）。")
            st.caption("ng2r / oddsTbl / 小数値 がすべて 0 のソースは JS 描画です。"
                       "どれか1つでも ng2r や小数値が多いソースがあれば、そこから取得できます。"
                       "その生HTML(view-source)を共有してもらえれば、その形式に合わせて確実に対応します。")


# ----------------------------- Tab3
with tab3:
    st.markdown("##### 選択日の全レースを解析し、勝率(3連単の予想確率)が一定以上の買い目を抽出します（オッズ取得なし）。")
    c3a, c3b = st.columns([1, 2])
    with c3a:
        d3_input = st.date_input("日付", value=datetime.now(JST).date(), key="t3_date")
    with c3b:
        sel_jcds = st.multiselect(
            "対象場（既定: 全場）",
            options=list(JCD_NAME.keys()),
            default=list(JCD_NAME.keys()),
            format_func=lambda x: JCD_NAME[x],
            key="t3_jcds",
        )
    win_th = st.slider("勝率しきい値(%)", 1.0, 30.0, 10.0, 0.5, key="t3_winth",
                        help="3連単の予想勝率（モデル確率）がこの値以上の買い目だけを表示します。"
                             "EVモードON時はEV判定で選定するため、この値は使いません。")
    ce1, ce2 = st.columns([1, 2])
    with ce1:
        ev_mode = st.checkbox("EVモード（公式オッズで期待値判定）", value=True, key="t3_ev",
                              help="公式(boatrace.jp)から3連単オッズ・展示タイム・進入コース・風・波を取得し、"
                                   "サイドバーのEV閾値/上位N点/上限点数で買い目を選定します。"
                                   "回収率を狙うなら必須（確率だけで買うと控除率ぶん負けやすい）。")
    with ce2:
        weather_k = st.slider("風・波補正の効き k（0で無効）", 0.0, 2.0, 0.7, 0.1, key="t3_weather_k",
                              help="公式直前情報の風速・波高から荒れ度を算出し、荒れ水面ではイン信頼度を下げ"
                                   "外コースを増幅するヒューリスティック補正（EVモードON時のみ気象が取れます）。")
    st.caption("⚠️ 各場の1Rで開催有無を確認し、開催場のみ全12レースを並列で取得して解析します。"
               "EVモードONではレースごとに公式の直前情報＋オッズも取得するため約3倍のリクエスト数になります"
               "（締切前のレースのみ意味があります。確定済レースは最終オッズで参考集計）。"
               "結果が出ているレースは三連単の結果・払戻を表示し、確定レースのみで当日回収率を集計します。")
    cs1, cs2 = st.columns([1, 2])
    with cs1:
        use_course = st.checkbox("コース別連対率で補正する", value=True, key="t3_course",
                                 help="各艇の選手ページ(racerYm)から進入コースの2連対率/3連対率を取得して確率を補正します。")
    with cs2:
        course_k = st.slider("コース別補正の効き k（0で無効）", 0.0, 3.0, 1.0, 0.1, key="t3_course_k",
                             help="進入コースの連対率が高い艇の組番を増幅。学習した重みではなくヒューリスティックな後段補正です。")
    st.caption("⚠️ コース別補正ONは選手ごとにページ取得するため非常に重いです（選手数ぶんのリクエスト）。"
               "全場だと数百〜千リクエストになるので、場を絞ることを強く推奨します。並列取得＋同一セッション内キャッシュにより、同じ日の再実行は大幅に速くなります。")

    n_workers = st.slider("同時取得数（並列ワーカー）", 1, 6, 3, 1, key="t3_workers",
                          help="ページ取得を並列化します。増やすほど速くなりますが、取得先サイトへの負荷とブロックの"
                               "リスクが増えるため 3 前後を推奨します。")

    if st.button("🚀 全レース解析", type="primary", use_container_width=True, key="t3_run"):
        if not sel_jcds:
            st.warning("解析する場を1つ以上選択してください。")
        else:
            dstr3 = d3_input.strftime("%Y%m%d")
            jcds = sel_jcds
            prog = st.progress(0.0)
            status = st.empty()
            hits = []            # 勝率しきい値以上の買い目（明細）
            race_records = []    # レース単位の予想・結果（回収率/結果表示）
            n_races = 0
            n_boat1_filtered = 0  # 1号艇フィルタで除外したレース数

            def _t3_fetch_race(jcd, rno):
                try:
                    if ev_mode:
                        res = fetch_race_data_hybrid(dstr3, jcd, rno)   # 5要素 (…, sources)
                    else:
                        base = fetch_race_data(dstr3, jcd, rno)
                        res = (base + ({},)) if base else None          # sources なしは空辞書
                except Exception:
                    res = None
                time.sleep(0.15)   # 取得先への節度（ワーカー毎）
                return jcd, rno, res

            # --- フェーズ1: 各場の1Rを並列取得して開催場を判定（非開催場は以降スキップ） ---
            race_data = {}       # (jcd, rno) -> fetch_race_data の結果
            status.write("開催場を確認中…（各場の1Rを取得）")
            probe_jcds = list(jcds)
            for _retry in range(2):   # 一時的な取得失敗で開催場を取り逃がさないよう1回だけ再試行
                if not probe_jcds:
                    break
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futs = [ex.submit(_t3_fetch_race, j, 1) for j in probe_jcds]
                    for i, f in enumerate(as_completed(futs), 1):
                        jcd, rno, res = f.result()
                        if res:
                            race_data[(jcd, rno)] = res
                        prog.progress(i / len(futs))
                probe_jcds = [j for j in probe_jcds if (j, 1) not in race_data]
            live_jcds = [j for j in jcds if (j, 1) in race_data]
            n_dark_jcds = len(jcds) - len(live_jcds)

            # --- フェーズ2: 開催場の2〜12Rを並列取得 ---
            if live_jcds:
                tasks = [(j, r) for j in live_jcds for r in range(2, 13)]
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futs = [ex.submit(_t3_fetch_race, j, r) for j, r in tasks]
                    for i, f in enumerate(as_completed(futs), 1):
                        jcd, rno, res = f.result()
                        if res:
                            race_data[(jcd, rno)] = res
                        status.write(f"レース取得中… {i}/{len(futs)}（開催 {len(live_jcds)} 場）")
                        prog.progress(i / len(futs))

            # --- フェーズ3: コース別補正用データを並列取得（同一セッション内はキャッシュ再利用） ---
            regno_cache = {}     # jcd -> {rno: {lane: 登録番号}}
            rate_cache = {}      # 登録番号 -> {course: {'r2','r3'}}（このスキャンで使う分）
            if use_course and course_k > 0 and live_jcds:
                ss_regno = st.session_state.setdefault("t3_regno_cache", {})  # (date, jcd) -> regno_map
                ss_rates = st.session_state.setdefault("t3_rate_cache", {})   # (date, regno) -> rates

                def _t3_fetch_regnos(jcd):
                    try:
                        res = fetch_racelist_regnos(dstr3, jcd)
                    except Exception:
                        res = {}
                    time.sleep(0.15)
                    return jcd, res

                def _t3_fetch_rates(reg):
                    try:
                        res = fetch_course_rates(dstr3, reg)
                    except Exception:
                        res = {}
                    time.sleep(0.15)
                    return reg, res

                miss_jcds = [j for j in live_jcds if (dstr3, j) not in ss_regno]
                if miss_jcds:
                    status.write(f"出走表（登録番号）取得中… {len(miss_jcds)} 場")
                    with ThreadPoolExecutor(max_workers=n_workers) as ex:
                        futs = [ex.submit(_t3_fetch_regnos, j) for j in miss_jcds]
                        for i, f in enumerate(as_completed(futs), 1):
                            jcd, res = f.result()
                            ss_regno[(dstr3, jcd)] = res
                            prog.progress(i / len(futs))
                regno_cache = {j: ss_regno.get((dstr3, j), {}) for j in live_jcds}

                # 取得できたレースに実際に出走する選手だけを対象に、未キャッシュ分のみ取得
                needed = set()
                for (jcd, rno) in race_data:
                    for lane in range(1, 7):
                        reg = regno_cache.get(jcd, {}).get(rno, {}).get(lane)
                        if reg:
                            needed.add(reg)
                miss_regs = [r for r in needed if (dstr3, r) not in ss_rates]
                if miss_regs:
                    with ThreadPoolExecutor(max_workers=n_workers) as ex:
                        futs = [ex.submit(_t3_fetch_rates, r) for r in miss_regs]
                        for i, f in enumerate(as_completed(futs), 1):
                            reg, res = f.result()
                            ss_rates[(dstr3, reg)] = res
                            status.write(f"選手データ取得中… {i}/{len(miss_regs)}"
                                         f"（キャッシュ再利用 {len(needed) - len(miss_regs)} 件）")
                            prog.progress(i / len(futs))
                rate_cache = {r: ss_rates.get((dstr3, r), {}) for r in needed}

            # --- フェーズ4: 解析（取得済みデータをモデルで評価。通信なし） ---
            status.write("解析中…")
            for jcd in live_jcds:
                regno_map = regno_cache.get(jcd, {}) if (use_course and course_k > 0) else {}
                for rno in range(1, 13):
                    base_res = race_data.get((jcd, rno))
                    if not base_res:
                        continue
                    racers, lane_to_rank, odds_map3, payoff, sources = base_res
                    n_races += 1
                    try:
                        feat_df = make_race_features(racers)
                        combo_probs = predict_combo_probs(feat_df, jcd)
                    except Exception:
                        continue

                    # コース別連対率でモデル確率を後段補正（進入コースの2連対率/3連対率）
                    if use_course and course_k > 0 and regno_map.get(rno):
                        course_by_lane = {int(rw["lane"]): int(rw["course_in"]) for rw in racers}
                        r2_by_lane, r3_by_lane = {}, {}
                        for lane in range(1, 7):
                            reg = regno_map[rno].get(lane)
                            rates = rate_cache.get(reg, {}) if reg else {}
                            cr = rates.get(course_by_lane.get(lane, lane), {})
                            r2_by_lane[lane] = cr.get("r2")
                            r3_by_lane[lane] = cr.get("r3")
                        combo_probs = apply_course_adjustment(combo_probs, r2_by_lane, r3_by_lane, course_k)

                    # 風・波（公式直前情報）による荒れ水面補正
                    wx = sources.get("weather", {}) if isinstance(sources, dict) else {}
                    if weather_k > 0 and wx:
                        cmap = {int(rw["lane"]): int(rw.get("course_in", rw["lane"])) for rw in racers}
                        combo_probs = apply_weather_adjustment(
                            combo_probs, wx.get("風速(m)"), wx.get("波高(cm)"), weather_k, cmap)

                    # 1号艇の1着率(モデル予想)による絞り込み（サイドバーの下限以上のレースのみ抽出）
                    if boat1_min > 0:
                        p1_boat1 = sum(p for c, p in combo_probs.items()
                                       if c.split("-")[0] == "1")
                        if p1_boat1 * 100 < boat1_min:
                            n_boat1_filtered += 1
                            n_races -= 1
                            continue

                    # 買い目選定: EVモード（オッズが取れたレース）は EV ベース、
                    # それ以外は従来どおり勝率しきい値ベース。
                    if ev_mode and odds_map3 and len(odds_map3) >= 20:
                        picks = select_bets_by_ev(combo_probs, odds_map3,
                                                  ev_th, top_n_prob, max_bets)
                        buys = [(pk["bet"], pk["prob"]) for pk in picks]
                        for pk in picks:
                            hits.append({
                                "場": JCD_NAME[jcd],
                                "R": rno,
                                "買い目": pk["bet"],
                                "勝率(%)": round(pk["prob"] * 100, 2),
                                "オッズ": pk["odds"],
                                "EV": round(pk["ev"], 2),
                            })
                    else:
                        buys = sorted(
                            [(c, p) for c, p in combo_probs.items() if p * 100 >= win_th],
                            key=lambda x: x[1], reverse=True,
                        )
                        for combo, p in buys:
                            hits.append({
                                "場": JCD_NAME[jcd],
                                "R": rno,
                                "買い目": combo,
                                "勝率(%)": round(p * 100, 2),
                            })

                    # 3連単の結果（着順→組番）と払戻
                    result = None
                    if lane_to_rank:
                        r1 = next((l for l, rk in lane_to_rank.items() if rk == 1), None)
                        r2 = next((l for l, rk in lane_to_rank.items() if rk == 2), None)
                        r3 = next((l for l, rk in lane_to_rank.items() if rk == 3), None)
                        if r1 and r2 and r3:
                            result = f"{r1}-{r2}-{r3}"
                    settled = (result is not None) and (payoff is not None) and (payoff > 0)

                    buy_combos = [c for c, _ in buys]
                    hit = bool(settled and (result in buy_combos))
                    inv = len(buy_combos) * bet_amt
                    ret = payoff * (bet_amt / 100.0) if hit else 0
                    race_records.append({
                        "place": JCD_NAME[jcd], "jcd": jcd, "rno": rno,
                        "buys": buy_combos, "n_buys": len(buy_combos),
                        "result": result, "payoff": payoff,
                        "settled": settled, "hit": hit, "inv": inv, "ret": ret,
                    })
            prog.empty()
            status.empty()

            bet_records  = [r for r in race_records if r["n_buys"] > 0]
            settled_bets = [r for r in bet_records if r["settled"]]
            pending_bets = [r for r in bet_records if not r["settled"]]
            n_skip       = n_races - len(bet_records)

            st.success(f"解析完了 — 開催 {len(live_jcds)} 場（非開催スキップ {n_dark_jcds} 場） / "
                       f"取得 {n_races:,} レース / 買い目あり {len(bet_records):,} レース"
                       f"（見送り {n_skip:,}） / 買い目 {len(hits):,} 点"
                       + (f"（EV≥{ev_th:.2f} で選定）" if ev_mode else f"（勝率≥{win_th:.1f}% で選定）"))
            if ev_mode:
                st.caption(f"EVモード: ON — 公式3連単オッズ×モデル確率で EV≥{ev_th:.2f}・"
                           f"確率上位{top_n_prob}点・1R最大{max_bets}点（サイドバー設定）。"
                           "オッズが取れないレース（販売前など）は勝率しきい値で代替選定。"
                           + (f" 風・波補正 k={weather_k:.1f}。" if weather_k > 0 else ""))
            if use_course and course_k > 0:
                n_regno_ok = sum(1 for v in regno_cache.values() if v)
                st.caption(f"コース別補正: ON（k={course_k:.1f}・進入コースの連対率で増幅） / "
                           f"登録番号を取得できた場 {n_regno_ok}/{len(live_jcds)} / 使用した選手データ {len(rate_cache):,} 件。"
                           "取得0はその箇所だけ無補正です（場名ローマ字・未掲載・選手ページ未取得が原因のことがあります）。")
            else:
                st.caption("コース別補正: OFF（モデル素の確率で表示）。")
            if boat1_min > 0:
                st.caption(f"1号艇フィルタ: ON（モデル予想の1号艇1着率 ≥ {boat1_min:.0f}% のレースのみ抽出） / "
                           f"条件を満たさず除外したレース {n_boat1_filtered:,} 件。")

            if not hits:
                if ev_mode:
                    st.info(f"EV≥{ev_th:.2f} を満たす買い目は見つかりませんでした。"
                            "（EV閾値を下げる・上位N点を増やすと候補は増えますが、閾値を下げるほど"
                            "理論上の優位性は薄れます。販売前でオッズ未確定の場合も出ません）")
                else:
                    st.info(f"勝率{win_th:.1f}%以上の買い目は見つかりませんでした。"
                            "（締切前で展示タイム未反映だと確率が割れて高勝率の買い目が出にくいことがあります）")
            else:
                # --- 当日回収率（結果が確定したレースのみで集計） ---
                st.subheader("💰 当日回収率（結果確定レースのみ）")
                if settled_bets:
                    tot_inv  = sum(r["inv"] for r in settled_bets)
                    tot_ret  = sum(r["ret"] for r in settled_bets)
                    n_hit    = sum(1 for r in settled_bets if r["hit"])
                    ret_rate = tot_ret / tot_inv * 100 if tot_inv > 0 else 0
                    hit_rate = n_hit / len(settled_bets) * 100
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("対象レース", f"{len(settled_bets):,}", f"未確定 {len(pending_bets):,}")
                    m2.metric("回収率", f"{ret_rate:.1f}%",
                              f"投資{int(tot_inv):,} / 回収{int(tot_ret):,}円")
                    m3.metric("的中率", f"{hit_rate:.1f}%", f"{n_hit}/{len(settled_bets)}")
                    m4.metric("買い目数", f"{sum(r['n_buys'] for r in settled_bets):,}",
                              f"1点 {int(bet_amt):,}円")
                    if ret_rate >= 100:
                        st.success(f"🎉 回収率 {ret_rate:.1f}% — 結果確定 {len(settled_bets)} レースでの集計です。")
                    else:
                        st.warning(f"回収率 {ret_rate:.1f}% — 結果確定 {len(settled_bets)} レースでの集計です。")
                else:
                    st.info("結果が確定したレースがまだありません（回収率は結果確定後に算出されます）。")

                # --- レース別 結果（買い目ありレース） ---
                st.subheader("📋 レース別 結果（予想 vs 結果）")
                res_rows = []
                for r in sorted(bet_records, key=lambda x: (x["jcd"], x["rno"])):
                    if r["settled"]:
                        res_rows.append({
                            "場": r["place"], "R": r["rno"],
                            "予想買い目": ",".join(r["buys"]),
                            "点数": r["n_buys"],
                            "結果": r["result"],
                            "払戻(円)": f"{int(r['payoff']):,}",
                            "判定": "🎯 的中" if r["hit"] else "❌ 外れ",
                            "投資(円)": f"{int(r['inv']):,}",
                            "回収(円)": f"{int(r['ret']):,}",
                        })
                    else:
                        res_rows.append({
                            "場": r["place"], "R": r["rno"],
                            "予想買い目": ",".join(r["buys"]),
                            "点数": r["n_buys"],
                            "結果": "—",
                            "払戻(円)": "—",
                            "判定": "未確定",
                            "投資(円)": f"{int(r['inv']):,}",
                            "回収(円)": "—",
                        })
                st.dataframe(pd.DataFrame(res_rows), use_container_width=True)
                st.caption("「結果」「払戻」は結果が確定したレースのみ表示。回収率・的中率は確定レースだけで集計しています。")

                # --- 勝率しきい値以上の買い目 明細 ---
                st.subheader("🎯 買い目 明細" + (f"（EV≥{ev_th:.2f}）" if ev_mode
                                                  else f"（勝率≥{win_th:.1f}%）"))
                df_hits = pd.DataFrame(hits)
                sort_col = "EV" if "EV" in df_hits.columns else "勝率(%)"
                df_hits = (df_hits.sort_values([sort_col, "場", "R"],
                                               ascending=[False, True, True])
                           .reset_index(drop=True))
                st.dataframe(df_hits, use_container_width=True)
                st.caption("EVモードON: EV（モデル確率×公式オッズ）の高い順。"
                           "OFF: 勝率の高い順（オッズ未取得のためEVなし）。")
