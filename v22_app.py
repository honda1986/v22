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
import streamlit.components.v1 as components
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


def fetch_race_data_hybrid(date: str, jcd: int, rno: int,
                           pre_base=None, pre_bi=None, pre_odds=None):
    """ハイブリッド取得:
       1) kyotei.fun から出走表・選手成績・初期オッズ・結果を取得
       2) 公式の直前情報で展示・コースINを上書き (取れたら)
       3) 公式のオッズで上書き (取れたら)
       返り値: (racers, lane_to_rank, odds_map, payoff, sources)
        sources は何が公式から取れたかのフラグ辞書
       pre_base/pre_bi/pre_odds: 呼び出し側で並列取得済みのデータを渡すと
        該当の取得（と待機）をスキップする（省略時は従来どおり直列取得）。"""
    sources = {"odds_from_official": False, "tenji_from_official": False,
               "course_in_from_official": False, "weather": {}}

    base_res = pre_base if pre_base is not None else fetch_race_data(date, jcd, rno)
    if not base_res:
        return None
    racers, lane_to_rank, odds_map, payoff = base_res

    # 公式直前情報
    if pre_bi is not None:
        bi = pre_bi
    else:
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
        if pre_odds is not None:
            official_odds = pre_odds
        else:
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
# レース展開シミュレーター (タブ4) — モデルの3連単確率から展開を再現
# ============================================================
_SIM_HTML = r'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  :root { --gold:#d8b15c; --bg:#0b0f1e; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:#e8e6df;
         font-family:"Hiragino Kaku Gothic ProN","Noto Sans JP",Meiryo,sans-serif; }
  .wrap { max-width:790px; margin:0 auto; padding:8px 10px 28px; }
  .hd { font-size:12px; color:#cdb87e; letter-spacing:2px; }
  .ttl { font-size:22px; font-weight:800; color:#fff; margin:2px 0 0; }
  .ttl small { font-size:12px; color:#cdb87e; font-weight:600; letter-spacing:2px; margin-left:8px; }
  .sliderRow { display:flex; justify-content:space-between; align-items:baseline;
               font-size:11px; color:#8d93a8; margin-top:12px; }
  .pct { color:var(--gold); font-weight:800; font-size:15px; }
  input[type=range] { width:100%; accent-color:var(--gold); margin:4px 0 2px; }
  canvas { width:100%; display:block; border-radius:14px; margin-top:10px; background:#0a0e1c; }
  .btnRow { display:flex; gap:10px; margin:14px 0 10px; align-items:center; flex-wrap:wrap; }
  button { cursor:pointer; border:none; font-family:inherit; }
  .btnStart { background:#e2403f; color:#fff; font-size:17px; font-weight:800;
              padding:12px 30px; border-radius:999px; min-width:150px; }
  .btnVerify { background:#141a30; border:1.5px solid var(--gold); color:#f0e3bd;
               font-size:15px; font-weight:700; padding:12px 20px; border-radius:14px; }
  .btnSpd { background:#1a2140; color:#aeb6cf; font-size:12px; font-weight:700;
            padding:8px 12px; border-radius:10px; }
  .phase { font-size:12px; color:#8d93a8; min-height:16px; margin:2px 2px 6px; }
  .rrow { display:flex; align-items:center; gap:12px; padding:9px 10px;
          border-radius:10px; margin-bottom:4px; background:rgba(255,255,255,0.025); }
  .rrow .pos { width:22px; font-size:20px; font-weight:800; color:#777f96; text-align:center; }
  .rrow.p1 .pos { color:#fff; }
  .badge { width:30px; height:30px; border-radius:50%; display:flex; align-items:center;
           justify-content:center; font-weight:800; font-size:14px; flex:none;
           border:1.5px solid rgba(255,255,255,.25); }
  .mk { width:22px; text-align:center; font-size:15px; }
  .nm { font-size:16px; font-weight:700; }
  .sub { font-size:11px; color:#8d93a8; margin-left:auto; text-align:right; }
  .sampled { margin:6px 2px 0; font-size:13px; color:#cdb87e; min-height:18px; }
  h3 { font-size:15px; color:#f0e3bd; border-left:3px solid var(--gold);
       padding-left:8px; margin:22px 0 10px; }
  .vbarRow { display:flex; align-items:center; gap:8px; margin-bottom:6px; font-size:12px; }
  .vbarBox { flex:1; background:#1a2140; border-radius:6px; height:18px; overflow:hidden; position:relative; }
  .vbar { height:100%; border-radius:6px; }
  .vmodel { position:absolute; top:0; bottom:0; width:2px; background:#fff; opacity:.85; }
  .vlab { width:128px; text-align:right; color:#aeb6cf; flex:none; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; margin-top:6px; }
  th, td { padding:6px 8px; text-align:right; border-bottom:1px solid #1d2547; }
  th { color:#8d93a8; font-weight:600; }
  td:first-child, th:first-child { text-align:left; font-weight:700; }
  tr.hitRow td { color:#ffd97a; }
  .resNote { font-size:12.5px; color:#aeb6cf; margin-top:8px; }
  .hidden { display:none; }
</style></head>
<body><div class="wrap">
  <div class="hd" id="hd"></div>
  <div class="ttl" id="ttl"></div>
  <div class="sliderRow"><span>展示・乱数のみ</span>
    <span>予想反映度 <span class="pct" id="pct">65%</span></span>
    <span>モデル確率重視</span></div>
  <input type="range" id="alpha" min="0" max="100" value="65">
  <canvas id="cv" width="780" height="442"></canvas>
  <div class="btnRow">
    <button class="btnStart" id="bStart">▶ レース開始</button>
    <button class="btnVerify" id="bVerify">📊 1000回検証（65%）</button>
    <button class="btnSpd" id="bSpd">速度 ×1</button>
  </div>
  <div class="phase" id="phase">▶ レース開始で、ピットアウト→待機行動（進入）→大時計→フライングスタート→3周1800mを再現します。着順はモデル確率から抽選。</div>
  <div id="verify" class="hidden">
    <h3 id="vTitle"></h3>
    <div id="vWin"></div>
    <h3>3連単 出現上位10（シミュレーション vs モデル理論値）</h3>
    <div id="vCombo"></div>
    <div class="resNote" id="vRes"></div>
  </div>
  <div id="ranks"></div>
  <div class="sampled" id="sampled"></div>
</div>
<script>
const DATA = __DATA__;
const KEYS = Object.keys(DATA.combos);
let PSUM = 0; KEYS.forEach(k => PSUM += DATA.combos[k]);
const MODEL = {}; KEYS.forEach(k => MODEL[k] = PSUM > 0 ? DATA.combos[k] / PSUM : 1 / KEYS.length);

// 各艇の周辺確率（1着・2着・3着）と評価・予想印
const W1 = {1:0,2:0,3:0,4:0,5:0,6:0}, W2 = {...W1}, W3 = {...W1};
KEYS.forEach(k => { const a = k.split("-").map(Number);
  W1[a[0]] += MODEL[k]; W2[a[1]] += MODEL[k]; W3[a[2]] += MODEL[k]; });
const EVALS = {};
for (let l = 1; l <= 6; l++) EVALS[l] = (3 * W1[l] + 2 * W2[l] + W3[l]) / 3 * 100;
const ORDER_BY_EVAL = [1,2,3,4,5,6].sort((a,b) => EVALS[b] - EVALS[a]);
const MARKS = {}; ["◎","○","▲","△","✕",""].forEach((m,i) => MARKS[ORDER_BY_EVAL[i]] = m);

const COL = {1:["#f2f2f2","#1a1a1a"],2:["#2e2e2e","#fff"],3:["#e2403f","#fff"],
             4:["#3f7fe0","#fff"],5:["#f0c93f","#1a1a1a"],6:["#3fae6a","#fff"]};
const BOATS = DATA.boats.slice().sort((a,b) => a.lane - b.lane);
BOATS.forEach(b => { if (!(b.course >= 1 && b.course <= 6)) b.course = b.lane; });
const NAME = {}; BOATS.forEach(b => NAME[b.lane] = b.lane + "号艇（" + b.cls + "）");

document.getElementById("hd").textContent =
  DATA.date.slice(0,4) + "." + DATA.date.slice(4,6) + "." + DATA.date.slice(6,8) +
  "　" + DATA.place + "　" + DATA.rno + "R　3周 1800m";
document.getElementById("ttl").innerHTML =
  DATA.place + " " + DATA.rno + "R <small>競艇 予想検証シミュレーター</small>";

// ---------- 確率ブレンド＆サンプリング ----------
function blendCum(alpha) {
  const u = 1 / KEYS.length, ps = [], cum = [];
  let t = 0;
  KEYS.forEach(k => { const p = alpha * MODEL[k] + (1 - alpha) * u; ps.push(p); t += p; });
  let c = 0;
  for (let i = 0; i < ps.length; i++) { c += ps[i] / t; cum.push(c); }
  return cum;
}
function sampleCombo(cum) {
  const r = Math.random();
  let lo = 0, hi = cum.length - 1;
  while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] < r) lo = m + 1; else hi = m; }
  return KEYS[lo];
}
function sampleOrder(cum) {
  const top = sampleCombo(cum).split("-").map(Number);
  const rest = [1,2,3,4,5,6].filter(l => !top.includes(l));
  while (rest.length) {
    const ws = rest.map(l => W1[l] + 0.02);
    let t = 0; ws.forEach(w => t += w);
    let r = Math.random() * t, i = 0;
    for (; i < rest.length - 1; i++) { r -= ws[i]; if (r <= 0) break; }
    top.push(rest.splice(i, 1)[0]);
  }
  return top;
}
function gauss() { let u = 0, v = 0;
  while (u === 0) u = Math.random(); while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

// ---------- コース幾何（左回り＝画面上は反時計回り。ホーム直線を左→右、1Mは右） ----------
const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
const CW = 780, CH = 442;
const R = 90, SL = 380, xL = 188, xR = xL + SL, cy = 188;
const yT = cy - R, yB = cy + R;                       // 上=バック直線 / 下=ホーム直線
const PER = 2 * SL + 2 * Math.PI * R;                 // 1周の周長(px)
const LINEX = xL + 168;                               // スタート/ゴールライン
const ULINE = (LINEX - xL) / PER;
function pos(u) {  // u: 周回の小数部。u=0 は (xL,yB) を右向き
  let d = ((u % 1) + 1) % 1 * PER;
  if (d < SL) return { x: xL + d, y: yB, ang: 0 };                       // ホーム直線 →
  d -= SL;
  if (d < Math.PI * R) { const a = Math.PI / 2 - d / R;                   // 1M(右)を左旋回
    return { x: xR + R * Math.cos(a), y: cy + R * Math.sin(a), ang: a - Math.PI / 2 }; }
  d -= Math.PI * R;
  if (d < SL) return { x: xR - d, y: yT, ang: Math.PI };                  // バック直線 ←
  d -= SL;
  const a = -Math.PI / 2 - d / R;                                         // 2M(左)を左旋回
  return { x: xL + R * Math.cos(a), y: cy + R * Math.sin(a), ang: a - Math.PI / 2 };
}

// ---------- 進行スケジュール ----------
const T_PIT_END = 4.6, T_FORM = 5.6, T_DASH = 8.6, T0 = 11.5, RACE_T = 20.0;

function smooth(s) { return s <= 0 ? 0 : s >= 1 ? 1 : s * s * (3 - 2 * s); }
function buildScript(b) {     // ピット(左下)→2Mの周りを「逆時計回り」に回り込む→待機隊形
  const lane = b.lane, c = b.course;
  b.slot = { x: xL - 150 + (lane - 1) * 21, y: yB + 42 };     // ピットは左下(2マーク後方)
  b.t0 = 0.3 + lane * 0.16;
  b.r2m = 34 + lane * 5;                                       // 旋回半径(艇ごとにずらして重なり防止)
  b.tArc0 = b.t0 + 0.8;                                        // 旋回開始
  b.tArc1 = b.tArc0 + 2.6;                                     // 旋回終了
  b.tForm = T_FORM + c * 0.16;                                 // 待機隊形完成
  b.wx = c <= 3 ? LINEX - 104 - c * 9 : LINEX - 178 - (c - 3) * 17;   // スロー / ダッシュ(深め)
  b.wy = yB + (c - 3.5) * 11;
}
const PH0 = 70 * Math.PI / 180, PH1 = -270 * Math.PI / 180;    // 2M下側→右→上→左→下と逆時計回りに一周
function scriptPos(b, t) {
  const ex = xL + b.r2m * Math.cos(PH0), ey = cy + b.r2m * Math.sin(PH0);
  if (t <= b.t0) return { x: b.slot.x, y: b.slot.y };
  if (t <= b.tArc0) {                                          // ピット→旋回入口(2M真上付近)
    const s = smooth((t - b.t0) / (b.tArc0 - b.t0));
    return { x: b.slot.x + (ex - b.slot.x) * s, y: b.slot.y + (ey - b.slot.y) * s };
  }
  if (t <= b.tArc1) {                                          // 円弧: 角度を単調減少=逆時計回り
    const s = smooth((t - b.tArc0) / (b.tArc1 - b.tArc0));
    const ph = PH0 + (PH1 - PH0) * s;
    return { x: xL + b.r2m * Math.cos(ph), y: cy + b.r2m * Math.sin(ph) };
  }
  const sx = xL + b.r2m * Math.cos(PH1), sy = cy + b.r2m * Math.sin(PH1);  // 2M南側に抜け右向きへ
  const s = smooth(Math.min(1, (t - b.tArc1) / (b.tForm - b.tArc1)));
  return { x: sx + (b.wx - sx) * s, y: sy + (b.wy - sy) * s };
}

// ---------- 描画 ----------
function roundRect(x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}
function drawPond(t, label, lap) {
  ctx.clearRect(0, 0, CW, CH);
  const g = ctx.createLinearGradient(0, 56, 0, 344);
  g.addColorStop(0, "#0d2438"); g.addColorStop(1, "#0f2f4c");
  ctx.fillStyle = g; roundRect(22, 56, CW - 44, 290, 54); ctx.fill();
  ctx.strokeStyle = "#1c3a5c"; ctx.lineWidth = 2; roundRect(22, 56, CW - 44, 290, 54); ctx.stroke();
  // 走行ライン（うっすら）
  ctx.setLineDash([5, 9]); ctx.strokeStyle = "rgba(255,255,255,0.07)"; ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i <= 220; i++) { const p = pos(i / 220); i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y); }
  ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
  // スタート/ゴールライン（ホーム直線を横切る）
  ctx.setLineDash([6, 5]); ctx.strokeStyle = "#e2403f"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(LINEX, cy + 36); ctx.lineTo(LINEX, 340); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = "#e2403f"; ctx.font = "bold 10px sans-serif"; ctx.textAlign = "center";
  ctx.fillText("START / GOAL", LINEX, 356);
  // ターンマーク（1M=右 / 2M=左）と小回り防止ブイ
  [[xR, cy, "1M"], [xL, cy, "2M"]].forEach(m => {
    ctx.beginPath(); ctx.arc(m[0], m[1], 8, 0, 7); ctx.fillStyle = "#e2403f"; ctx.fill();
    ctx.beginPath(); ctx.arc(m[0], m[1], 3.5, 0, 7); ctx.fillStyle = "#fff"; ctx.fill();
    ctx.fillStyle = "#ffd97a"; ctx.font = "bold 11px sans-serif";
    ctx.fillText(m[2], m[0], m[1] - 14);
  });
  ctx.beginPath(); ctx.arc(xL + 26, cy, 4, 0, 7); ctx.fillStyle = "#9aa0b5"; ctx.fill();
  ctx.fillStyle = "#6d7590"; ctx.font = "9px sans-serif";
  ctx.fillText("小回り防止ブイ", xL + 42, cy + 18);
  // ピット（左下・2マーク後方）
  ctx.fillStyle = "#27314f"; roundRect(xL - 162, yB + 30, 142, 24, 6); ctx.fill();
  ctx.fillStyle = "#8d93a8"; ctx.font = "bold 10px sans-serif"; ctx.fillText("PIT", xL - 91, yB + 68);
  // 場名・フェーズ
  ctx.fillStyle = "#cdb87e"; ctx.font = "bold 14px sans-serif"; ctx.fillText(DATA.place, CW / 2, cy - 38);
  ctx.fillStyle = "#6d7590"; ctx.font = "11px sans-serif"; ctx.fillText("3周 1800m・左回り", CW / 2, cy - 20);
  if (label) {
    ctx.font = "bold 14px sans-serif";
    const w = ctx.measureText(label).width + 56;
    ctx.fillStyle = "#141a36"; roundRect(CW / 2 - w / 2, cy - 6, w, 30, 8); ctx.fill();
    ctx.strokeStyle = "#2a3565"; ctx.lineWidth = 1; roundRect(CW / 2 - w / 2, cy - 6, w, 30, 8); ctx.stroke();
    ctx.fillStyle = "#ffd97a"; ctx.fillText(label, CW / 2, cy + 14);
  }
  if (lap) { ctx.fillStyle = "#8d93a8"; ctx.font = "bold 12px sans-serif"; ctx.fillText(lap, CW / 2, cy + 42); }
  drawClock(t);
}
function drawClock(t) {  // 大時計: ピットアウトから動き出し、針が真上(0)を指した瞬間がスタート
  const x = LINEX + 86, y = 398, r = 28;
  ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fillStyle = "#101730"; ctx.fill();
  ctx.lineWidth = 2.5; ctx.strokeStyle = "#d8b15c"; ctx.stroke();
  for (let i = 0; i < 12; i++) { const a = i / 12 * 2 * Math.PI;
    ctx.beginPath();
    ctx.moveTo(x + Math.cos(a) * (r - 5), y + Math.sin(a) * (r - 5));
    ctx.lineTo(x + Math.cos(a) * (r - 2), y + Math.sin(a) * (r - 2));
    ctx.strokeStyle = "#5a6587"; ctx.lineWidth = 1.5; ctx.stroke(); }
  const frac = Math.min(t / T0, 1);
  const a = -Math.PI / 2 + frac * 2 * Math.PI;
  ctx.beginPath(); ctx.moveTo(x, y);
  ctx.lineTo(x + Math.cos(a) * (r - 7), y + Math.sin(a) * (r - 7));
  ctx.strokeStyle = "#e2403f"; ctx.lineWidth = 2.5; ctx.stroke();
  ctx.beginPath(); ctx.arc(x, y, 2.5, 0, 7); ctx.fillStyle = "#e2403f"; ctx.fill();
  ctx.fillStyle = "#8d93a8"; ctx.font = "9px sans-serif"; ctx.textAlign = "center";
  ctx.fillText("大時計", x, y + r + 11);
  if (t < T0 && T0 - t <= 5.5) {
    ctx.fillStyle = "#ffd97a"; ctx.font = "bold 15px sans-serif";
    ctx.fillText("スタートまで " + (T0 - t).toFixed(1), x + 116, y + 5);
  } else if (t >= T0 && t < T0 + 1.6) {
    ctx.fillStyle = "#ff6b5e"; ctx.font = "bold 17px sans-serif";
    ctx.fillText("スタート！", x + 96, y + 5);
  }
}
function drawBoat(b, x, y, ang, t) {
  ctx.strokeStyle = "rgba(255,255,255,0.18)"; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(x, y);
  ctx.lineTo(x - Math.cos(ang) * 19, y - Math.sin(ang) * 19); ctx.stroke();
  const c = COL[b.lane];
  ctx.beginPath(); ctx.arc(x, y, 12, 0, 7); ctx.fillStyle = c[0]; ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,.4)"; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.fillStyle = c[1]; ctx.font = "bold 12px sans-serif"; ctx.textAlign = "center";
  ctx.fillText(b.lane, x, y + 4);
  if (b.crossT && t >= b.crossT && t < b.crossT + 2.6) {     // STタイム表示
    ctx.fillStyle = "#ffd97a"; ctx.font = "bold 11px sans-serif";
    ctx.fillText("." + ("" + Math.round(b.st * 100)).padStart(2, "0"), x, y - 17);
  }
}

// ---------- レース ----------
let race = null, running = false, speed = 1, lastTs = 0;
const elPhase = document.getElementById("phase"), elRanks = document.getElementById("ranks");
const elSampled = document.getElementById("sampled");
const slider = document.getElementById("alpha"), elPct = document.getElementById("pct");
const bStart = document.getElementById("bStart"), bVerify = document.getElementById("bVerify");
const bSpd = document.getElementById("bSpd");
function alphaVal() { return slider.value / 100; }
function syncLabels() {
  elPct.textContent = slider.value + "%";
  bVerify.textContent = "📊 1000回検証（" + slider.value + "%）";
}
slider.oninput = syncLabels; syncLabels();

function newRace() {
  const cum = blendCum(alphaVal());
  const order = sampleOrder(cum);
  const rank = {}; order.forEach((l, i) => rank[l] = i + 1);
  BOATS.forEach(b => {
    b.rank = rank[b.lane];
    b.st = Math.min(0.45, Math.max(0.02,
      (b.avg_st || 0.16) + gauss() * 0.05 - (b.course >= 4 ? 0.015 : 0)));  // ダッシュ勢は助走で僅かに有利
    b.crossT = T0 + b.st;
    b.finT = T0 + RACE_T + (b.rank - 1) * (0.9 + Math.random() * 0.5) + Math.random() * 0.25;
    b.amp = (3.5 - b.rank) * 0.02 + (Math.random() - 0.5) * 0.045;          // 1周1Mで概ね決着
    b.p = 0; b.done = false;
    buildScript(b);
  });
  return { t: 0, order, goal: false };
}
function boatXY(b, t) {
  if (t < T_DASH) {                                          // ピットアウト〜待機行動
    let q = scriptPos(b, t);
    if (t > T_FORM && b.course >= 4)                          // ダッシュ勢は助走距離を取りに下がる
      q.x -= smooth((t - T_FORM) / (T_DASH - T_FORM)) * 34;
    const q2 = scriptPos(b, t + 0.08);                        // 進行方向を向く
    const ang = (Math.abs(q2.x - q.x) + Math.abs(q2.y - q.y)) > 0.3
      ? Math.atan2(q2.y - q.y, q2.x - q.x) : 0;
    return { x: q.x, y: q.y, ang: ang };
  }
  if (t < b.crossT) {                                         // 助走〜スタートライン通過
    const x0 = b.wx - (b.course >= 4 ? 34 : 0);
    const s = (t - T_DASH) / (b.crossT - T_DASH);
    return { x: x0 + (LINEX - x0) * Math.pow(Math.max(0, s), 1.7), y: b.wy, ang: 0 };
  }
  const s = Math.min((t - b.crossT) / (b.finT - b.crossT), 1.02);  // 本走: 3周
  b.p = Math.min(3.04, 3 * (Math.min(s, 1.02) + b.amp * Math.sin(Math.PI * Math.min(s, 1))));
  if (s >= 1) { b.p = Math.max(b.p, 3.0); b.done = true; }
  const u = ULINE + b.p;                                       // u = ライン位置 + 周回数
  const q = pos(u);
  const off = (b.course - 3.5) * 9;                            // 1コースが最内
  const nx = Math.cos(q.ang + Math.PI / 2), ny = Math.sin(q.ang + Math.PI / 2);
  return { x: q.x + nx * off, y: q.y + ny * off, ang: q.ang };
}
function phaseText(t, maxp, goal) {
  if (goal) return "🏁 ゴール！";
  if (t < T_PIT_END) return "ピットアウト — 小回り防止ブイ・2マークを左回りに旋回";
  if (t < T_DASH) return "待機行動 — コース取り（進入確定）";
  if (t < T0) return "助走 — 大時計0でスタートライン通過（フライングスタート方式）";
  if (maxp < 0.28) return "1周1マーク — ここで展開がほぼ決まる！";
  const lap = Math.min(3, Math.floor(maxp) + 1);
  return lap >= 3 ? "最終周" : lap + "周目";
}
function renderRanks(list, mode) {
  elRanks.innerHTML = list.map((b, i) => {
    const c = COL[b.lane];
    const sub = mode === "entry"
      ? "進入 " + b.course + "コース（" + (b.course <= 3 ? "スロー" : "ダッシュ") + "）・平均ST " + (b.avg_st || 0).toFixed(2)
      : (b.crossT ? "ST ." + ("" + Math.round(b.st * 100)).padStart(2, "0") + "　" : "") +
        "モデル1着率 " + (W1[b.lane] * 100).toFixed(1) + "%";
    return '<div class="rrow' + (i === 0 && mode === "race" ? ' p1' : '') + '">' +
      '<div class="pos">' + (mode === "entry" ? b.course : i + 1) + '</div>' +
      '<div class="badge" style="background:' + c[0] + ';color:' + c[1] + '">' + b.lane + '</div>' +
      '<div class="mk">' + (MARKS[b.lane] || "") + '</div>' +
      '<div class="nm" style="color:' + (b.lane === 2 ? "#cfd3e0" : c[0]) + '">' + NAME[b.lane] + '</div>' +
      '<div class="sub">' + sub + '</div></div>';
  }).join("");
}
function frame(ts) {
  if (!running) return;
  if (!lastTs) lastTs = ts;
  const dt = Math.min(0.05, (ts - lastTs) / 1000) * speed;
  lastTs = ts;
  race.t += dt;
  const t = race.t;
  const ps = BOATS.map(b => ({ b, q: boatXY(b, t) }));
  const maxp = Math.max(...BOATS.map(b => b.p));
  if (BOATS.every(b => b.done)) race.goal = true;
  drawPond(t, phaseText(t, maxp, race.goal),
           t >= T0 && !race.goal ? "LAP " + Math.min(3, Math.floor(maxp) + 1) + " / 3" : "");
  ps.slice().sort((a, b2) => a.b.p - b2.b.p)
    .forEach(o => drawBoat(o.b, o.q.x, o.q.y, o.q.ang, t));
  if (t < T_DASH) {
    renderRanks(BOATS.slice().sort((a, b2) => a.course - b2.course), "entry");
    elPhase.textContent = "進入はスタート展示ベース（実進入と異なる場合があります）";
  } else {
    const key = b2 => b2.p > 0 ? b2.p * 1e6 : boatXY(b2, t).x - LINEX;
    const sorted = race.goal ? BOATS.slice().sort((a, b2) => a.rank - b2.rank)
                             : BOATS.slice().sort((a, b2) => key(b2) - key(a));
    renderRanks(sorted, "race");
    elPhase.textContent = "";
  }
  if (race.goal) {
    running = false;
    bStart.textContent = "▶ もう一度";
    const tri = race.order.slice(0, 3).join("-");
    elSampled.innerHTML = "今回の展開: <b style='color:#ffd97a'>" + tri + "</b>" +
      "（モデル確率 " + (MODEL[tri] * 100).toFixed(2) + "%）" +
      (DATA.result ? "　／　実際の結果: <b>" + DATA.result + "</b>" +
        (DATA.result === tri ? " 🎯 一致！" : "") : "");
    return;
  }
  requestAnimationFrame(frame);
}
bStart.onclick = () => {
  if (running) { running = false; bStart.textContent = "▶ 再開"; return; }
  if (!race || race.goal) { race = newRace(); elSampled.textContent = ""; }
  running = true; lastTs = 0; bStart.textContent = "⏸ 停止";
  requestAnimationFrame(frame);
};
bSpd.onclick = () => { speed = speed === 1 ? 2 : (speed === 2 ? 4 : 1); bSpd.textContent = "速度 ×" + speed; };

// ---------- 1000回検証 ----------
bVerify.onclick = () => {
  const a = alphaVal(), cum = blendCum(a), N = 1000;
  const win = {1:0,2:0,3:0,4:0,5:0,6:0}, cc = {};
  for (let i = 0; i < N; i++) {
    const k = sampleCombo(cum);
    win[+k[0]]++; cc[k] = (cc[k] || 0) + 1;
  }
  document.getElementById("verify").classList.remove("hidden");
  let wsum = 0; for (let l = 1; l <= 6; l++) wsum += win[l];
  document.getElementById("vTitle").textContent =
    "📊 1000回検証結果（予想反映度 " + slider.value + "%）— 各艇の1着回数（合計 " + wsum + "回）";
  const lanes = [1,2,3,4,5,6].sort((x, y) => win[y] - win[x]);
  const u = 1 / 6;
  document.getElementById("vWin").innerHTML = lanes.map(l => {
    const sim = win[l] / N * 100;
    const model = (a * W1[l] + (1 - a) * u) * 100;
    const c = COL[l];
    return '<div class="vbarRow">' +
      '<div class="badge" style="background:' + c[0] + ';color:' + c[1] + '">' + l + '</div>' +
      '<div class="mk">' + (MARKS[l] || "") + '</div>' +
      '<div class="vbarBox"><div class="vbar" style="width:' + Math.min(100, sim) +
      '%;background:' + c[0] + ';opacity:.85"></div>' +
      '<div class="vmodel" style="left:' + Math.min(100, model) + '%"></div></div>' +
      '<div class="vlab">' + win[l] + '回（' + sim.toFixed(1) + '% / 理論 ' + model.toFixed(1) + '%）</div></div>';
  }).join("") +
  '<div class="resNote">白い縦線＝ブレンド後の理論1着率。シミュレーションは乱数なので毎回少し変動します。</div>';
  const top = Object.keys(cc).sort((x, y) => cc[y] - cc[x]).slice(0, 10);
  document.getElementById("vCombo").innerHTML =
    "<table><tr><th>3連単</th><th>出現回数</th><th>シミュ%</th><th>モデル理論%</th></tr>" +
    top.map(k => '<tr' + (k === DATA.result ? ' class="hitRow"' : '') + '><td>' + k +
      (k === DATA.result ? " 🎯" : "") + '</td><td>' + cc[k] + '回</td><td>' +
      (cc[k] / N * 100).toFixed(1) + '%</td><td>' + (MODEL[k] * 100).toFixed(2) + '%</td></tr>').join("") +
    "</table>";
  document.getElementById("vRes").textContent = (DATA.result
    ? "実際の結果 " + DATA.result + " は " + N + "回中 " + (cc[DATA.result] || 0) +
      "回（" + ((cc[DATA.result] || 0) / N * 100).toFixed(1) + "%）出現しました。"
    : "このレースはまだ結果が確定していません（確定後にタブ2で再取得すると結果と照合できます）。") +
    "　※毎回その場で1000回独立に抽選しています（出現した3連単の種類数: " +
    Object.keys(cc).length + "通り／全120通り）。";
  document.getElementById("verify").scrollIntoView({ behavior: "smooth", block: "start" });
};

// 初期描画: ピットに6艇待機
BOATS.forEach(b => { b.p = 0; b.crossT = 0; buildScript(b); });
drawPond(0, "", "");
BOATS.forEach(b => { const q = scriptPos(b, 0); drawBoat(b, q.x, q.y, 0, 0); });
renderRanks(BOATS.slice().sort((a, b) => EVALS[b.lane] - EVALS[a.lane]), "race");
</script></body></html>'''


def build_simulator_html(sim: Dict) -> str:
    """タブ2の予想（3連単120点の補正後確率）からシミュレーターHTMLを生成する。"""
    cls_inv = {4: "A1", 3: "A2", 2: "B1", 1: "B2"}
    boats = []
    for rw in sim.get("racers", []):
        boats.append({
            "lane": int(rw["lane"]),
            "cls": cls_inv.get(int(rw.get("cls_val", 1)), "B2"),
            "tenji": float(rw.get("tenji", 0) or 0),
            "avg_st": float(rw.get("avg_st", 0.16) or 0.16),
            "course": int(rw.get("course_in", rw["lane"])),
        })
    ltr = sim.get("lane_to_rank") or {}
    r1 = next((l for l, r in ltr.items() if r == 1), None)
    r2 = next((l for l, r in ltr.items() if r == 2), None)
    r3 = next((l for l, r in ltr.items() if r == 3), None)
    result = f"{r1}-{r2}-{r3}" if (r1 and r2 and r3) else None
    data = {
        "place": sim.get("place", ""),
        "rno": int(sim.get("rno", 0)),
        "date": str(sim.get("date", "")),
        "combos": {k: float(v) for k, v in (sim.get("combo_probs") or {}).items()},
        "boats": boats,
        "result": result,
    }
    return _SIM_HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))


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

tab1, tab2, tab3, tab4 = st.tabs(["📊 バックテスト", "🎯 当日予想", "🔎 全レース勝率スキャン", "🚤 レースシミュレーター"])

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
    cD, cE, cF = st.columns(3)
    with cD:
        use_course2 = st.checkbox("コース別連対率で補正", value=True, key="t2_course",
                                  help="タブ3と同じ後段補正。選手ページ(racerYm)の進入コース別2連対率/3連対率で増幅。")
    with cE:
        course_k2 = st.slider("コース補正 k", 0.0, 3.0, 1.0, 0.1, key="t2_course_k")
    with cF:
        weather_k2 = st.slider("風・波補正 k", 0.0, 2.0, 0.7, 0.1, key="t2_weather_k",
                               help="公式直前情報の風速・波高で荒れ水面補正（タブ3と同じ）。")

    if st.button("🔍 取得して予想", type="primary", use_container_width=True):
        dstr = d_input.strftime("%Y%m%d")
        # タブ3とキャッシュ共有: タブ3でスキャン済みの日付なら選手データ取得を省略できる
        ss_regno = st.session_state.setdefault("t3_regno_cache", {})  # (date, jcd) -> {rno: {lane: regno}}
        ss_rates = st.session_state.setdefault("t3_rate_cache", {})   # (date, regno) -> rates
        need_reg = use_course2 and course_k2 > 0 and (dstr, v_idx) not in ss_regno
        with st.spinner("並列取得中 (kyotei.fun ＋ 公式 直前情報/オッズ ＋ sakura 出走表)..."):
            with ThreadPoolExecutor(max_workers=4) as ex:
                f_base = ex.submit(fetch_race_data, dstr, v_idx, r_idx)
                f_bi   = ex.submit(fetch_official_beforeinfo, dstr, v_idx, r_idx)
                f_odds = (ex.submit(fetch_official_odds3t, dstr, v_idx, r_idx)
                          if OFFICIAL_ODDS3T_ENABLED else None)
                f_reg  = ex.submit(fetch_racelist_regnos, dstr, v_idx) if need_reg else None
                base_pre = f_base.result()
                bi_pre   = f_bi.result()
                odds_pre = f_odds.result() if f_odds is not None else None
                if f_reg is not None:
                    ss_regno[(dstr, v_idx)] = f_reg.result() or {}
        res = fetch_race_data_hybrid(dstr, v_idx, r_idx,
                                     pre_base=base_pre, pre_bi=bi_pre, pre_odds=odds_pre)
        if not res:
            st.error("取得失敗。日付・場・レース番号を確認してください。"); st.stop()
        racers, lane_to_rank, odds_map, payoff, sources = res

        # コース別連対率: このレースの6選手ぶんだけ、未キャッシュ分を並列取得
        regno_map_r = (ss_regno.get((dstr, v_idx), {}).get(r_idx, {})
                       if (use_course2 and course_k2 > 0) else {})
        if regno_map_r:
            miss = [reg for reg in regno_map_r.values()
                    if reg and (dstr, reg) not in ss_rates]
            if miss:
                with st.spinner(f"選手コース別データ取得中… {len(miss)} 名（並列）"):
                    with ThreadPoolExecutor(max_workers=4) as ex:
                        futs = {ex.submit(fetch_course_rates, dstr, reg): reg for reg in miss}
                        for f in as_completed(futs):
                            ss_rates[(dstr, futs[f])] = f.result() or {}

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

        # --- タブ3と同じ後段補正 ---
        adj_notes = []
        if use_course2 and course_k2 > 0 and regno_map_r:
            course_by_lane = {int(rw["lane"]): int(rw.get("course_in", rw["lane"])) for rw in racers}
            r2_by_lane, r3_by_lane = {}, {}
            for lane in range(1, 7):
                reg = regno_map_r.get(lane)
                rates = ss_rates.get((dstr, reg), {}) if reg else {}
                cr = rates.get(course_by_lane.get(lane, lane), {})
                r2_by_lane[lane] = cr.get("r2")
                r3_by_lane[lane] = cr.get("r3")
            combo_probs = apply_course_adjustment(combo_probs, r2_by_lane, r3_by_lane, course_k2)
            n_known = sum(1 for v in r2_by_lane.values() if v is not None)
            adj_notes.append(f"コース別補正 k={course_k2:.1f}（連対率データ {n_known}/6 艇）")
        elif use_course2 and course_k2 > 0:
            adj_notes.append("コース別補正: 出走表（登録番号）が取得できず無補正")
        wx2 = sources.get("weather", {})
        if weather_k2 > 0 and wx2:
            cmap2 = {int(rw["lane"]): int(rw.get("course_in", rw["lane"])) for rw in racers}
            combo_probs = apply_weather_adjustment(
                combo_probs, wx2.get("風速(m)"), wx2.get("波高(cm)"), weather_k2, cmap2)
            adj_notes.append(f"風・波補正 k={weather_k2:.1f}"
                             f"（風{wx2.get('風速(m)', '—')}m / 波{wx2.get('波高(cm)', '—')}cm）")
        if adj_notes:
            st.caption("🔧 " + " ・ ".join(adj_notes) + " — 以下の評価値・EVは補正後の確率です。")

        # レースシミュレーター(タブ4)用に補正後の予想データを保存
        st.session_state["sim_data"] = {
            "date": dstr, "jcd": v_idx, "rno": r_idx,
            "place": JCD_NAME.get(v_idx, str(v_idx)),
            "combo_probs": combo_probs, "racers": racers,
            "lane_to_rank": lane_to_rank,
        }

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


# ----------------------------- Tab4: レース展開シミュレーター
with tab4:
    st.markdown("##### モデル予想（3連単120点の確率）から着順を抽選し、レース展開をアニメーション再現＆1000回検証")
    sim = st.session_state.get("sim_data")
    if not sim or not sim.get("combo_probs"):
        st.info("先にタブ2『🎯 当日予想』で対象レースを取得・予想してください。\n\n"
                "タブ2で予想すると、その**補正後の3連単確率**がこのタブに引き継がれ、"
                "予想反映度を変えながら展開シミュレーション・1000回検証ができます。")
    else:
        st.caption(f"対象レース: {sim['date']} {sim['place']} {sim['rno']}R "
                   "（タブ2の補正後確率を使用。タブ2で別レースを予想すると切り替わります）")
        components.html(build_simulator_html(sim), height=1450, scrolling=True)
        st.caption("💡 予想反映度100%＝モデルの3連単確率そのままで着順を抽選。0%＝120通り完全ランダム。"
                   "1000回検証では、各艇の1着回数と3連単の出現分布をモデル理論値と比較できます。")
