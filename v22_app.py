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
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    return unicodedata.normalize("NFKC", s or "").replace(" ", "").replace("\u3000", "")


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
            current_label = tds[0].get_text(strip=True).replace("\n","").replace(" ","").replace("\u3000","")
            data_tds = tds[-6:]
        elif len(tds) == 6 and current_label:
            data_tds = tds
        else:
            current_label = ""
            continue
        for i in range(6):
            td = data_tds[i]
            txt = td.get_text(" ", strip=True).replace(" ","").replace("\u3000","").replace("\n","")
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
    url = f"https://www.boatrace.jp/owpc
