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

    odds_map = {}
    h3_target = None
    for h3 in soup.find_all("h3"):
        if "3連単" in h3.get_text() and "人気" in h3.get_text():
            h3_target = h3; break
    if h3_target:
        container = h3_target.find_parent("div", id="raceData") or h3_target.parent
        for tbl in container.find_all("table", id="oddsTbl"):
            for tr in tbl.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) != 2: continue
                ng23 = tds[0].find("div", class_="ng23")
                if not ng23: continue
                divs = ng23.find_all("div")
                nums = []
                for d in divs[:3]:
                    m = re.search(r"ng2r(\d)", " ".join(d.get("class", [])))
                    if m: nums.append(int(m.group(1)))
                if len(nums) != 3 or len(set(nums)) != 3: continue
                txt = tds[1].get_text(strip=True).replace(",","")
                try: v = float(txt)
                except: continue
                odds_map[f"{nums[0]}-{nums[1]}-{nums[2]}"] = v

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
    """公式の直前情報ページから展示タイム・コースIN・気象を取得。
    返り値: {'tenji': {lane:float}, 'course_in': {lane:int}, 'weather': {...}}
    失敗時は空辞書を返す。"""
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd:02d}&hd={date}"
    out = {"tenji": {}, "course_in": {}, "weather": {}}
    try:
        r = req_sess.get(url, timeout=7)
        r.encoding = r.apparent_encoding
        if r.status_code != 200:
            return {}
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception:
        return {}

    # 直前情報テーブル: 各艇行に展示タイム・進入コースが入っている想定
    # 公式HTMLは表構造が複雑なので、出現する数値パターンで検出する
    try:
        # 展示タイム: 6.xx か 7.xx (秒) のパターン
        # まず "is-fBold" の数値セルから展示タイム候補を集める
        for tr in soup.select("table tbody tr"):
            tds = tr.find_all("td")
            if len(tds) < 3: continue
            head = tds[0].get_text(strip=True)
            if not head.isdigit() or not (1 <= int(head) <= 6):
                continue
            lane = int(head)
            for td in tds:
                txt = td.get_text(strip=True)
                if re.fullmatch(r'[67]\.\d{2}', txt):
                    out["tenji"][lane] = float(txt)
                    break
    except Exception:
        pass

    # 進入コース: スタート展示の進入図/テキストから抽出
    # 公式の HTML 構造を仮定（"枠なり" や明示的なコース番号があるパターン）
    try:
        # 「進入」見出し付近のテーブル/divを探索
        for elem in soup.find_all(string=re.compile(r"進入|スタート展示")):
            parent = elem.find_parent(["table", "div", "section"])
            if not parent: continue
            # 1〜6の数値がコース順に並ぶパターンを試行
            txt_seq = parent.get_text(" ", strip=True)
            nums = re.findall(r"\b([1-6])\b", txt_seq)
            if 6 <= len(nums) <= 18:
                # 最初の6個を「コース順→艇番」と仮定
                # course_in[lane] = course のマップを作る
                course_order = [int(n) for n in nums[:6]]
                if sorted(course_order) == [1,2,3,4,5,6]:
                    for course, lane in enumerate(course_order, 1):
                        out["course_in"][lane] = course
                    break
    except Exception:
        pass

    # 気象
    try:
        wtxt = soup.get_text(" ", strip=True)
        mw = re.search(r"風速\s*([\d.]+)\s*m", wtxt)
        if mw: out["weather"]["風速(m)"] = float(mw.group(1))
        mt = re.search(r"気温\s*([\d.]+)", wtxt)
        if mt: out["weather"]["気温"] = float(mt.group(1))
        mv = re.search(r"波高\s*([\d.]+)\s*cm", wtxt)
        if mv: out["weather"]["波高(cm)"] = float(mv.group(1))
    except Exception:
        pass

    return out


def fetch_official_odds3t(date: str, jcd: int, rno: int) -> Dict[str, float]:
    """公式の3連単オッズページから取得。
    返り値: {'1-2-3': 7.5, ...} 120点。取れなければ空辞書。
    ※ JavaScript描画で取れない可能性があるため、フォールバック前提で使うこと。"""
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={date}"
    try:
        r = req_sess.get(url, timeout=7)
        r.encoding = r.apparent_encoding
        if r.status_code != 200:
            return {}
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception:
        return {}

    # td.oddsPoint を全部拾う (公式の3連単オッズセル)
    cells = soup.select("td.oddsPoint")
    vals = []
    for c in cells:
        txt = c.get_text(strip=True).replace(",", "")
        try:
            vals.append(float(txt))
        except ValueError:
            vals.append(0.0)

    if len(vals) != 120:
        return {}   # JS描画等で取れていない

    # 並び順: 1着外ループ→2着ブロック→3着行 (辞書順)
    # ※前にスクリーンショットから候補Aが正解と判明済み
    combos = [f"{a}-{b}-{c}" for a in range(1,7) for b in range(1,7)
              for c in range(1,7) if len({a,b,c}) == 3]
    return {c: o for c, o in zip(combos, vals) if o > 0}


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
    sources["weather"] = bi.get("weather", {})

    # 公式オッズ
    time.sleep(0.5)
    official_odds = fetch_official_odds3t(date, jcd, rno)
    if official_odds and len(official_odds) >= 100:
        odds_map = official_odds
        sources["odds_from_official"] = True

    return racers, lane_to_rank, odds_map, payoff, sources


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

tab1, tab2 = st.tabs(["📊 バックテスト", "🎯 当日予想"])

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
        with st.spinner("取得中 (kyotei.fun + 公式 boatrace.jp)..."):
            res = fetch_race_data_hybrid(dstr, v_idx, r_idx)
            time.sleep(1.0)
        if not res:
            st.error("取得失敗"); st.stop()
        racers, lane_to_rank, odds_map, payoff, sources = res

        # 取得元の表示
        with st.container():
            col1, col2, col3 = st.columns(3)
            col1.metric("公式 オッズ", "✅ 取得" if sources["odds_from_official"] else "❌ kyotei.funを使用")
            col2.metric("公式 展示タイム", "✅ 取得" if sources["tenji_from_official"] else "❌ kyotei.funを使用")
            col3.metric("公式 コースIN", "✅ 取得" if sources["course_in_from_official"] else "❌ kyotei.funを使用")
        if sources.get("weather"):
            w = sources["weather"]
            st.info("🌬️ 気象(参考・モデル未使用): " + " / ".join(f"{k} {v}" for k, v in w.items()))

        st.subheader("出走表")
        df_show = pd.DataFrame(racers)[["lane","cls_val","age","avg_st","n_win","m_2ren","tenji","course_in"]]
        df_show.columns = ["枠","級","年齢","平均ST","勝率","M2連率","展示","コースIN"]
        st.dataframe(df_show.set_index("枠"), use_container_width=True)
        st.caption(f"オッズ取得: {len(odds_map)}/120")

        feat_df = make_race_features(racers)
        combo_probs = predict_combo_probs(feat_df, v_idx)

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
