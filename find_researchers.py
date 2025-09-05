# Run with:  streamlit run ResearcherScreening_v4_app.py
# Requirements: pip install streamlit requests pandas python-dateutil

import re
import time
from typing import List, Dict, Any, Tuple

import pandas as pd
import requests
import streamlit as st

# Safety: avoid NameError if legacy blocks still reference `run`
run = False

st.set_page_config(page_title="InsighTCROSS Literature Scorer v4", layout="wide")

# --- パスワード認証処理（堅牢化版） ---
def check_password():
    """
    GitHub/Streamlit Cloud前提のパスワード保護。
    - .streamlit/secrets.toml の [passwords].app_password を使用（APIキー不要）
    """
    # secrets の設定漏れチェック
    if "passwords" not in st.secrets or "app_password" not in st.secrets["passwords"]:
        st.error("アプリ設定エラー：.streamlit/secrets.toml に [passwords].app_password を設定してください。")
        st.stop()

    # 状態管理
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if "pw_attempts" not in st.session_state:
        st.session_state["pw_attempts"] = 0

    # 未認証：フォーム
    if not st.session_state["password_correct"]:
        with st.form("login_form", clear_on_submit=False):
            password = st.text_input("パスワードを入力してください", type="password")
            submitted = st.form_submit_button("ログイン")

        if submitted:
            if password == st.secrets["passwords"]["app_password"]:
                st.session_state["password_correct"] = True
                st.session_state["pw_attempts"] = 0
                st.rerun()
            else:
                st.session_state["pw_attempts"] += 1
                st.error("パスワードが違います。")
                if st.session_state["pw_attempts"] >= 5:
                    st.warning("試行回数が多すぎます。しばらく時間を空けてから再度お試しください。")
                    st.stop()
        return False

    return True

# -------------------- Helpers --------------------
EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

ARTICLE_TYPE_MAP = {
    "Clinical Trial": "Clinical Trial",
    "Meta-Analysis": "Meta-Analysis",
    "Randomized Controlled Trial": "Randomized Controlled Trial",
    "Review": "Review",
    "Systematic Review": "Systematic Review",
}

def build_query(
    disease: str,
    country: str,
    year_from: int,
    year_to: int,
    department: str,
    keywords: str,
    ta_abstract: bool,
    ta_free_full: bool,
    ta_full_text: bool,
    atypes: List[str],
    src_med: bool,
    src_pmc: bool,
    src_ppr: bool,
    excl_ppr: bool,
) -> str:
    parts = []

    def sanitize(x: str) -> str:
        x = re.sub(r'["“”„‟«»‹›「」『』＂]', '', x.strip())
        return re.sub(r"\s+", " ", x)

    # Disease: フレーズ一致 + 全語AND（取りこぼし減）
    if disease:
        dz = sanitize(disease)
        tokens = [t for t in dz.split(" ") if t]
        phrase = f'"{dz}"'
        if len(tokens) > 1:
            and_block = " AND ".join([f'"{t}"' for t in tokens])
            parts.append(
                f'(TITLE:{phrase} OR ABSTRACT:{phrase} OR TITLE:({and_block}) OR ABSTRACT:({and_block}))'
            )
        else:
            parts.append(f'(TITLE:{phrase} OR ABSTRACT:{phrase})')

    # Department/Division in affiliation（和英・表記揺れを少し吸収）
    if department:
        dept_tokens = [sanitize(t) for t in re.split(r",|、|;|/|\|", department) if sanitize(t)]
        if dept_tokens:
            exp = []
            for t in dept_tokens:
                exp.extend(
                    [
                        f'AFF:"{t}"',
                        f'AFF:"Department of {t}"',
                        f'AFF:"Division of {t}"',
                        f'AFF:"Dept. of {t}"',
                        f'AFF:"{t} Department"',
                        f'AFF:"{t} Division"',
                        f'AFF:"{t} Unit"',
                        f'AFF:"{t}科"',
                        f'AFF:"{t}部"',
                        f'AFF:"{t}講座"',
                    ]
                )
            parts.append("(" + " OR ".join(exp) + ")")

    # Free keywords in Title/Abstract
    if keywords:
        toks = [sanitize(t) for t in re.split(r",|、|;|/", keywords) if sanitize(t)]
        if toks:
            or_block = " OR ".join([f'"{t}"' for t in toks])
            parts.append(f"(TITLE:({or_block}) OR ABSTRACT:({or_block}))")

    if country:
        parts.append(f'AFF:"{sanitize(country)}"')

    # Date range: FIRST_PDATE を使って厳密にフィルタ（YYYY-MM-DD）
    parts.append(f"FIRST_PDATE:[{int(year_from)}-01-01 TO {int(year_to)}-12-31]")

    if ta_abstract:
        parts.append("HAS_ABSTRACT:Y")
    if ta_free_full:
        parts.append("OPEN_ACCESS:Y")
    if ta_full_text:
        parts.append("HAS_FULL_TEXT:Y")

    if atypes:
        mapped = [ARTICLE_TYPE_MAP.get(a) for a in atypes if ARTICLE_TYPE_MAP.get(a)]
        if mapped:
            typ_block = " OR ".join([f'"{t}"' for t in mapped])
            parts.append(f"PUB_TYPE:({typ_block})")

    # Sources
    source_tokens = []
    if src_med:
        source_tokens.append("SRC:MED")
    if src_pmc:
        source_tokens.append("SRC:PMC")
    if src_ppr:
        source_tokens.append("SRC:PPR")
    if source_tokens:
        parts.append("(" + " OR ".join(source_tokens) + ")")
    if excl_ppr:
        parts.append("NOT SRC:PPR")

    return " AND ".join(parts) if parts else "*"

def extract_department(aff: str) -> str:
    if not aff:
        return ""
    m = re.search(r"(Department of [^.;|]+|Dept\. of [^.;|]+|Division of [^.;|]+|科[^；。|]+)", aff, flags=re.I)
    return m.group(0) if m else ""

def fetch_eupmc_all(query: str, max_rows: int = 5000, synonym: bool = False) -> Tuple[List[Dict[str, Any]], int]:
    """最大 max_rows まで取得し、(items, hitCount) を返す（APIキー不要のEurope PMC）"""
    results = []
    cursor = "*"
    page_size = 1000 if max_rows >= 1000 else max_rows
    fetched = 0
    total_hit_count = 0

    while fetched < max_rows:
        params = {
            "query": query,
            "format": "json",
            "resultType": "core",
            "pageSize": page_size,
            "cursorMark": cursor,
            "synonym": str(synonym).lower(),
        }
        r = requests.get(EPMC_SEARCH_URL, params=params, timeout=60)
        r.raise_for_status()
        j = r.json()
        if not total_hit_count:
            total_hit_count = int(j.get("hitCount", 0))
        items = j.get("resultList", {}).get("result", [])
        if not items:
            break
        results.extend(items)
        fetched += len(items)
        cursor = j.get("nextCursorMark") or cursor
        if not j.get("nextCursorMark") or len(items) < page_size:
            break
        time.sleep(0.2)

    return results[:max_rows], total_hit_count

def get_hit_count(query: str, synonym: bool = False) -> int:
    params = {
        "query": query,
        "format": "json",
        "resultType": "core",
        "pageSize": 1,
        "synonym": str(synonym).lower(),
    }
    r = requests.get(EPMC_SEARCH_URL, params=params, timeout=30)
    r.raise_for_status()
    return int(r.json().get("hitCount", 0))

def pick_main_affiliation(aff_value) -> str:
    if not aff_value:
        return ""
    if isinstance(aff_value, list):
        parts = [a for a in aff_value if a]
        if not parts:
            return ""
        first = parts[0]
    else:
        first = str(aff_value)
    return re.split(r"\s*\|\s*", first)[0].strip()

# ---- ローマ字→カタカナ（簡易ヘボン式）----
VOWELS = set("aeiou")

TRI_MAP = {
    "kya":"キャ","kyu":"キュ","kyo":"キョ",
    "gya":"ギャ","gyu":"ギュ","gyo":"ギョ",
    "sha":"シャ","shu":"シュ","sho":"ショ",
    "cha":"チャ","chu":"チュ","cho":"チョ",
    "ja":"ジャ","ju":"ジュ","jo":"ジョ",
    "nya":"ニャ","nyu":"ニュ","nyo":"ニョ",
    "hya":"ヒャ","hyu":"ヒュ","hyo":"ヒョ",
    "mya":"ミャ","myu":"ミュ","myo":"ミョ",
    "rya":"リャ","ryu":"リュ","ryo":"リョ",
    "bya":"ビャ","byu":"ビュ","byo":"ビョ",
    "pya":"ピャ","pyu":"ピュ","pyo":"ピョ",
    "tya":"チャ","tyu":"チュ","tyo":"チョ",
    "dya":"ジャ","dyu":"ジュ","dyo":"ジョ",
}

DI_MAP = {
    "shi":"シ","chi":"チ","tsu":"ツ","fu":"フ","ji":"ジ",
    "ti":"ティ","di":"ディ","tu":"トゥ","du":"ドゥ",
    "je":"ジェ","che":"チェ","she":"シェ",
    "ci":"シ","ce":"セ","wi":"ウィ","we":"ウェ","wo":"ヲ",
    "ja":"ジャ","ju":"ジュ","jo":"ジョ",
}

BASE = {
    "a":"ア","i":"イ","u":"ウ","e":"エ","o":"オ",
    "ka":"カ","ki":"キ","ku":"ク","ke":"ケ","ko":"コ",
    "ga":"ガ","gi":"ギ","gu":"グ","ge":"ゲ","go":"ゴ",
    "sa":"サ","si":"シ","su":"ス","se":"セ","so":"ソ",
    "za":"ザ","zi":"ジ","zu":"ズ","ze":"ゼ","zo":"ゾ",
    "ta":"タ","ti":"ティ","tu":"トゥ","te":"テ","to":"ト",
    "da":"ダ","di":"ディ","du":"ドゥ","de":"デ","do":"ド",
    "na":"ナ","ni":"ニ","nu":"ヌ","ne":"ネ","no":"ノ",
    "ha":"ハ","hi":"ヒ","hu":"フ","he":"ヘ","ho":"ホ",
    "ba":"バ","bi":"ビ","bu":"ブ","be":"ベ","bo":"ボ",
    "pa":"パ","pi":"ピ","pu":"プ","pe":"ペ","po":"ポ",
    "ma":"マ","mi":"ミ","mu":"ム","me":"メ","mo":"モ",
    "ra":"ラ","ri":"リ","ru":"ル","re":"レ","ro":"ロ",
    "ya":"ヤ","yu":"ユ","yo":"ヨ",
    "wa":"ワ","we":"ウェ","wo":"ヲ",
    "va":"ヴァ","vi":"ヴィ","vu":"ヴ","ve":"ヴェ","vo":"ヴォ",
    "fa":"ファ","fi":"フィ","fe":"フェ","fo":"フォ",
    "la":"ラ","li":"リ","lu":"ル","le":"レ","lo":"ロ",
    "ca":"カ","ci":"シ","cu":"ク","ce":"セ","co":"コ",
    "qa":"カ","qi":"キ","qu":"ク","qe":"ケ","qo":"コ",
}

MACRON = str.maketrans({"ā":"aa","ī":"ii","ū":"uu","ē":"ee","ō":"ou","â":"aa","î":"ii","û":"uu","ê":"ee","ô":"ou"})

# 個別上書き（姓）
KATAKANA_LASTNAME_OVERRIDE = {
    "amano": "アマノ",
    "ando": "アンドウ",
    "domei": "ドウメイ",
    "gohbara": "ゴウハラ",
    "goriki": "ゴウリキ",
    "hanyu": "ハニュウ",
    "homma": "ホンマ",
    "honye": "ホンエ",
    "ito": "イトウ",
    "jujo": "ジュウジョウ",
    "kato": "カトウ",
    "kohro": "コウロ",
    "kohsaka": "コウサカ",
    "kozuma": "コウズマ",
    "mitsudo": "ミツドウ",
    "morino": "モリノ",
    "nanasato": "ナナサト",
    "obayashi": "オオバヤシ",
    "ohi": "オオイ",
    "ohara": "オオハラ",
    "ohashi": "オオハシ",
    "ohata": "オオハタ",
    "ohba": "オオバ",
    "ohguchi": "オオグチ",
    "ohguro": "オオグロ",
    "ohishi": "オオイシ",
    "ohkawa": "オオカワ",
    "ohkita": "オオキタ",
    "ohkubo": "オオクボ",
    "ohmori": "オオモリ",
    "ohnishi": "オオニシ",
    "ohno": "オオノ",
    "ohta": "オオタ",
    "ohtani": "オオタニ",
    "ohtomo": "オオトモ",
    "ohsaka": "オオサカ",
    "ohsawa": "オオサワ",
    "ohshiro": "オオシロ",
    "ohshima": "オオシマ",
    "ohsugi": "オオスギ",
    "ohsumi": "オオスミ",
    "ohuchi": "オオウチ",
    "ohwada": "オオワダ",
    "ohya": "オオヤ",
    "oiwa": "オオイワ",
    "oishi": "オオイシ",
    "oizumi": "オオイズミ",
    "okubo": "オオクボ",
    "okura": "オオクラ",
    "omori": "オオモリ",
    "omura": "オオムラ",
    "onishi": "オオニシ",
    "ono": "オノ",
    "ooba": "オオバ",
    "oobayashi": "オオバヤシ",
    "oochi": "オオチ",
    "oodate": "オオダテ",
    "oohara": "オオハラ",
    "oohashi": "オオハシ",
    "ooguchi": "オオグチ",
    "ooguro": "オオグロ",
    "ookawa": "オオカワ",
    "ookita": "オオキタ",
    "ookubo": "オオクボ",
    "oomori": "オオモリ",
    "oomura": "オオムラ",
    "oonishi": "オオニシ",
    "oono": "オオノ",
    "oosaka": "オオサカ",
    "oosawa": "オオサワ",
    "ooshima": "オオシマ",
    "oosugi": "オオスギ",
    "oouchi": "オオウチ",
    "oowada": "オオワダ",
    "ooya": "オオヤ",
    "ooi": "オオイ",
    "osaka": "オオサカ",
    "oshima": "オオシマ",
    "osumi": "オオスミ",
    "ota": "オオタ",
    "otake": "オオタケ",
    "otomo": "オオトモ",
    "otsuba": "オオツバ",
    "otsubo": "オオツボ",
    "otsuka": "オオツカ",
    "otsuki": "オオツキ",
    "oyama": "オヤマ",
    "saito": "サイトウ",
    "sato": "サトウ",
    "shindo": "シンドウ",
    "sudo": "スドウ",
}

# 個別上書き（名 / FirstName）
KATAKANA_FIRSTNAME_OVERRIDE = {
    "asataro": "アサタロウ",
    "eiichiro": "エイイチロウ",
    "eizo": "エイゾウ",
    "go": "ゴウ",
    "goro": "ゴロウ",
    "ichitaro": "イチタロウ",
    "issei": "イッセイ",
    "itsuro": "イツロウ",
    "jiro": "ジロウ",
    "jo": "ジョウ",
    "joji": "ジョウジ",
    "jun": "ジュン",
    "jun-ichi": "ジュンイチ",
    "junichi": "ジュンイチ",
    "junko": "ジュンコ",
    "junya": "ジュンヤ",
    "ken-ichi": "ケンイチ",
    "kenichi": "ケンイチ",
    "kenichiro": "ケンイチロウ",
    "kenya": "ケンヤ",
    "kentaro": "ケンタロウ",
    "ko": "コウ",
    "koh": "コウ",
    "kohei": "コウヘイ",
    "koichi": "コウイチ",
    "koichiro": "コウイチロウ",
    "koji": "コウジ",
    "koki": "コウキ",
    "kosei": "コウセイ",
    "koshi": "コウシ",
    "koshiro": "コウシロウ",
    "kosuke": "コウスケ",
    "kota": "コウタ",
    "kotaro": "コウタロウ",
    "kyohei": "キョウヘイ",
    "kyoko": "キョウコ",
    "rensuke": "レンスケ",
    "ryo": "リョウ",
    "ryohei": "リョウヘイ",
    "ryoichi": "リョウイチ",
    "ryoji": "リョウジ",
    "ryoko": "リョウコ",
    "ryoma": "リョウマ",
    "ryosuke": "リョウスケ",
    "ryozo": "リョウゾウ",
    "ryuichi": "リュウイチ",
    "ryuki": "リュウキ",
    "ryusuke": "リュウスケ",
    "sho": "ショウ",
    "shohei": "ショウヘイ",
    "shoichi": "ショウイチ",
    "shoji": "ショウジ",
    "shogo": "ショウゴ",
    "shoko": "ショウコ",
    "shota": "ショウタ",
    "shotaro": "ショウタロウ",
    "shunichiro": "シュンイチロウ",
    "shunsuke": "シュンスケ",
    "shusuke": "シュウスケ",
    "sohsuke": "ソウスケ",
    "soichi": "ソウイチ",
    "soichiro": "ソウイチロウ",
    "sota": "ソウタ",
    "taro": "タロウ",
    "tohru": "トオル",
    "toshiro": "トシロウ",
    "toru": "トオル",
    "yohei": "ヨウヘイ",
    "yoichiro": "ヨウイチロウ",
    "yoko": "ヨウコ",
    "yosuke": "ヨウスケ",
    "yu": "ユウ",
    "yudai": "ユウダイ",
    "yuetsu": "ユウエツ",
    "yuji": "ユウジ",
    "yujiro": "ユウジロウ",
    "yuko": "ユウコ",
    "yuma": "ユウマ",
    "yusuke": "ユウスケ",
    "yuta": "ユウタ",
    "yuto": "ユウト",
    "yuya": "ユウヤ",
}

# 末尾 to/do を長音化の対象にする苗字（完全一致・小文字）
LONG_O_SURNAMES = {
    "saito", "saitoh", "saitou",
    "sato",
    "ando", "kondo", "endo", "shindo",
    "goto", "mitsudo", "kato", "sudo",
}

def roman_to_katakana(s: str, long_vowel: bool = True) -> str:
    """ローマ字→カタカナ（簡易）
    修正点: n + 母音 は「ン+ア」ではなく na/ni/nu/ne/no → ナ/ニ/ヌ/ネ/ノ にする。
    """
    if not s:
        return ""
    s = s.lower().translate(MACRON)
    s = re.sub(r"[^a-z]", "", s)
    # Hepburn系の "oh" を長音の ō とみなす（子音or語末の直前のみ）
    s = re.sub(r"oh(?=[bcdfghjklmnpqrstvwxyz]|$)", "ou", s)
    out = []
    i = 0
    while i < len(s):
        # 促音（ッ）
        if i + 1 < len(s) and s[i] == s[i + 1] and s[i] not in VOWELS and s[i] != "n":
            out.append("ッ"); i += 1; continue
        # 三文字コンボ
        tri = s[i:i+3]
        if tri in TRI_MAP:
            out.append(TRI_MAP[tri]); i += 3; continue
        # 二〜三文字の特殊綴り
        di3 = s[i:i+3]
        if di3 in DI_MAP:
            out.append(DI_MAP[di3]); i += 3; continue
        di2 = s[i:i+2]
        if di2 in DI_MAP:
            out.append(DI_MAP[di2]); i += 2; continue
        # 'n' の扱い
        if s[i] == 'n':
            if i + 1 < len(s) and s[i+1] in 'aiueo':
                key = 'n' + s[i+1]
                kana = BASE.get(key)
                if kana:
                    out.append(kana); i += 2; continue
            if i + 1 < len(s) and s[i+1] == 'y':
                out.append('ニ'); i += 1; continue
            out.append('ン'); i += 1; continue
        # 子音+母音の基本形
        if i + 1 < len(s) and s[i] not in VOWELS and s[i+1] in VOWELS:
            key = ("r" if s[i] == "l" else s[i]) + s[i+1]
            kana = BASE.get(key)
            if kana:
                out.append(kana); i += 2; continue
        # 母音単独
        if s[i] in VOWELS:
            out.append(BASE.get(s[i], "")); i += 1; continue
        i += 1
    kat = "".join(out)
    if long_vowel:
        kat = re.sub(r"オウウ", "オウ", kat)
    return kat

def to_katakana_fullname(first: str, last: str, long_vowel: bool = True) -> str:
    return roman_to_katakana(last, long_vowel) + roman_to_katakana(first, long_vowel)

def apply_surname_long_o(ln_lower: str, ln_kana: str) -> str:
    """特定苗字だけ末尾 to/do を長音化（例: Saito→サイトウ, Ando→アンドウ）。"""
    if ln_lower in LONG_O_SURNAMES:
        if ln_lower.endswith("to") and ln_kana.endswith("ト") and not ln_kana.endswith("トウ"):
            return ln_kana + "ウ"
        if ln_lower.endswith("do") and ln_kana.endswith("ド") and not ln_kana.endswith("ドウ"):
            return ln_kana + "ウ"
    return ln_kana

def explode_authors(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in items:
        title = rec.get("title", "")
        pmid = rec.get("pmid") or rec.get("id") or ""
        journal = rec.get("journalTitle", "")
        year = rec.get("pubYear") or rec.get("firstPublicationDate", "")
        try:
            if isinstance(year, str) and len(year) >= 4:
                year = int(year[:4])
        except Exception:
            pass
        authors = (rec.get("authorList", {}) or {}).get("author", [])
        for idx, a in enumerate(authors):
            full = a.get("fullName") or ""
            first = a.get("firstName") or ""
            last = a.get("lastName") or ""
            affs = a.get("affiliation", [])
            display_name = (f"{first} {last}".strip()) if (first or last) else full
            if isinstance(affs, list):
                aff_joined = " | ".join([x for x in affs if x])
            else:
                aff_joined = affs or ""
            dept = extract_department(aff_joined)
            score = 2 if idx == 0 else 1
            rows.append(
                {
                    "PMID": pmid,
                    "Title": title,
                    "Journal": journal,
                    "Year": year,
                    "AuthorOrder": idx + 1,
                    "FullName": full,
                    "FirstName": first,
                    "LastName": last,
                    "DisplayName": display_name,
                    "Affiliation": aff_joined,
                    "MainAffiliation": pick_main_affiliation(affs),
                    "Department_guess": dept,
                    "Score": score,
                }
            )
    return pd.DataFrame(rows)

def aggregate_author_scores(df: pd.DataFrame, long_vowel: bool = True, surname_long_o: bool = False) -> pd.DataFrame:
    if df.empty:
        return df
    name_col = df["DisplayName"].where(df["DisplayName"].astype(bool), df["FullName"])
    main_aff = df["MainAffiliation"].fillna("")
    base = (
        df.assign(Author=name_col, MainAff=main_aff)
          .groupby(["Author", "MainAff"], dropna=False)["Score"]
          .sum()
          .reset_index()
          .rename(columns={"MainAff": "MainAffiliation"})
    )
    rep = (
        df.assign(Author=name_col)
          .dropna(subset=["Author"])
          .groupby("Author")
          .agg({"FirstName":"first", "LastName":"first"})
          .reset_index()
    )

    def _kana_parts(a: str):
        rec = rep[rep["Author"] == a]
        if not rec.empty:
            fn = str(rec.iloc[0]["FirstName"] or "")
            ln = str(rec.iloc[0]["LastName"] or "")
        else:
            parts = (a or "").split()
            fn = parts[0] if parts else ""
            ln = parts[-1] if len(parts) >= 2 else ""

        ln_lower = (ln or "").lower()
        ln_kana = KATAKANA_LASTNAME_OVERRIDE.get(ln_lower) or roman_to_katakana(ln, long_vowel)
        if surname_long_o and ln_lower not in KATAKANA_LASTNAME_OVERRIDE:
            ln_kana = apply_surname_long_o(ln_lower, ln_kana)

        fn_lower = (fn or "").lower()
        if fn_lower in KATAKANA_FIRSTNAME_OVERRIDE:
            fn_kana = KATAKANA_FIRSTNAME_OVERRIDE[fn_lower]
        else:
            fn_proc = re.sub(r"ryo(?=([bcdfghjklmnpqrstvwxyz]|$))", "ryou", fn_lower)
            fn_proc = re.sub(r"ryo(?=i)", "ryoui", fn_proc)
            fn_kana = roman_to_katakana(fn_proc, long_vowel)

        return ln_kana, fn_kana, (ln_kana + fn_kana)

    kana_tuple = base["Author"].apply(_kana_parts)
    base["KanaLastName"]  = kana_tuple.apply(lambda t: t[0])
    base["KanaFirstName"] = kana_tuple.apply(lambda t: t[1])
    base["KanaName"]      = kana_tuple.apply(lambda t: t[2])

    out = base.sort_values(["Score", "Author"], ascending=[False, True])
    cols = ["Author", "KanaName", "KanaLastName", "KanaFirstName", "Score"]
    return out[cols]

# --- アプリ本体 ---
if check_password():
    st.title("InsighTCROSS® Literature Scorer")
    st.caption("Europe PMC（APIキー不要）を利用して文献を検索・著者スコア化します。")

    # 認証後のみ表示されるサイドバー（ログアウト＋検索条件）
    with st.sidebar:
        if st.button("ログアウト"):
            st.session_state["password_correct"] = False
            st.rerun()

        st.header("Search Filters")
        disease = st.text_input("Disease (例: peripheral artery disease / heart failure ※引用符は不要)")
        country = st.text_input("Country (例: Japan)")

        col_y1, col_y2 = st.columns(2)
        with col_y1:
            year_from = st.number_input("Year From", min_value=1900, max_value=2100, value=2020, step=1)
        with col_y2:
            year_to = st.number_input("Year To", min_value=1900, max_value=2100, value=2025, step=1)

        dept = st.text_input("Department/Division in Affiliation（例: Cardiology, 循環器内科）")
        keywords = st.text_input("Keywords（例: Cardiology, Interventional）")

        st.markdown("**Text availability**")
        ta_abstract = st.checkbox("Abstract")
        ta_free_full = st.checkbox("Free full text")
        ta_full_text = st.checkbox("Full text (any)")

        st.markdown("**Article type**")
        atypes = st.multiselect(
            "Select article types",
            ["Clinical Trial", "Meta-Analysis", "Randomized Controlled Trial", "Review", "Systematic Review"],
        )

        st.markdown("**Sources**")
        src_med = st.checkbox("PubMed/Medline (SRC:MED)", value=True)
        src_pmc = st.checkbox("PubMed Central (SRC:PMC)", value=False)
        src_ppr = st.checkbox("Preprints (SRC:PPR)", value=False)
        excl_ppr = st.checkbox("Exclude preprints (NOT SRC:PPR)", value=True)

        st.markdown("**Options**")
        use_synonyms = st.checkbox("MeSHシノニムを展開する（広く拾う）", value=False)
        kana_long = st.checkbox("カタカナ：長音（Ko→コウ など）を推定", value=True)
        surname_o = st.checkbox("日本人姓の末尾 to/doを長音化（〜トウ/ドウ）", value=True)
        yuki_long = st.checkbox("FirstName『Yuki』をユウキ扱いにする（男性名想定）", value=False)  # 予約：現行ロジック未使用
        max_rows = st.number_input("Max records to fetch (safety cap)", 1000, 20000, 5000, step=500)
        relax_if_zero = st.checkbox("🔁 0件なら自動で条件をゆるめて再検索", value=True)

        st.divider()
        run = st.button("🔍 Search & Score")

# ===== 検索実行（認証後のみ）=====
if run:
    if not disease and not keywords and not dept:
        st.warning("少なくとも Disease / Keywords / Department のいずれかを入力してください。")
        st.stop()

    # クエリ作成
    active_query = build_query(
        disease, country, year_from, year_to, dept, keywords,
        ta_abstract, ta_free_full, ta_full_text, atypes,
        src_med, src_pmc, src_ppr, excl_ppr,
    )
    st.info(f"Query: {active_query}")

    # 検索
    with st.spinner("Europe PMC を検索中..."):
        try:
            items, total_hits = fetch_eupmc_all(active_query, max_rows=max_rows, synonym=use_synonyms)
        except Exception as e:
            st.error(f"検索でエラーが発生しました: {e}")
            st.stop()

    # 0件なら条件緩和で再検索
    if len(items) == 0 and relax_if_zero:
        st.warning("0件だったため、Keywords / Full text / Article type を外して再検索します…")
        active_query = build_query(
            disease, country, year_from, year_to, dept, "",
            ta_abstract, False, False, [],  # ← 緩和
            src_med, src_pmc, False, True,  # ← プレプリント除外のまま
        )
        st.info(f"Relaxed Query: {active_query}")
        with st.spinner("条件を緩めて再検索中..."):
            try:
                items, total_hits = fetch_eupmc_all(active_query, max_rows=max_rows, synonym=use_synonyms)
            except Exception as e:
                st.error(f"再検索でエラーが発生しました: {e}")
                st.stop()

    # 概要表示
    unique_pmids = len({(it.get("pmid") or it.get("id")) for it in items})
    try:
        med_hits = get_hit_count(f"({active_query}) AND SRC:MED", synonym=use_synonyms)
    except Exception:
        med_hits = None

    msg = f"ヒット件数（API全体）: {total_hits}"
    if med_hits is not None:
        msg += f"  /  PubMed/Medlineのみ: {med_hits}"
    msg += f"  /  取得（上限内）: {len(items)}  /  論文件数（取得分のユニーク）: {unique_pmids}"
    st.success(msg)

    # 著者行に展開
    df = explode_authors(items)

    # 念のため年レンジで最終フィルタ
    def _coerce_year(x):
        try:
            return int(str(x)[:4])
        except Exception:
            return None

    if not df.empty:
        df["YearNum"] = df["Year"].apply(_coerce_year)
        df = df[(df["YearNum"].isna()) | ((df["YearNum"] >= int(year_from)) & (df["YearNum"] <= int(year_to)))]
        df = df.drop(columns=["YearNum"])

    st.subheader("Results (Author-level rows)")
    st.dataframe(df, use_container_width=True, hide_index=True)

    if not df.empty:
        # 明細ダウンロード
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Download author rows (CSV)", csv_bytes,
                           file_name="literature_author_rows.csv", mime="text/csv")

        # スコア集計
        agg = aggregate_author_scores(df, long_vowel=kana_long, surname_long_o=surname_o)
        st.subheader("Author Score (Full name × MainAffiliation)")

        if "ConfirmKana" not in agg.columns:
            agg["ConfirmKana"] = False

        edited = st.data_editor(
            agg,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ConfirmKana": st.column_config.CheckboxColumn(
                    "✓Kana確認", help="カタカナ表記を目視確認したらチェック", default=False
                )
            },
        )
        csv2 = edited.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Download author scores (CSV)", csv2,
                           file_name="literature_author_scores.csv", mime="text/csv")

    st.caption(
        "注: Europe PMC の 'OPEN_ACCESS', 'HAS_FULL_TEXT', 'HAS_ABSTRACT', 'PUB_TYPE', 'SRC' フィールドでフィルタしています。"
        "\nヒット件数は Europe PMC の 'hitCount' を使用。MeSHシノニム展開は 'synonym' オプションでON/OFF可能です（APIキー不要）。"
    )
else:
    st.info("左の条件を入力し、'Search & Score' を押してください。")
