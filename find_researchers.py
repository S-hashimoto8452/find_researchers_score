# Run with:  streamlit run find_researchers.py
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

# --- ログイン（許可メンバー制・Secrets管理） ---
# --- ログイン（許可メンバー制・Secrets管理：堅牢化版） ---
def check_signin():
    """
    Secrets 例:
    [auth]
    allow_users = ["alice@example.com", "bob@example.com", "carol"]
    common_password = "SharedPW"  # or omit and use [auth.users]

    [auth.users]
    alice@example.com = "AlicePW123"
    """
    # 互換: [auth] が無ければ [passwords].app_password へフォールバック（任意）
    if "auth" not in st.secrets:
        # フォールバックを使いたくない場合はここでエラー停止のままでもOK
        if "passwords" in st.secrets and "app_password" in st.secrets["passwords"]:
            return legacy_password_gate()
        st.error("設定エラー：Secretsに [auth] セクションがありません。StreamlitのSecretsで [auth] を設定してください。")
        st.stop()

    auth = st.secrets["auth"]

    # allow_users の取り込みを柔軟に（配列 / カンマ / 全角カンマ / 改行に対応）
    allow_users = set()
    raw_allow = auth.get("allow_users", None)
    if raw_allow is None:
        st.error("設定エラー：Secretsの [auth].allow_users が未設定です。")
        st.stop()

    def _split_any(s: str):
        # カンマ（半角/全角）と改行のどれでも分割
        return [x.strip() for x in re.split(r"[,\u3001\n\r]+", s) if x.strip()]

    if isinstance(raw_allow, (list, tuple)):
        allow_users = {str(u).strip().lower() for u in raw_allow if str(u).strip()}
    elif isinstance(raw_allow, str):
        allow_users = {x.lower() for x in _split_any(raw_allow)}
    else:
        st.error("設定エラー：[auth].allow_users は配列または文字列で指定してください。")
        st.stop()

    # 共通PW or 個別PW
    common_pw = auth.get("common_password", None)
    user_pw_map = dict(st.secrets.get("auth.users", {}))

    # セッション初期化
    ss = st.session_state
    ss.setdefault("signed_in", False)
    ss.setdefault("signin_attempts", 0)

    # 未ログインフォーム
    if not ss.signed_in:
        with st.form("signin_form", clear_on_submit=False):
            user_id = st.text_input("メールまたはユーザーID（許可ユーザーのみ）").strip().lower()
            password = st.text_input("パスワード", type="password")
            submitted = st.form_submit_button("ログイン")

        if submitted:
            if user_id not in allow_users:
                st.error("このユーザーは許可されていません。管理者に連絡してください。")
                return False

            ok = False
            if user_id in user_pw_map:
                ok = (password == str(user_pw_map[user_id]))
            elif common_pw is not None:
                ok = (password == str(common_pw))

            if ok:
                ss.signed_in = True
                ss.user = user_id
                ss.signin_attempts = 0
                st.rerun()
            else:
                ss.signin_attempts += 1
                st.error("IDまたはパスワードが違います。")
                if ss.signin_attempts >= 5:
                    st.warning("試行回数が多すぎます。しばらく時間を空けてから再度お試しください。")
                    st.stop()
            return False

        return False

    return True


def legacy_password_gate():
    """[passwords].app_password 用（後方互換）。[auth] が無いときだけ動作。"""
    pw = st.secrets["passwords"]["app_password"]
    ss = st.session_state
    ss.setdefault("signed_in", False)
    ss.setdefault("signin_attempts", 0)

    if not ss.signed_in:
        with st.form("pw_form"):
            password = st.text_input("パスワード", type="password")
            submitted = st.form_submit_button("ログイン")

        if submitted:
            if password == str(pw):
                ss.signed_in = True
                st.rerun()
            else:
                ss.signin_attempts += 1
                st.error("パスワードが違います。")
                if ss.signin_attempts >= 5:
                    st.warning("試行回数が多すぎます。しばらく時間を空けてください。")
                    st.stop()
            return False
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
    "abe": "アベ",    #追加2
    "abiko": "アビコ", #追加2
    "adachi": "アダチ",#追加2
    "agematsu": "アゲマツ",#追加2
    "aihara": "アイハラ",#追加2
    "aikawa": "アイカワ",#追加2
    "aizawa": "アイザワ",#追加2
    "ajimi": "アジミ",#追加2
    "akabame": "アカバメ",#追加2
    "akagi": "アカギ",#追加2
    "akahori": "アカホリ",#追加2
    "akama": "アカマ",#追加2
    "akamine": "アカミネ",#追加2
    "akao": "アカオ",#追加2
    "akari": "アカリ",#追加2
    "akasaka": "アカサカ",#追加2
    "akase": "アカセ",#追加2
    "akashi": "アカシ",#追加2
    "akazawa": "アカザワ",#追加2
    "akima": "アキマ",#追加2
    "akimaru": "アキマル",#追加2
    "akioka": "アキオカ",#追加2
    "akita": "アキタ",#追加2
    "akiyama": "アキヤマ",#追加2
    "ako": "アコ",#追加2
    "akutsu": "アクツ",#追加2
    "amagaya": "アマガヤ",#追加2
    "amaki": "アマキ",#追加2
    "amanai": "アマナイ",#追加2
    "amano": "アマノ",
    "amano": "アマノ",#追加2
    "amemiya": "アメミヤ",#追加2
    "amisaki": "アミサキ",#追加2
    "anai": "アナイ",#追加2
    "ando": "アンドウ",
    "anzai": "アンザイ",#追加2
    "anzaki": "アンザキ",#追加2
    "aoki": "アオキ",#追加2
    "aono": "アオノ",#追加2
    "aoyama": "アオヤマ",#追加2
    "arai": "アライ",#追加2
    "arakaki": "アラカキ",#追加2
    "araki": "アラキ",#追加2
    "aramaki": "アラマキ",#追加2
    "arao": "アラオ",#追加2
    "arashi": "アラシ",#追加2
    "aratame": "アラタメ",#追加2
    "arikawa": "アリカワ",#追加2
    "arima": "アリマ",#追加2
    "arimoto": "アリモト",#追加2
    "asada": "アサダ",#追加2
    "asai": "アサイ",#追加2
    "asakawa": "アサカワ",#追加2
    "asakura": "アサクラ",#追加2
    "asami": "アサミ",#追加2
    "asano": "アサノ",#追加2
    "asaoka": "アサオカ",#追加2
    "asari": "アサリ",#追加2
    "asaumi": "アサウミ",#追加2
    "ashida": "アシダ",#追加2
    "ashikaga": "アシカガ",#追加2
    "ashikawa": "アシカワ",#追加2
    "asukai": "アスカイ",#追加2
    "atsuchi": "アツチ",#追加2
    "atsumi": "アツミ",#追加2
    "awaya": "アワヤ",#追加2
    "ayabe": "アヤベ",#追加2
    "baba": "ババ",#追加2
    "bando": "バンドウ",#追加2
    "chatani": "チャタニ",#追加2
    "chiba": "チバ",#追加2
    "chida": "チダ",#追加2
    "chikata": "チカタ",#追加2
    "chinen": "チネン",#追加2
    "chishiki": "チシキ",#追加2
    "chisiki": "チシキ",#追加2
    "dai": "ダイ",#追加2
    "daida": "ダイダ",#追加2
    "daimon": "ダイモン",#追加2
    "dan": "ダン",#追加2
    "deguchi": "デグチ",#追加2
    "dochi": "ドチ",#追加2
    "dohi": "ドイ",#追加2
    "doi": "ドイ",#追加2
    "doijiri": "ドイジリ",#追加2
    "domei": "ドウメイ",
    "ebara": "エバラ",#追加2
    "ebato": "エバト",#追加2
    "ebihara": "エビハラ",#追加2
    "ebina": "エビナ",#追加2
    "ebisawa": "エビサワ",#追加2
    "ebuchi": "エブチ",#追加2
    "egami": "エガミ",#追加2
    "ehara": "エハラ",#追加2
    "eizawa": "エイザワ",#追加2
    "ejiri": "エジリ",#追加2
    "emori": "エモリ",#追加2
    "emoto": "エモト",#追加2
    "endo": "エンドウ",#追加2
    "enomoto": "エノモト",#追加2
    "enta": "エンタ",#追加2
    "esaki": "エサキ",#追加2
    "eto": "エトウ",#追加2
    "etoh": "エトウ",#追加2
    "ezure": "エズレ",#追加2
    "fujibayashi": "フジバヤシ",#追加2
    "fujihara": "フジハラ",#追加2
    "fujii": "フジイ",#追加2
    "fujikawa": "フジカワ",#追加2
    "fujima": "フジマ",#追加2
    "fujimi": "フジミ",#追加2
    "fujimiya": "フジミヤ",#追加2
    "fujimoto": "フジモト",#追加2
    "fujimura": "フジムラ",#追加2
    "fujinami": "フジナミ",#追加2
    "fujino": "フジノ",#追加2
    "fujinuma": "フジヌマ",#追加2
    "fujioka": "フジオカ",#追加2
    "fujisaka": "フジサカ",#追加2
    "fujisaki": "フジサキ",#追加2
    "fujisawa": "フジサワ",#追加2
    "fujishima": "フジシマ",#追加2
    "fujisue": "フジスエ",#追加2
    "fujita": "フジタ",#追加2
    "fujito": "フジト",#追加2
    "fujiu": "フジウ",#追加2
    "fujiwara": "フジワラ",#追加2
    "fujiyama": "フジヤマ",#追加2
    "fujiyoshi": "フジヨシ",#追加2
    "fukae": "フカエ",#追加2
    "fukagai": "フカガイ",#追加2
    "fukamachi": "フカマチ",#追加2
    "fukami": "フカミ",#追加2
    "fukamizu": "フカミズ",#追加2
    "fukase": "フカセ",#追加2
    "fukata": "フカタ",#追加2
    "fukazawa": "フカザワ",#追加2
    "fuki": "フキ",#追加2
    "fuku": "フク",#追加2
    "fukuda": "フクダ",#追加2
    "fukuhara": "フクハラ",#追加2
    "fukui": "フクイ",#追加2
    "fukuishi": "フクイシ",#追加2
    "fukuizumi": "フクイズミ",#追加2
    "fukuma": "フクマ",#追加2
    "fukumoto": "フクモト",#追加2
    "fukunaga": "フクナガ",#追加2
    "fukunami": "フクナミ",#追加2
    "fukuoka": "フクオカ",#追加2
    "fukushima": "フクシマ",#追加2
    "fukutomi": "フクトミ",#追加2
    "fukuyado": "フクヤド",#追加2
    "fukuyama": "フクヤマ",#追加2
    "fukuzawa": "フクザワ",#追加2
    "funabiki": "フナビキ",#追加2
    "funaki": "フナキ",#追加2
    "funakubo": "フナクボ",#追加2
    "funamizu": "フナミズ",#追加2
    "funatsu": "フナツ",#追加2
    "funayama": "フナヤマ",#追加2
    "furugori": "フルゴオリ",#追加2
    "furukawa": "フルカワ",#追加2
    "furusawa": "フルサワ",#追加2
    "furushima": "フルシマ",#追加2
    "furusho": "フルショウ",#追加2
    "furuta": "フルタ",#追加2
    "furuya": "フルヤ",#追加2
    "furuyashiki": "フルヤシキ",#追加2
    "fusazaki": "フサザキ",#追加2
    "fuseya": "フセヤ",#追加2
    "fushimi": "フシミ",#追加2
    "gatate": "ガタテ",#追加2
    "gibo": "ギボ",#追加2
    "godo": "ゴウド",#追加2
    "gohbara": "ゴウハラ",
    "gohbara": "ゴオバラ",#追加2
    "gomi": "ゴミ",#追加2
    "goriki": "ゴウリキ",
    "gotanda": "ゴタンダ",#追加2
    "goto": "ゴトウ",#追加2
    "gunji": "グンジ",#追加2
    "haba": "ハバ",#追加2
    "habara": "ハバラ",#追加2
    "hachinohe": "ハチノヘ",#追加2
    "hada": "ハダ",#追加2
    "hadase": "ハダセ",#追加2
    "haga": "ハガ",#追加2
    "hagikura": "ハギクラ",#追加2
    "hagiwara": "ハギワラ",#追加2
    "hagiya": "ハギヤ",#追加2
    "hai": "ハイ",#追加2
    "haishi": "ハイシ",#追加2
    "hakoda": "ハコダ",#追加2
    "hamada": "ハマダ",#追加2
    "hamadate": "ハマダテ",#追加2
    "hamaguchi": "ハマグチ",#追加2
    "hamana": "ハマナ",#追加2
    "hamanaka": "ハマナカ",#追加2
    "hamano": "ハマノ",#追加2
    "hamashige": "ハマシゲ",#追加2
    "hamatani": "ハマタニ",#追加2
    "hamaya": "ハマヤ",#追加2
    "hamazaki": "ハマザキ",#追加2
    "hanada": "ハナダ",#追加2
    "hanajima": "ハナジマ",#追加2
    "hanaoka": "ハナオカ",#追加2
    "hanatani": "ハナタニ",#追加2
    "hanyu": "ハニュウ",
    "hao": "ハオ",#追加2
    "hara": "ハラ",#追加2
    "harada": "ハラダ",#追加2
    "haruki": "ハルキ",#追加2
    "haruta": "ハルタ",#追加2
    "hasebe": "ハセベ",#追加2
    "hasegawa": "ハセガワ",#追加2
    "hashiba": "ハシバ",#追加2
    "hashikata": "ハシカタ",#追加2
    "hashimoto": "ハシモト",#追加2
    "hata": "ハタ",#追加2
    "hatori": "ハットリ",#追加2
    "hattori": "ハットリ",#追加2
    "hayakawa": "ハヤカワ",#追加2
    "hayama": "ハヤマ",#追加2
    "hayasaka": "ハヤサカ",#追加2
    "hayase": "ハヤセ",#追加2
    "hayashi": "ハヤシ",#追加2
    "hayashida": "ハヤシダ",#追加2
    "hayatsu": "ハヤツ",#追加2
    "hibi": "ヒビ",#追加2
    "hibino": "ヒビノ",#追加2
    "hida": "ヒダ",#追加2
    "hidaka": "ヒダカ",#追加2
    "hieda": "ヒエダ",#追加2
    "hifumi": "ヒフミ",#追加2
    "higa": "ヒガ",#追加2
    "higaki": "ヒガキ",#追加2
    "higami": "ヒガミ",#追加2
    "higashida": "ヒガシダ",#追加2
    "higashihara": "ヒガシハラ",#追加2
    "higashikuni": "ヒガシクニ",#追加2
    "higashino": "ヒガシノ",#追加2
    "higashioka": "ヒガシオカ",#追加2
    "higashiya": "ヒガシヤ",#追加2
    "higo": "ヒゴ",#追加2
    "higuchi": "ヒグチ",#追加2
    "higuma": "ヒグマ",#追加2
    "hiki": "ヒキ",#追加2
    "hikichi": "ヒキチ",#追加2
    "hikida": "ヒキダ",#追加2
    "hikita": "ヒキタ",#追加2
    "hikoso": "ヒコソ",#追加2
    "himi": "ヒミ",#追加2
    "himuro": "ヒムロ",#追加2
    "hino": "ヒノ",#追加2
    "hiraga": "ヒラガ",#追加2
    "hirai": "ヒライ",#追加2
    "hiraide": "ヒライデ",#追加2
    "hiraishi": "ヒライシ",#追加2
    "hiraiwa": "ヒライワ",#追加2
    "hirakawa": "ヒラカワ",#追加2
    "hiramatsu": "ヒラマツ",#追加2
    "hiramori": "ヒラモリ",#追加2
    "hirano": "ヒラノ",#追加2
    "hirao": "ヒラオ",#追加2
    "hiraoka": "ヒラオカ",#追加2
    "hirase": "ヒラセ",#追加2
    "hirashima": "ヒラシマ",#追加2
    "hirata": "ヒラタ",#追加2
    "hiraya": "ヒラヤ",#追加2
    "hirayama": "ヒラヤマ",#追加2
    "hirayu": "ヒラユ",#追加2
    "hiro": "ヒロ",#追加2
    "hirofuji": "ヒロフジ",#追加2
    "hirofuzi": "ヒロフジ",#追加2
    "hirohata": "ヒロハタ",#追加2
    "hiromasa": "ヒロマサ",#追加2
    "hironaga": "ヒロナガ",#追加2
    "hirooka": "ヒロオカ",#追加2
    "hirose": "ヒロセ",#追加2
    "hirota": "ヒロタ",#追加2
    "hiruma": "ヒルマ",#追加2
    "hisadome": "ヒサドメ",#追加2
    "hisamune": "ヒサムネ",#追加2
    "hisanaga": "ヒサナガ",#追加2
    "hisauchi": "ヒサウチ",#追加2
    "hishikari": "ヒシカリ",#追加2
    "hitomi": "ヒトミ",#追加2
    "hoji": "ホジ",#追加2
    "hojo": "ホウジョウ",#追加2
    "hokama": "ホカマ",#追加2
    "hokimoto": "ホキモト",#追加2
    "homma": "ホンマ",
    "honda": "ホンダ",#追加2
    "honde": "ホンデ",#追加2
    "hondera": "ホンデラ",#追加2
    "hongo": "ホンゴウ",#追加2
    "honye": "ホンエ",
    "hori": "ホリ",#追加2
    "horibe": "ホリベ",#追加2
    "horie": "ホリエ",#追加2
    "horii": "ホリイ",#追加2
    "horikoshi": "ホリコシ",#追加2
    "horimoto": "ホリモト",#追加2
    "horio": "ホリオ",#追加2
    "horita": "ホリタ",#追加2
    "horiuchi": "ホリウチ",#追加2
    "hoshi": "ホシ",#追加2
    "hoshiga": "ホシガ",#追加2
    "hoshika": "ホシカ",#追加2
    "hoshina": "ホシナ",#追加2
    "hoshino": "ホシノ",#追加2
    "hoshiyama": "ホシヤマ",#追加2
    "hosoda": "ホソダ",#追加2
    "hosoe": "ホソエ",#追加2
    "hosogi": "ホソギ",#追加2
    "hosoi": "ホソイ",#追加2
    "hosokawa": "ホソカワ",#追加2
    "hosoya": "ホソヤ",#追加2
    "hosozawa": "ホソザワ",#追加2
    "hotsuki": "ホツキ",#追加2
    "hotta": "ホッタ",#追加2
    "hoyano": "ホヤノ",#追加2
    "hozawa": "ホザワ",#追加2
    "hyodo": "ヒョオドウ",#追加2
    "ibaraki": "イバラキ",#追加2
    "ichibori": "イチボリ",#追加2
    "ichihara": "イチハラ",#追加2
    "ichikawa": "イチカワ",#追加2
    "ichinokawa": "イチノカワ",#追加2
    "ichioka": "イチオカ",#追加2
    "ida": "イダ",#追加2
    "ide": "イデ",#追加2
    "ideguchi": "イデグチ",#追加2
    "ieda": "イエダ",#追加2
    "iehara": "イエハラ",#追加2
    "igarashi": "イガラシ",#追加2
    "igawa": "イガワ",#追加2
    "igeta": "イゲタ",#追加2
    "iha": "イハ",#追加2
    "ihara": "イハラ",#追加2
    "iida": "イイダ",#追加2
    "iihara": "イイハラ",#追加2
    "iijima": "イイジマ",#追加2
    "iimori": "イイモリ",#追加2
    "iino": "イイノ",#追加2
    "iioka": "イイオカ",#追加2
    "iiya": "イイヤ",#追加2
    "ijuin": "イジュウイン",#追加2
    "ikai": "イカイ",#追加2
    "ikari": "イカリ",#追加2
    "ike": "イケ",#追加2
    "ikebe": "イケベ",#追加2
    "ikeda": "イケダ",#追加2
    "ikegami": "イケガミ",#追加2
    "ikemiyagi": "イケミヤギ",#追加2
    "ikemura": "イケムラ",#追加2
    "ikenaga": "イケナガ",#追加2
    "ikeno": "イケノ",#追加2
    "ikeoka": "イケオカ",#追加2
    "ikeuchi": "イケウチ",#追加2
    "ikuchi": "イクチ",#追加2
    "ikuta": "イクタ",#追加2
    "imada": "イマダ",#追加2
    "imagawa": "イマガワ",#追加2
    "imai": "イマイ",#追加2
    "imaizumi": "イマイズミ",#追加2
    "imamura": "イマムラ",#追加2
    "imanaka": "イマナカ",#追加2
    "imaoka": "イマオカ",#追加2
    "imori": "イモリ",#追加2
    "inaba": "イナバ",#追加2
    "inada": "イナダ",#追加2
    "inagaki": "イナガキ",#追加2
    "inden": "インデン",#追加2
    "ino": "イノウ",#追加2
    "inohara": "イノハラ",#追加2
    "inoko": "イノコ",#追加2
    "inomata": "イノマタ",#追加2
    "inoue": "イノウエ",#追加2
    "inuzuka": "イヌズカ",#追加2
    "ioji": "イオジ",#追加2
    "irie": "イリエ",#追加2
    "iritani": "イリタニ",#追加2
    "isawa": "イサワ",#追加2
    "iseki": "イセキ",#追加2
    "ishi": "イシイ",#追加2
    "ishibashi": "イシバシ",#追加2
    "ishibuchi": "イシブチ",#追加2
    "ishida": "イシダ",#追加2
    "ishidoya": "イシドヤ",#追加2
    "ishigaki": "イシガキ",#追加2
    "ishigami": "イシガミ",#追加2
    "ishiguro": "イシグロ",#追加2
    "ishihara": "イシハラ",#追加2
    "ishii": "イシイ",#追加2
    "ishikawa": "イシカワ",#追加2
    "ishimura": "イシムラ",#追加2
    "ishino": "イシノ",#追加2
    "ishio": "イシオ",#追加2
    "ishisone": "イシソネ",#追加2
    "ishiwata": "イシワタ",#追加2
    "ishizawa": "イシザワ",#追加2
    "ishizu": "イシズ",#追加2
    "iso": "イソ",#追加2
    "isobe": "イソベ",#追加2
    "isoda": "イソダ",#追加2
    "isodono": "イソドノ",#追加2
    "isogai": "イソガイ",#追加2
    "isomura": "イソムラ",#追加2
    "isozaki": "イソザキ",#追加2
    "isshiki": "イッシキ",#追加2
    "itabashi": "イタバシ",#追加2
    "itagaki": "イタガキ",#追加2
    "itaya": "イタヤ",#追加2
    "ito": "イトウ",
    "itoh": "イトウ",#追加2
    "itokawa": "イトカワ",#追加2
    "itoshima": "イトシマ",#追加2
    "iwabe": "イワベ",#追加2
    "iwabuchi": "イワブチ",#追加2
    "iwagami": "イワガミ",#追加2
    "iwahashi": "イワハシ",#追加2
    "iwahori": "イワホリ",#追加2
    "iwai": "イワイ",#追加2
    "iwakura": "イワクラ",#追加2
    "iwama": "イワマ",#追加2
    "iwamoto": "イワモト",#追加2
    "iwanaga": "イワナガ",#追加2
    "iwanami": "イワナミ",#追加2
    "iwane": "イワネ",#追加2
    "iwasa": "イワサ",#追加2
    "iwasaki": "イワサキ",#追加2
    "iwata": "イワタ",#追加2
    "iwatsubo": "イワツボ",#追加2
    "iwayama": "イワヤマ",#追加2
    "izawa": "イザワ",#追加2
    "izumi": "イズミ",#追加2
    "izumikawa": "イズミカワ",#追加2
    "izumiya": "イズミヤ",#追加2
    "izumo": "イズモ",#追加2
    "jinno": "ジンノ",#追加2
    "jinnouchi": "ジンノウチ",#追加2
    "jojima": "ジョウジマ",#追加2
    "joki": "ジョウキ",#追加2
    "jujo": "ジュウジョウ",
    "kabutoya": "カブトヤ",#追加2
    "kachi": "カチ",#追加2
    "kadokami": "カドカミ",#追加2
    "kadooka": "カドオカ",#追加2
    "kadota": "カドタ",#追加2
    "kadotani": "カドタニ",#追加2
    "kaga": "カガ",#追加2
    "kagawa": "カガワ",#追加2
    "kageyama": "カゲヤマ",#追加2
    "kai": "カイ",#追加2
    "kaichi": "カイチ",#追加2
    "kaikita": "カイキタ",#追加2
    "kainuma": "カイヌマ",#追加2
    "kaitani": "カイタニ",#追加2
    "kaji": "カジ",#追加2
    "kajihara": "カジハラ",#追加2
    "kajikawa": "カジカワ",#追加2
    "kajimoto": "カジモト",#追加2
    "kajinami": "カジナミ",#追加2
    "kajiwara": "カジワラ",#追加2
    "kajiya": "カジヤ",#追加2
    "kakazu": "カカズ",#追加2
    "kakei": "カケイ",#追加2
    "kakimoto": "カキモト",#追加2
    "kakizaki": "カキザキ",#追加2
    "kaku": "カク",#追加2
    "kakumori": "カクモリ",#追加2
    "kakuta": "カクタ",#追加2
    "kamada": "カマダ",#追加2
    "kamagata": "カマガタ",#追加2
    "kamba": "カンバ",#追加2
    "kameda": "カメダ",#追加2
    "kameshima": "カメシマ",#追加2
    "kameyama": "カメヤマ",#追加2
    "kamigaki": "カミガキ",#追加2
    "kamikawa": "カミカワ",#追加2
    "kamisago": "カミサゴ",#追加2
    "kamishima": "カミシマ",#追加2
    "kamishita": "カミシタ",#追加2
    "kamiunten": "カミウンテン",#追加2
    "kamiya": "カミヤ",#追加2
    "kamon": "カモン",#追加2
    "kanaji": "カナジ",#追加2
    "kanamori": "カナモリ",#追加2
    "kanaoka": "カナオカ",#追加2
    "kanaya": "カナヤ",#追加2
    "kanazawa": "カナザワ",#追加2
    "kanda": "カンダ",#追加2
    "kandori": "カンドリ",#追加2
    "kaneda": "カネダ",#追加2
    "kanegawa": "カネガワ",#追加2
    "kanehama": "カネハマ",#追加2
    "kaneko": "カネコ",#追加2
    "kanemitsu": "カネミツ",#追加2
    "kanemura": "カネムラ",#追加2
    "kanenawa": "カネナワ",#追加2
    "kanie": "カニエ",#追加2
    "kanno": "カンノ",#追加2
    "kanzaki": "カンザキ",#追加2
    "karasawa": "カラサワ",#追加2
    "kario": "カリオ",#追加2
    "kasahara": "カサハラ",#追加2
    "kasai": "カサイ",#追加2
    "kasama": "カサマ",#追加2
    "kashima": "カシマ",#追加2
    "kashimura": "カシムラ",#追加2
    "kashiwagi": "カシワギ",#追加2
    "kashiyama": "カシヤマ",#追加2
    "kasuga": "カスガ",#追加2
    "katagata": "カタガタ",#追加2
    "katagiri": "カタギリ",#追加2
    "katahira": "カタヒラ",#追加2
    "katamine": "カタミネ",#追加2
    "kataoka": "カタオカ",#追加2
    "katawaki": "カタワキ",#追加2
    "katayama": "カタヤマ",#追加2
    "kato": "カトウ",
    "katoh": "カトウ",#追加2
    "katsuki": "カツキ",#追加2
    "katsumata": "カツマタ",#追加2
    "katsura": "カツラ",#追加2
    "katsushika": "カツシカ",#追加2
    "kawachi": "カワチ",#追加2
    "kawada": "カワダ",#追加2
    "kawagoe": "カワゴエ",#追加2
    "kawaguchi": "カワグチ",#追加2
    "kawahara": "カワハラ",#追加2
    "kawahatsu": "カワハツ",#追加2
    "kawahira": "カワヒラ",#追加2
    "kawai": "カワイ",#追加2
    "kawakami": "カワカミ",#追加2
    "kawakubo": "カワクボ",#追加2
    "kawamori": "カワモリ",#追加2
    "kawamoto": "カワモト",#追加2
    "kawamura": "カワムラ",#追加2
    "kawanami": "カワナミ",#追加2
    "kawanishi": "カワニシ",#追加2
    "kawano": "カワノ",#追加2
    "kawasaki": "カワサキ",#追加2
    "kawase": "カワセ",#追加2
    "kawashima": "カワシマ",#追加2
    "kawasoe": "カワソエ",#追加2
    "kawata": "カワタ",#追加2
    "kawauchi": "カワウチ",#追加2
    "kawaura": "カワウラ",#追加2
    "kayanuma": "カヤヌマ",#追加2
    "kazama": "カザマ",#追加2
    "kazui": "カズイ",#追加2
    "kazuya": "カズヤ",#追加2
    "keira": "ケイラ",#追加2
    "kibe": "キベ",#追加2
    "kida": "キダ",#追加2
    "kido": "キド",#追加2
    "kijima": "キジマ",#追加2
    "kiko": "キコ",#追加2
    "kikuchi": "キクチ",#追加2
    "kikuta": "キクタ",#追加2
    "kimishima": "キミシマ",#追加2
    "kimura": "キムラ",#追加2
    "kinjo": "キンジョウ",#追加2
    "kinoshita": "キノシタ",#追加2
    "kinugawa": "キヌガワ",#追加2
    "kira": "キラ",#追加2
    "kirigaya": "キリガヤ",#追加2
    "kiriyama": "キリヤマ",#追加2
    "kise": "キセ",#追加2
    "kishi": "キシ",#追加2
    "kishima": "キシマ",#追加2
    "kishimoto": "キシモト",#追加2
    "kishino": "キシノ",#追加2
    "kishita": "キシタ",#追加2
    "kita": "キタ",#追加2
    "kitabata": "キタバタ",#追加2
    "kitada": "キタダ",#追加2
    "kitagawa": "キタガワ",#追加2
    "kitahara": "キタハラ",#追加2
    "kitai": "キタイ",#追加2
    "kitajima": "キタジマ",#追加2
    "kitakaze": "キタカゼ",#追加2
    "kitamura": "キタムラ",#追加2
    "kitani": "キタニ",#追加2
    "kitano": "キタノ",#追加2
    "kitao": "キタオ",#追加2
    "kitaoka": "キタオカ",#追加2
    "kitayama": "キタヤマ",#追加2
    "kitazono": "キタゾノ",#追加2
    "kitsuka": "キツカ",#追加2
    "kiuchi": "キウチ",#追加2
    "kiyohara": "キヨハラ",#追加2
    "kiyono": "キヨノ",#追加2
    "kiyoshige": "キヨシゲ",#追加2
    "kiyosue": "キヨスエ",#追加2
    "koba": "コバ",#追加2
    "kobara": "コバラ",#追加2
    "kobata": "コバタ",#追加2
    "kobayashi": "コバヤシ",#追加2
    "kodaira": "コダイラ",#追加2
    "kodama": "コダマ",#追加2
    "kodera": "コデラ",#追加2
    "koeda": "コエダ",#追加2
    "koezuka": "コエズカ",#追加2
    "koga": "コガ",#追加2
    "kogame": "コガメ",#追加2
    "kogo": "コウゴ",#追加2
    "kogure": "コグレ",#追加2
    "kohjitani": "コウジタニ",#追加2
    "kohno": "コウノ",#追加2
    "kohro": "コウロ",
    "kohsaka": "コウサカ",
    "kohyama": "コウヤマ",#追加2
    "koide": "コイデ",#追加2
    "koido": "コイド",#追加2
    "koike": "コイケ",#追加2
    "koitabashi": "コイタバシ",#追加2
    "koiwa": "コイワ",#追加2
    "koiwaya": "コイワヤ",#追加2
    "koizumi": "コイズミ",#追加2
    "kojima": "コジマ",#追加2
    "kokubu": "コクブ",#追加2
    "komaki": "コマキ",#追加2
    "komasa": "コマサ",#追加2
    "komatsu": "コマツ",#追加2
    "komiya": "コミヤ",#追加2
    "komiyama": "コミヤマ",#追加2
    "komooka": "コモオカ",#追加2
    "komori": "コモリ",#追加2
    "komukai": "コムカイ",#追加2
    "komuro": "コムロ",#追加2
    "kondo": "コンドウ",#追加2
    "kongoji": "コンゴウジ",#追加2
    "konishi": "コニシ",#追加2
    "konno": "コンノ",#追加2
    "konta": "コンタ",#追加2
    "korematsu": "コレマツ",#追加2
    "kosedo": "コセドウ",#追加2
    "kosuga": "コスガ",#追加2
    "kosuge": "コスゲ",#追加2
    "kosugi": "コスギ",#追加2
    "kotani": "コタニ",#追加2
    "kotoku": "コウトク",#追加2
    "koura": "コウラ",#追加2
    "kowatari": "コワタリ",#追加2
    "koyama": "コヤマ",#追加2
    "koyanagi": "コヤナギ",#追加2
    "kozuka": "コズカ",#追加2
    "kozuki": "コウズキ",#追加2
    "kozuma": "コウズマ",
    "kozuma": "コオズマ",#追加2
    "kubo": "クボ",#追加2
    "kubokawa": "クボカワ",#追加2
    "kubota": "クボタ",#追加2
    "kubozono": "クボゾノ",#追加2
    "kudo": "クドウ",#追加2
    "kuga": "クガ",#追加2
    "kugiyama": "クギヤマ",#追加2
    "kuji": "クジ",#追加2
    "kujiraoka": "クジラオカ",#追加2
    "kumada": "クマダ",#追加2
    "kumagai": "クマガイ",#追加2
    "kumamaru": "クママル",#追加2
    "kume": "クメ",#追加2
    "kunieda": "クニエダ",#追加2
    "kunii": "クニイ",#追加2
    "kunimasa": "クニマサ",#追加2
    "kunimoto": "クニモト",#追加2
    "kunimura": "クニムラ",#追加2
    "kunioka": "クニオカ",#追加2
    "kunisawa": "クニサワ",#追加2
    "kuno": "クノ",#追加2
    "kurahashi": "クラハシ",#追加2
    "kuramitsu": "クラミツ",#追加2
    "kuraoka": "クラオカ",#追加2
    "kurata": "クラタ",#追加2
    "kure": "クレ",#追加2
    "kuribara": "クリバラ",#追加2
    "kurihara": "クリハラ",#追加2
    "kurimoto": "クリモト",#追加2
    "kurita": "クリタ",#追加2
    "kuriyama": "クリヤマ",#追加2
    "kurobe": "クロベ",#追加2
    "kuroda": "クロダ",#追加2
    "kurogi": "クロギ",#追加2
    "kuroi": "クロイ",#追加2
    "kuroita": "クロイタ",#追加2
    "kurosaki": "クロサキ",#追加2
    "kurosawa": "クロサワ",#追加2
    "kurozumi": "クロズミ",#追加2
    "kusachi": "クサチ",#追加2
    "kushida": "クシダ",#追加2
    "kusuda": "クスダ",#追加2
    "kusumoto": "クスモト",#追加2
    "kusunose": "クスノセ",#追加2
    "kusuyama": "クスヤマ",#追加2
    "kutsuzawa": "クツザワ",#追加2
    "kuwabara": "クワバラ",#追加2
    "kuwahara": "クワハラ",#追加2
    "kuwano": "クワノ",#追加2
    "kuwata": "クワタ",#追加2
    "kuyama": "クヤマ",#追加2
    "kyodo": "キョウドウ",#追加2
    "long": "ロン",#追加2
    "mabuchi": "マブチ",#追加2
    "machida": "マチダ",#追加2
    "machino": "マチノ",#追加2
    "maeba": "マエバ",#追加2
    "maeda": "マエダ",#追加2
    "maehara": "マエハラ",#追加2
    "maejima": "マエジマ",#追加2
    "maekawa": "マエカワ",#追加2
    "maemura": "マエムラ",#追加2
    "maenaka": "マエナカ",#追加2
    "magota": "マゴタ",#追加2
    "makabe": "マカベ",#追加2
    "makimoto": "マキモト",#追加2
    "makino": "マキノ",#追加2
    "manabe": "マナベ",#追加2
    "mano": "マノ",#追加2
    "marubayashi": "マルバヤシ",#追加2
    "maruhashi": "マルハシ",#追加2
    "marui": "マルイ",#追加2
    "maruo": "マルオ",#追加2
    "maruta": "マルタ",#追加2
    "maruyama": "マルヤマ",#追加2
    "masada": "マサダ",#追加2
    "masaki": "マサキ",#追加2
    "masamura": "マサムラ",#追加2
    "masawa": "マサワ",#追加2
    "mase": "マセ",#追加2
    "mastui": "マツイ",#追加2
    "mastuoka": "マツオカ",#追加2
    "masuda": "マスダ",#追加2
    "masumura": "マスムラ",#追加2
    "masuyama": "マスヤマ",#追加2
    "matama": "マタマ",#追加2
    "matoba": "マトバ",#追加2
    "matsubara": "マツバラ",#追加2
    "matsubayashi": "マツバヤシ",#追加2
    "matsuda": "マツダ",#追加2
    "matsue": "マツエ",#追加2
    "matsuhama": "マツハマ",#追加2
    "matsuhashi": "マツハシ",#追加2
    "matsuhiro": "マツヒロ",#追加2
    "matsuhisa": "マツヒサ",#追加2
    "matsui": "マツイ",#追加2
    "matsukage": "マツカゲ",#追加2
    "matsukawa": "マツカワ",#追加2
    "matsukura": "マツクラ",#追加2
    "matsumaru": "マツマル",#追加2
    "matsumiya": "マツミヤ",#追加2
    "matsumoto": "マツモト",#追加2
    "matsumura": "マツムラ",#追加2
    "matsuna": "マツナ",#追加2
    "matsunaga": "マツナガ",#追加2
    "matsuno": "マツノ",#追加2
    "matsuo": "マツオ",#追加2
    "matsuoka": "マツオカ",#追加2
    "matsusaka": "マツサカ",#追加2
    "matsushima": "マツシマ",#追加2
    "matsushita": "マツシタ",#追加2
    "matsutani": "マツタニ",#追加2
    "matsuura": "マツウラ",#追加2
    "matsuwaki": "マツワキ",#追加2
    "matsuyama": "マツヤマ",#追加2
    "matsuzaki": "マツザキ",#追加2
    "matsuzawa": "マツザワ",#追加2
    "metoki": "メトキ",#追加2
    "mibiki": "ミビキ",#追加2
    "michikura": "ミチクラ",#追加2
    "michishita": "ミチシタ",#追加2
    "michiura": "ミチウラ",#追加2
    "michiwaki": "ミチワキ",#追加2
    "migita": "ミギタ",#追加2
    "mihara": "ミハラ",#追加2
    "mikami": "ミカミ",#追加2
    "miki": "ミキ",#追加2
    "minagawa": "ミナガワ",#追加2
    "minami": "ミナミ",#追加2
    "minamimoto": "ミナミモト",#追加2
    "minamino": "ミナミノ",#追加2
    "minamisawa": "ミナミサワ",#追加2
    "minatoguchi": "ミナトグチ",#追加2
    "minatoya": "ミナトヤ",#追加2
    "minatsuki": "ミナツキ",#追加2
    "mineki": "ミネキ",#追加2
    "minematsu": "ミネマツ",#追加2
    "mineo": "ミネオ",#追加2
    "misaka": "ミサカ",#追加2
    "misawa": "ミサワ",#追加2
    "mishima": "ミシマ",#追加2
    "misumi": "ミスミ",#追加2
    "mita": "ミタ",#追加2
    "mitamura": "ミタムラ",#追加2
    "mito": "ミト",#追加2
    "mitomo": "ミトモ",#追加2
    "mitsuba": "ミツバ",#追加2
    "mitsudo": "ミツドウ",
    "mitsuhara": "ミツハラ",#追加2
    "mitsuhashi": "ミツハシ",#追加2
    "mitsui": "ミツイ",#追加2
    "mitsuishi": "ミツイシ",#追加2
    "mitsuke": "ミツケ",#追加2
    "mitsumata": "ミツマタ",#追加2
    "miura": "ミウラ",#追加2
    "miwa": "ミワ",#追加2
    "miyabe": "ミヤベ",#追加2
    "miyachi": "ミヤチ",#追加2
    "miyagawa": "ミヤガワ",#追加2
    "miyagi": "ミヤギ",#追加2
    "miyahara": "ミヤハラ",#追加2
    "miyaji": "ミヤジ",#追加2
    "miyajima": "ミヤジマ",#追加2
    "miyake": "ミヤケ",#追加2
    "miyama": "ミヤマ",#追加2
    "miyamoto": "ミヤモト",#追加2
    "miyanaga": "ミヤナガ",#追加2
    "miyasaka": "ミヤサカ",#追加2
    "miyashita": "ミヤシタ",#追加2
    "miyata": "ミヤタ",#追加2
    "miyauchi": "ミヤウチ",#追加2
    "miyazaki": "ミヤザキ",#追加2
    "miyazawa": "ミヤザワ",#追加2
    "miyosawa": "ミヨサワ",#追加2
    "miyoshi": "ミヨシ",#追加2
    "mizobuchi": "ミゾブチ",#追加2
    "mizokami": "ミゾカミ",#追加2
    "mizota": "ミゾタ",#追加2
    "mizote": "ミゾテ",#追加2
    "mizuguchi": "ミズグチ",#追加2
    "mizukami": "ミズカミ",#追加2
    "mizuno": "ミズノ",#追加2
    "mizunoya": "ミズノヤ",#追加2
    "mizusawa": "ミズサワ",#追加2
    "mizutani": "ミズタニ",#追加2
    "mochidome": "モチドメ",#追加2
    "mochizuki": "モチズキ",#追加2
    "mogi": "モギ",#追加2
    "momoi": "モモイ",#追加2
    "monden": "モンデン",#追加2
    "moniwa": "モニワ",#追加2
    "mori": "モリ",#追加2
    "moribayashi": "モリバヤシ",#追加2
    "moriishi": "モリイシ",#追加2
    "morikawa": "モリカワ",#追加2
    "morimoto": "モリモト",#追加2
    "morino": "モリノ",
    "morioka": "モリオカ",#追加2
    "morisaki": "モリサキ",#追加2
    "morishige": "モリシゲ",#追加2
    "morishima": "モリシマ",#追加2
    "morishita": "モリシタ",#追加2
    "morita": "モリタ",#追加2
    "moriuchi": "モリウチ",#追加2
    "moriwaki": "モリワキ",#追加2
    "moriya": "モリヤ",#追加2
    "moriyama": "モリヤマ",#追加2
    "morofuji": "モロフジ",#追加2
    "morota": "モロタ",#追加2
    "motohashi": "モトハシ",#追加2
    "motoki": "モトキ",#追加2
    "motooka": "モトオカ",#追加2
    "motoshima": "モトシマ",#追加2
    "motoyama": "モトヤマ",#追加2
    "motozawa": "モトザワ",#追加2
    "mozawa": "モザワ",#追加2
    "mukai": "ムカイ",#追加2
    "mukaida": "ムカイダ",#追加2
    "munakata": "ムナカタ",#追加2
    "munehisa": "ムネヒサ",#追加2
    "murai": "ムライ",#追加2
    "muraishi": "ムライシ",#追加2
    "murakami": "ムラカミ",#追加2
    "murakawa": "ムラカワ",#追加2
    "muramatsu": "ムラマツ",#追加2
    "muraoka": "ムラオカ",#追加2
    "murasato": "ムラサト",#追加2
    "murase": "ムラセ",#追加2
    "murata": "ムラタ",#追加2
    "muro": "ムロ",#追加2
    "murohara": "ムロハラ",#追加2
    "murohashi": "ムロハシ",#追加2
    "murotani": "ムロタニ",#追加2
    "muroya": "ムロヤ",#追加2
    "mushiake": "ムシアケ",#追加2
    "muto": "ムトウ",#追加2
    "mutoh": "ムトウ",#追加2
    "mutsuga": "ムツガ",#追加2
    "myojin": "ミョジン",#追加2
    "nabeta": "ナベタ",#追加2
    "nagae": "ナガエ",#追加2
    "nagahara": "ナガハラ",#追加2
    "nagai": "ナガイ",#追加2
    "nagamatsu": "ナガマツ",#追加2
    "nagamine": "ナガミネ",#追加2
    "naganawa": "ナガナワ",#追加2
    "nagano": "ナガノ",#追加2
    "naganuma": "ナガヌマ",#追加2
    "nagao": "ナガオ",#追加2
    "nagasaka": "ナガサカ",#追加2
    "nagasawa": "ナガサワ",#追加2
    "nagase": "ナガセ",#追加2
    "nagata": "ナガタ",#追加2
    "nagatomo": "ナガトモ",#追加2
    "nagoshi": "ナゴシ",#追加2
    "nagumo": "ナグモ",#追加2
    "naito": "ナイトウ",#追加2
    "naka": "ナカ",#追加2
    "nakachi": "ナカチ",#追加2
    "nakada": "ナカダ",#追加2
    "nakagami": "ナカガミ",#追加2
    "nakagata": "ナカガタ",#追加2
    "nakagawa": "ナカガワ",#追加2
    "nakahara": "ナカハラ",#追加2
    "nakahashi": "ナカハシ",#追加2
    "nakai": "ナカイ",#追加2
    "nakajima": "ナカジマ",#追加2
    "nakama": "ナカマ",#追加2
    "nakamae": "ナカマエ",#追加2
    "nakamaru": "ナカマル",#追加2
    "nakamura": "ナカムラ",#追加2
    "nakanishi": "ナカニシ",#追加2
    "nakano": "ナカノ",#追加2
    "nakao": "ナカオ",#追加2
    "nakaoka": "ナカオカ",#追加2
    "nakase": "ナカセ",#追加2
    "nakashima": "ナカシマ",#追加2
    "nakasuka": "ナカスカ",#追加2
    "nakata": "ナカタ",#追加2
    "nakatani": "ナカタニ",#追加2
    "nakatsuma": "ナカツマ",#追加2
    "nakaura": "ナカウラ",#追加2
    "nakayama": "ナカヤマ",#追加2
    "nakayoshi": "ナカヨシ",#追加2
    "nakazato": "ナカザト",#追加2
    "nakazawa": "ナカザワ",#追加2
    "namba": "ナンバ",#追加2
    "namiki": "ナミキ",#追加2
    "namiuchi": "ナミウチ",#追加2
    "nanao": "ナナオ",#追加2
    "nanasato": "ナナサト",
    "nangoya": "ナンゴヤ",#追加2
    "naniwa": "ナニワ",#追加2
    "nanto": "ナント",#追加2
    "narita": "ナリタ",#追加2
    "narui": "ナルイ",#追加2
    "narukawa": "ナルカワ",#追加2
    "naruse": "ナルセ",#追加2
    "nasu": "ナス",#追加2
    "natsuaki": "ナツアキ",#追加2
    "natsukawa": "ナツカワ",#追加2
    "natsumeda": "ナツメダ",#追加2
    "nawada": "ナワダ",#追加2
    "negishi": "ネギシ",#追加2
    "neishi": "ネイシ",#追加2
    "niida": "ニイダ",#追加2
    "niimi": "ニイミ",#追加2
    "niinami": "ニイナミ",#追加2
    "niitsuma": "ニイツマ",#追加2
    "niiyama": "ニイヤマ",#追加2
    "niizeki": "ニイゼキ",#追加2
    "nikaido": "ニカイドウ",#追加2
    "ninomiya": "ニノミヤ",#追加2
    "nishi": "ニシ",#追加2
    "nishida": "ニシダ",#追加2
    "nishiguchi": "ニシグチ",#追加2
    "nishihata": "ニシハタ",#追加2
    "nishihira": "ニシヒラ",#追加2
    "nishijima": "ニシジマ",#追加2
    "nishikawa": "ニシカワ",#追加2
    "nishikura": "ニシクラ",#追加2
    "nishimiya": "ニシミヤ",#追加2
    "nishimizu": "ニシミズ",#追加2
    "nishimori": "ニシモリ",#追加2
    "nishimoto": "ニシモト",#追加2
    "nishimura": "ニシムラ",#追加2
    "nishina": "ニシナ",#追加2
    "nishino": "ニシノ",#追加2
    "nishio": "ニシオ",#追加2
    "nishioka": "ニシオカ",#追加2
    "nishiura": "ニシウラ",#追加2
    "nishiya": "ニシヤ",#追加2
    "nishiyama": "ニシヤマ",#追加2
    "nishizaki": "ニシザキ",#追加2
    "nishizawa": "ニシザワ",#追加2
    "nitta": "ニッタ",#追加2
    "niwa": "ニワ",#追加2
    "nobuta": "ノブタ",#追加2
    "nochioka": "ノチオカ",#追加2
    "noda": "ノダ",#追加2
    "node": "ノデ",#追加2
    "nogami": "ノガミ",#追加2
    "nogi": "ノギ",#追加2
    "noguchi": "ノグチ",#追加2
    "nohara": "ノハラ",#追加2
    "noike": "ノイケ",#追加2
    "noiri": "ノイリ",#追加2
    "nojiri": "ノジリ",#追加2
    "noma": "ノマ",#追加2
    "nomi": "ノミ",#追加2
    "nomoto": "ノモト",#追加2
    "nomura": "ノムラ",#追加2
    "nonaka": "ノナカ",#追加2
    "nonogi": "ノノギ",#追加2
    "norita": "ノリタ",#追加2
    "nosaka": "ノサカ",#追加2
    "notake": "ノタケ",#追加2
    "nozaki": "ノザキ",#追加2
    "nozato": "ノザト",#追加2
    "nozoe": "ノゾエ",#追加2
    "nuki": "ヌキ",#追加2
    "numahata": "ヌマハタ",#追加2
    "numajiri": "ヌマジリ",#追加2
    "numao": "ヌマオ",#追加2
    "numasawa": "ヌマサワ",#追加2
    "numata": "ヌマタ",#追加2
    "oba": "オオバ",#追加2
    "obara": "オバラ",#追加2
    "obata": "オバタ",#追加2
    "obayashi": "オオバヤシ",
    "obi": "オビ",#追加2
    "obunai": "オブナイ",#追加2
    "ochi": "オチ",#追加2
    "ochiai": "オチアイ",#追加2
    "ochiumi": "オチウミ",#追加2
    "oda": "オダ",#追加2
    "odagiri": "オダギリ",#追加2
    "odanaka": "オダナカ",#追加2
    "oe": "オエ",#追加2
    "ogaku": "オガク",#追加2
    "ogasawara": "オガサワラ",#追加2
    "ogata": "オガタ",#追加2
    "ogawa": "オガワ",#追加2
    "ogimoto": "オギモト",#追加2
    "ogita": "オギタ",#追加2
    "oguma": "オグマ",#追加2
    "ogura": "オグラ",#追加2
    "oguri": "オグリ",#追加2
    "ohama": "オオハマ",#追加2
    "ohara": "オオハラ", # 追加
    "ohashi": "オオハシ", # 追加
    "ohata": "オオハタ",
    "ohba": "オオバ", # 追加
    "ohbe": "オオベ",#追加2
    "ohe": "オオエ",#追加2
    "ohga": "オウガ",#追加2
    "ohguchi": "オオグチ", # 追加
    "ohguro": "オオグロ", # 追加
    "ohi": "オオイ",
    "ohira": "オオヒラ",#追加2
    "ohishi": "オオイシ",
    "ohkawa": "オオカワ", # 追加
    "ohki": "オオキ",#追加2
    "ohkita": "オオキタ", # 追加
    "ohkubo": "オオクボ",
    "ohkura": "オオクラ",#追加2
    "ohmori": "オオモリ",
    "ohnishi": "オオニシ",
    "ohno": "オオノ", # 追加
    "ohota": "オホタ",#追加2
    "ohsaka": "オオサカ", # 追加
    "ohsawa": "オオサワ", # 追加
    "ohshima": "オオシマ",
    "ohshiro": "オオシロ", # 追加
    "ohsuga": "オオスガ",#追加2
    "ohsugi": "オオスギ", # 追加
    "ohsumi": "オオスミ", # 追加
    "ohta": "オオタ", # 追加
    "ohtaka": "オウタカ",#追加2
    "ohtake": "オオタケ",#追加2
    "ohtani": "オオタニ",
    "ohtomo": "オオトモ",
    "ohuchi": "オオウチ", # 追加
    "ohwada": "オオワダ", # 追加
    "ohwaki": "オオワキ",#追加2
    "ohya": "オオヤ", # 追加
    "ohyama": "オオヤマ",#追加2
    "oikawa": "オイカワ",#追加2
    "oishi": "オオイシ", # 追加（大石系で安全）
    "oiwa": "オオイワ",
    "oizumi": "オオイズミ", # 追加（大泉系で安全）
    "oka": "オカ",#追加2
    "okabe": "オカベ",#追加2
    "okada": "オカダ",#追加2
    "okai": "オカイ",#追加2
    "okajima": "オカジマ",#追加2
    "okamatsu": "オカマツ",#追加2
    "okamoto": "オカモト",#追加2
    "okamura": "オカムラ",#追加2
    "okata": "オカタ",#追加2
    "okayama": "オカヤマ",#追加2
    "okazaki": "オカザキ",#追加2
    "okazawa": "オカザワ",#追加2
    "oketani": "オケタニ",#追加2
    "oki": "オオキ",#追加2
    "okina": "オキナ",#追加2
    "okino": "オキノ",#追加2
    "okita": "オキタ",#追加2
    "okonogi": "オコノギ",#追加2
    "okoshi": "オオコシ",#追加2
    "oku": "オク",#追加2
    "okubo": "オオクボ", # 追加（大久保）
    "okuda": "オクダ",#追加2
    "okui": "オクイ",#追加2
    "okumoto": "オクモト",#追加2
    "okumura": "オクムラ",#追加2
    "okuno": "オクノ",#追加2
    "okura": "オオクラ",
    "okutsu": "オオクツ",#追加2
    "okuyama": "オクヤマ",#追加2
    "omori": "オオモリ", # 追加（大森）
    "omote": "オモテ",#追加2
    "omura": "オオムラ", # 追加（大村）
    "omuro": "オムロ",#追加2
    "onish": "オオニシ",#追加2
    "onishi": "オオニシ", # 追加（大西）
    "ono": "オノ",
    "onodera": "オノデラ",#追加2
    "onoue": "オノウエ",#追加2
    "onozato": "オノザト",#追加2
    "onuma": "オヌマ",#追加2
    "ooba": "オオバ", # 追加（oo 綴り）
    "oobayashi": "オオバヤシ", # 追加
    "oochi": "オオチ", # 追加
    "oodate": "オオダテ", # 追加
    "oogaku": "オオガク",#追加2
    "ooguchi": "オオグチ", # 追加
    "ooguro": "オオグロ", # 追加
    "oohara": "オオハラ", # 追加
    "oohashi": "オオハシ", # 追加
    "ooi": "オオイ",
    "ookawa": "オオカワ", # 追加
    "ookita": "オオキタ", # 追加
    "ookubo": "オオクボ", # 追加
    "oomori": "オオモリ", # 追加
    "oomura": "オオムラ", # 追加
    "oonishi": "オオニシ", # 追加
    "oono": "オオノ", # 追加
    "oosaka": "オオサカ", # 追加
    "oosaki": "オオサキ",#追加2
    "oosawa": "オオサワ", # 追加
    "ooshima": "オオシマ", # 追加
    "oosugi": "オオスギ", # 追加
    "oouchi": "オオウチ", # 追加
    "oowada": "オオワダ", # 追加
    "ooya": "オオヤ", # 追加
    "oozato": "オオザト",#追加2
    "orihashi": "オリハシ",#追加2
    "osaka": "オオサカ",
    "osakada": "オサカダ",#追加2
    "osaki": "オサキ",#追加2
    "osawa": "オオサワ",#追加2
    "oshida": "オシダ",#追加2
    "oshima": "オオシマ",
    "oshino": "オシノ",#追加2
    "oshinomi": "オシノミ",#追加2
    "oshita": "オシタ",#追加2
    "oshitomi": "オシトミ",#追加2
    "osuga": "オスガ",#追加2
    "osumi": "オオスミ",
    "ota": "オオタ",
    "otaka": "オタカ",#追加2
    "otake": "オオタケ",
    "otaki": "オオタキ",#追加2
    "otani": "オオタニ",#追加2
    "otomo": "オオトモ",
    "otowa": "オトワ",#追加2
    "otsuba": "オオツバ", # 追加（大坪系：otsubo と迷う場合は下行を優先）
    "otsubo": "オオツボ", # 追加（大坪）
    "otsuji": "オオツジ",#追加2
    "otsuka": "オオツカ",
    "otsuki": "オオツキ",
    "ouchi": "オオウチ",#追加2
    "oura": "オオウラ",#追加2
    "owa": "オワ",#追加2
    "oyama": "オヤマ",
    "oyatani": "オヤタニ",#追加2
    "oyoshi": "オオヨシ",#追加2
    "ozaki": "オザキ",#追加2
    "ozasa": "オザサ",#追加2
    "ozawa": "オザワ",#追加2
    "ozono": "オオゾノ",#追加2
    "ozu": "オズ",#追加2
    "riku": "リク",#追加2
    "saburi": "サブリ",#追加2
    "sada": "サダ",#追加2
    "sadachi": "サダチ",#追加2
    "sadamatsu": "サダマツ",#追加2
    "saeki": "サエキ",#追加2
    "sago": "サゴ",#追加2
    "sahashi": "サハシ",#追加2
    "sai": "サイ",#追加2
    "saigan": "サイガン",#追加2
    "saigusa": "サイグサ",#追加2
    "saiin": "サイイン",#追加2
    "saiki": "サイキ",#追加2
    "saito": "サイトウ",
    "saitoh": "サイトウ",#追加2
    "saji": "サヂ",#追加2
    "saka": "サカ",#追加2
    "sakagami": "サカガミ",#追加2
    "sakaguchi": "サカグチ",#追加2
    "sakai": "サカイ",#追加2
    "sakaino": "サカイノ",#追加2
    "sakakura": "サカクラ",#追加2
    "sakamoto": "サカモト",#追加2
    "sakane": "サカネ",#追加2
    "sakata": "サカタ",#追加2
    "sakaue": "サカウエ",#追加2
    "sakio": "サキオ",#追加2
    "sakon": "サコン",#追加2
    "saku": "サク",#追加2
    "sakui": "サクイ",#追加2
    "sakuma": "サクマ",#追加2
    "sakurada": "サクラダ",#追加2
    "sakurai": "サクライ",#追加2
    "sambe": "サンべ",#追加2
    "sangen": "サンゲン",#追加2
    "sano": "サノ",#追加2
    "sanomura": "サノムラ",#追加2
    "sanui": "サヌイ",#追加2
    "saotome": "サオトメ",#追加2
    "sarai": "サライ",#追加2
    "sasabuchi": "ササブチ",#追加2
    "sasahira": "ササヒラ",#追加2
    "sasaki": "ササキ",#追加2
    "sasano": "ササノ",#追加2
    "sasaoka": "ササオカ",#追加2
    "sata": "サタ",#追加2
    "sato": "サトウ",
    "satogami": "サトガミ",#追加2
    "satoh": "サトウ",#追加2
    "satomi": "サトミ",#追加2
    "satow": "サトウ",#追加2
    "satsurai": "サツライ",#追加2
    "sawabata": "サワバタ",#追加2
    "sawada": "サワダ",#追加2
    "sawaguchi": "サワグチ",#追加2
    "sawano": "サワノ",#追加2
    "sawanobori": "サワノボリ",#追加2
    "sawatani": "サワタニ",#追加2
    "sawatari": "サワタリ",#追加2
    "sawayama": "サワヤマ",#追加2
    "sayama": "サヤマ",#追加2
    "segawa": "セガワ",#追加2
    "seguchi": "セグチ",#追加2
    "seike": "セイケ",#追加2
    "seiyama": "セイヤマ",#追加2
    "seki": "セキ",#追加2
    "sekido": "セキド",#追加2
    "sekiguchi": "セキグチ",#追加2
    "sekimoto": "セキモト",#追加2
    "seno": "セノ",#追加2
    "senoo": "セノオ",#追加2
    "seo": "セオ",#追加2
    "serikawa": "セリカワ",#追加2
    "setake": "セタケ",#追加2
    "seto": "セト",#追加2
    "setogawa": "セトガワ",#追加2
    "setoguchi": "セトグチ",#追加2
    "setoyama": "セトヤマ",#追加2
    "shiba": "シバ",#追加2
    "shibahashi": "シバハシ",#追加2
    "shibasaki": "シバサキ",#追加2
    "shibata": "シバタ",#追加2
    "shibui": "シブイ",#追加2
    "shibutani": "シブタニ",#追加2
    "shigematsu": "シゲマツ",#追加2
    "shigemoto": "シゲモト",#追加2
    "shigeta": "シゲタ",#追加2
    "shigetoshi": "シゲトシ",#追加2
    "shigihara": "シギハラ",#追加2
    "shiina": "シイナ",#追加2
    "shiiya": "シイヤ",#追加2
    "shikanai": "シカナイ",#追加2
    "shiko": "シコ",#追加2
    "shima": "シマ",#追加2
    "shimabukuro": "シマブクロ",#追加2
    "shimada": "シマダ",#追加2
    "shimahara": "シマハラ",#追加2
    "shimamoto": "シマモト",#追加2
    "shimamura": "シマムラ",#追加2
    "shimazaki": "シマザキ",#追加2
    "shimazu": "シマズ",#追加2
    "shimizu": "シミズ",#追加2
    "shimoda": "シモダ",#追加2
    "shimoji": "シモジ",#追加2
    "shimojo": "シモジョウ",#追加2
    "shimokawa": "シモカワ",#追加2
    "shimomura": "シモムラ",#追加2
    "shimonaga": "シモナガ",#追加2
    "shimono": "シモノ",#追加2
    "shimosato": "シモサト",#追加2
    "shimoyama": "シモヤマ",#追加2
    "shimozawa": "シモザワ",#追加2
    "shimura": "シムラ",#追加2
    "shinboku": "シンボク",#追加2
    "shindo": "シンドウ",
    "shinke": "シンケ",#追加2
    "shinmura": "シンムラ",#追加2
    "shinoda": "シノダ",#追加2
    "shinohara": "シノハラ",#追加2
    "shinozaki": "シノザキ",#追加2
    "shintaku": "シンタク",#追加2
    "shintani": "シンタニ",#追加2
    "shiode": "シオデ",#追加2
    "shiohira": "シオヒラ",#追加2
    "shioji": "シオジ",#追加2
    "shiojima": "シオジマ",#追加2
    "shiokawa": "シオカワ",#追加2
    "shiomi": "シオミ",#追加2
    "shiomura": "シオムラ",#追加2
    "shiono": "シオノ",#追加2
    "shiozaki": "シオザキ",#追加2
    "shiozawa": "シオザワ",#追加2
    "shirahama": "シラハマ",#追加2
    "shirai": "シライ",#追加2
    "shiraishi": "シライシ",#追加2
    "shirakabe": "シラカベ",#追加2
    "shiraki": "シラキ",#追加2
    "shirasaka": "シラサカ",#追加2
    "shirasaki": "シラサキ",#追加2
    "shiroshita": "シロシタ",#追加2
    "shirota": "シロタ",#追加2
    "shirotani": "シロタニ",#追加2
    "shiroto": "シロト",#追加2
    "shishido": "シシド",#追加2
    "shishikura": "シシクラ",#追加2
    "shitan": "シタン",#追加2
    "shitara": "シタラ",#追加2
    "shite": "シテ",#追加2
    "shizuta": "シズタ",#追加2
    "shoda": "ショウダ",#追加2
    "shoji": "ショウジ",#追加2
    "shojima": "ショウジマ",#追加2
    "shoumura": "ショウムラ",#追加2
    "shutta": "シュッタ",#追加2
    "sindo": "シンドウ",#追加2
    "sobue": "ソブエ",#追加2
    "soeda": "ソエダ",#追加2
    "soejima": "ソエジマ",#追加2
    "soga": "ソガ",#追加2
    "sogabe": "ソガベ",#追加2
    "sogo": "ソゴウ",#追加2
    "sohma": "ソウマ",#追加2
    "soma": "ソマ",#追加2
    "sone": "ソネ",#追加2
    "sonoda": "ソノダ",#追加2
    "sorimachi": "ソリマチ",#追加2
    "sotomi": "ソトミ",#追加2
    "suda": "スダ",#追加2
    "sudo": "スドウ",
    "suematsu": "スエマツ",#追加2
    "sueta": "スエタ",#追加2
    "suga": "スガ",#追加2
    "sugae": "スガエ",#追加2
    "sugane": "スガネ",#追加2
    "sugano": "スガノ",#追加2
    "sugawara": "スガワラ",#追加2
    "sugaya": "スガヤ",#追加2
    "sugie": "スギエ",#追加2
    "sugihara": "スギハラ",#追加2
    "sugimoto": "スギモト",#追加2
    "sugimura": "スギムラ",#追加2
    "sugino": "スギノ",#追加2
    "sugitani": "スギタニ",#追加2
    "sugiura": "スギウラ",#追加2
    "sugiyama": "スギヤマ",#追加2
    "sugizaki": "スギザキ",#追加2
    "suguro": "スグロ",#追加2
    "sukeda": "スケダ",#追加2
    "sukehiro": "スケヒロ",#追加2
    "sumida": "スミダ",#追加2
    "sumii": "スミイ",#追加2
    "sumimoto": "スミモト",#追加2
    "sumita": "スミタ",#追加2
    "sumitsuji": "スミツジ",#追加2
    "sumiyoshi": "スミヨシ",#追加2
    "suna": "スナ",#追加2
    "sunaga": "スナガ",#追加2
    "sunamura": "スナムラ",#追加2
    "sunohara": "スノハラ",#追加2
    "suwa": "スワ",#追加2
    "suzuki": "スズキ",#追加2
    "tabata": "タバタ",#追加2
    "tabita": "タビタ",#追加2
    "tabuchi": "タブチ",#追加2
    "tachibana": "タチバナ",#追加2
    "tachimori": "タチモリ",#追加2
    "tada": "タダ",#追加2
    "tadano": "タダノ",#追加2
    "tadokoro": "タドコロ",#追加2
    "tagawa": "タガワ",#追加2
    "taguchi": "タグチ",#追加2
    "taguri": "タグリ",#追加2
    "tahara": "タハラ",#追加2
    "taira": "タイラ",#追加2
    "tajima": "タジマ",#追加2
    "takada": "タカダ",#追加2
    "takae": "タカエ",#追加2
    "takafumi": "タカフミ",#追加2
    "takagaki": "タカガキ",#追加2
    "takagi": "タカギ",#追加2
    "takahama": "タカハマ",#追加2
    "takahara": "タカハラ",#追加2
    "takahashi": "タカハシ",#追加2
    "takahata": "タカハタ",#追加2
    "takaki": "タカキ",#追加2
    "takama": "タカマ",#追加2
    "takamatsu": "タカマツ",#追加2
    "takami": "タカミ",#追加2
    "takamisawa": "タカミサワ",#追加2
    "takamiya": "タカミヤ",#追加2
    "takamizawa": "タカミザワ",#追加2
    "takamura": "タカムラ",#追加2
    "takano": "タカノ",#追加2
    "takaoka": "タカオカ",#追加2
    "takasaki": "タカサキ",#追加2
    "takase": "タカセ",#追加2
    "takashima": "タカシマ",#追加2
    "takashio": "タカシオ",#追加2
    "takasu": "タカス",#追加2
    "takata": "タカタ",#追加2
    "takatsu": "タカツ",#追加2
    "takaya": "タカヤ",#追加2
    "takayama": "タカヤマ",#追加2
    "takebayashi": "タケバヤシ",#追加2
    "takeda": "タケダ",#追加2
    "takegami": "タケガミ",#追加2
    "takehara": "タケハラ",#追加2
    "takei": "タケイ",#追加2
    "takeishi": "タケイシ",#追加2
    "takeji": "タケジ",#追加2
    "takekawa": "タケカワ",#追加2
    "takemoto": "タケモト",#追加2
    "takemura": "タケムラ",#追加2
    "takenaka": "タケナカ",#追加2
    "takenouchi": "タケノウチ",#追加2
    "takenoya": "タケノヤ",#追加2
    "takeshita": "タケシタ",#追加2
    "takeuchi": "タケウチ",#追加2
    "takewa": "タケワ",#追加2
    "takeyama": "タケヤマ",#追加2
    "takigami": "タキガミ",#追加2
    "takiguchi": "タキグチ",#追加2
    "takii": "タキイ",#追加2
    "takita": "タキタ",#追加2
    "takou": "タコウ",#追加2
    "takuma": "タクマ",#追加2
    "takumi": "タクミ",#追加2
    "takura": "タクラ",#追加2
    "tama": "タマ",#追加2
    "tamada": "タマダ",#追加2
    "tamaki": "タマキ",#追加2
    "tamanaha": "タマナハ",#追加2
    "tambara": "タバラ",#追加2
    "taminishi": "タミニシ",#追加2
    "tamoto": "タモト",#追加2
    "tamura": "タムラ",#追加2
    "tanabe": "タナベ",#追加2
    "tanaka": "タナカ",#追加2
    "tang": "タン",#追加2
    "tani": "タニ",#追加2
    "tanichi": "タニチ",#追加2
    "tanigaki": "タニガキ",#追加2
    "tanigawa": "タニガワ",#追加2
    "taniguchi": "タニグチ",#追加2
    "tanimoto": "タニモト",#追加2
    "tanimura": "タニムラ",#追加2
    "taninobu": "タニノブ",#追加2
    "tanisawa": "タニサワ",#追加2
    "tanita": "タニタ",#追加2
    "taniuchi": "タニウチ",#追加2
    "taniwaki": "タニワキ",#追加2
    "tanizaki": "タニザキ",#追加2
    "tanouchi": "タノウチ",#追加2
    "tao": "タオ",#追加2
    "taomoto": "タオモト",#追加2
    "tara": "タラ",#追加2
    "taruya": "タルヤ",#追加2
    "tasaka": "タサカ",#追加2
    "tashima": "タシマ",#追加2
    "tashiro": "タシロ",#追加2
    "tatami": "タタミ",#追加2
    "tateishi": "タテイシ",#追加2
    "tateyama": "タテヤマ",#追加2
    "tatsugami": "タツガミ",#追加2
    "tatsumi": "タツミ",#追加2
    "tatsushima": "タツシマ",#追加2
    "tawara": "タワラ",#追加2
    "tawarahara": "タワラハラ",#追加2
    "tayama": "タヤマ",#追加2
    "tazaki": "タザキ",#追加2
    "tazawa": "タザワ",#追加2
    "terada": "テラダ",#追加2
    "terai": "テライ",#追加2
    "terajima": "テラジマ",#追加2
    "teramura": "テラムラ",#追加2
    "terao": "テラオ",#追加2
    "terasaka": "テラサカ",#追加2
    "terashita": "テラシタ",#追加2
    "terauchi": "テラウチ",#追加2
    "terui": "テルイ",#追加2
    "teshima": "テシマ",#追加2
    "tezuka": "テズカ",#追加2
    "toba": "トバ",#追加2
    "tobari": "トバリ",#追加2
    "tobaru": "トバル",#追加2
    "tobe": "トベ",#追加2
    "tobita": "トビタ",#追加2
    "tochiya": "トチヤ",#追加2
    "toda": "トダ",#追加2
    "todoroki": "トドロキ",#追加2
    "toguchi": "トグチ",#追加2
    "toh": "トウ",#追加2
    "tohara": "トハラ",#追加2
    "toida": "トイダ",#追加2
    "tojo": "トウジョウ",#追加2
    "tokai": "トウカイ",#追加2
    "tokano": "トカノ",#追加2
    "tokashiki": "トカシキ",#追加2
    "tokioka": "トキオカ",#追加2
    "tokita": "トキタ",#追加2
    "tokuda": "トクダ",#追加2
    "tokuhisa": "トクヒサ",#追加2
    "tokunaga": "トクナガ",#追加2
    "tokuno": "トクノ",#追加2
    "tokushige": "トクシゲ",#追加2
    "tokuyama": "トクヤマ",#追加2
    "toma": "トマ",#追加2
    "tomii": "トミイ",#追加2
    "tominaga": "トミナガ",#追加2
    "tomita": "トミタ",#追加2
    "tomoda": "トモダ",#追加2
    "tomotsuka": "トモツカ",#追加2
    "tomura": "トムラ",#追加2
    "tonomura": "トノムラ",#追加2
    "torii": "トリイ",#追加2
    "torikoshi": "トリコシ",#追加2
    "toriya": "トリヤ",#追加2
    "tosaka": "トサカ",#追加2
    "toshihiro": "トシヒロ",#追加2
    "toshiki": "トシキ",#追加2
    "toshima": "トシマ",#追加2
    "toubaru": "トウバル",#追加2
    "toya": "トヤ",#追加2
    "toyama": "トヤマ",#追加2
    "toyoda": "トヨダ",#追加2
    "toyofuku": "トヨフク",#追加2
    "toyoshima": "トヨシマ",#追加2
    "toyota": "トヨタ",#追加2
    "tsubakimoto": "ツバキモト",#追加2
    "tsubata": "ツバタ",#追加2
    "tsuboi": "ツボイ",#追加2
    "tsubono": "ツボノ",#追加2
    "tsuboyama": "ツボヤマ",#追加2
    "tsuchida": "ツチダ",#追加2
    "tsuchikane": "ツチカネ",#追加2
    "tsuchiya": "ツチヤ",#追加2
    "tsuchiyama": "ツチヤマ",#追加2
    "tsuda": "ツダ",#追加2
    "tsudome": "ツドメ",#追加2
    "tsugawa": "ツガワ",#追加2
    "tsugita": "ツギタ",#追加2
    "tsugu": "ツグ",#追加2
    "tsuji": "ツジ",#追加2
    "tsujihata": "ツジハタ",#追加2
    "tsujimoto": "ツジモト",#追加2
    "tsujimura": "ツジムラ",#追加2
    "tsujino": "ツジノ",#追加2
    "tsujita": "ツジタ",#追加2
    "tsukada": "ツカダ",#追加2
    "tsukahara": "ツカハラ",#追加2
    "tsukamoto": "ツカモト",#追加2
    "tsukinowa": "ツキノワ",#追加2
    "tsukiyama": "ツキヤマ",#追加2
    "tsukui": "ツクイ",#追加2
    "tsumaru": "ツマル",#追加2
    "tsunaki": "ツナキ",#追加2
    "tsunamoto": "ツナモト",#追加2
    "tsuneyoshi": "ツネヨシ",#追加2
    "tsunoda": "ツノダ",#追加2
    "tsuru": "ツル",#追加2
    "tsuruda": "ツルダ",#追加2
    "tsuruta": "ツルタ",#追加2
    "tsushima": "ツシマ",#追加2
    "tsutsui": "ツツイ",#追加2
    "tsutsumi": "ツツミ",#追加2
    "uchida": "ウチダ",#追加2
    "uchigata": "ウチガタ",#追加2
    "uchino": "ウチノ",#追加2
    "uchio": "ウチオ",#追加2
    "uchiyama": "ウチヤマ",#追加2
    "ueda": "ウエダ",#追加2
    "uegaito": "ウエガイト",#追加2
    "uehara": "ウエハラ",#追加2
    "ueki": "ウエキ",#追加2
    "uematsu": "ウエマツ",#追加2
    "uemura": "ウエムラ",#追加2
    "ueno": "ウエノ",#追加2
    "ueshima": "ウエシマ",#追加2
    "uesugi": "ウエスギ",#追加2
    "ueyama": "ウエヤマ",#追加2
    "ugawa": "ウガワ",#追加2
    "ukai": "ウカイ",#追加2
    "ukawa": "ウカワ",#追加2
    "ukita": "ウキタ",#追加2
    "umeda": "ウメダ",#追加2
    "umemoto": "ウメモト",#追加2
    "umemura": "ウメムラ",#追加2
    "umetani": "ウメタニ",#追加2
    "umino": "ウミノ",#追加2
    "une": "ウネ",#追加2
    "uno": "ウノ",#追加2
    "unoki": "ウノキ",#追加2
    "urabe": "ウラベ",#追加2
    "uranaka": "ウラナカ",#追加2
    "urata": "ウラタ",#追加2
    "urushida": "ウルシダ",#追加2
    "usami": "ウサミ",#追加2
    "ushijima": "ウシジマ",#追加2
    "usuda": "ウスダ",#追加2
    "usui": "ウスイ",#追加2
    "usuku": "ウスク",#追加2
    "usumoto": "ウスモト",#追加2
    "uwatoko": "ウワトコ",#追加2
    "uzu": "ウズ",#追加2
    "uzui": "ウズイ",#追加2
    "wada": "ワダ",#追加2
    "wakabayashi": "ワカバヤシ",#追加2
    "wakami": "ワカミ",#追加2
    "wakana": "ワカナ",#追加2
    "wakasa": "ワカサ",#追加2
    "wakatsuki": "ワカツキ",#追加2
    "wake": "ワケ",#追加2
    "waki": "ワキ",#追加2
    "wakita": "ワキタ",#追加2
    "wakugawa": "ワクガワ",#追加2
    "wanezaki": "ワネザキ",#追加2
    "warisawa": "ワリサワ",#追加2
    "waseda": "ワセダ",#追加2
    "washima": "ワシマ",#追加2
    "washimi": "ワシミ",#追加2
    "washiyama": "ワシヤマ",#追加2
    "watabe": "ワタベ",#追加2
    "watanabe": "ワタナベ",#追加2
    "watarai": "ワタライ",#追加2
    "yabe": "ヤベ",#追加2
    "yabumoto": "ヤブモト",#追加2
    "yabushita": "ヤブシタ",#追加2
    "yagasaki": "ヤガサキ",#追加2
    "yagi": "ヤギ",#追加2
    "yagihashi": "ヤギハシ",#追加2
    "yaginuma": "ヤギヌマ",#追加2
    "yaguchi": "ヤグチ",#追加2
    "yahagi": "ヤハギ",#追加2
    "yahata": "ヤハタ",#追加2
    "yahikozawa": "ヤヒコザワ",#追加2
    "yajima": "ヤジマ",#追加2
    "yaku": "ヤク",#追加2
    "yakushiji": "ヤクシジ",#追加2
    "yamabe": "ヤマベ",#追加2
    "yamada": "ヤマダ",#追加2
    "yamagami": "ヤマガミ",#追加2
    "yamaguchi": "ヤマグチ",#追加2
    "yamaji": "ヤマジ",#追加2
    "yamakage": "ヤマカゲ",#追加2
    "yamakami": "ヤマカミ",#追加2
    "yamakawa": "ヤマカワ",#追加2
    "yamaki": "ヤマキ",#追加2
    "yamamoto": "ヤマモト",#追加2
    "yamamura": "ヤマムラ",#追加2
    "yamana": "ヤマナ",#追加2
    "yamanaga": "ヤマナガ",#追加2
    "yamanaka": "ヤマナカ",#追加2
    "yamane": "ヤマネ",#追加2
    "yamano": "ヤマノ",#追加2
    "yamasaki": "ヤマサキ",#追加2
    "yamase": "ヤマセ",#追加2
    "yamashina": "ヤマシナ",#追加2
    "yamashiro": "ヤマシロ",#追加2
    "yamashita": "ヤマシタ",#追加2
    "yamato": "ヤマト",#追加2
    "yamauchi": "ヤマウチ",#追加2
    "yamawaki": "ヤマワキ",#追加2
    "yamaya": "ヤマヤ",#追加2
    "yamazaki": "ヤマザキ",#追加2
    "yanagawa": "ヤナガワ",#追加2
    "yanagi": "ヤナギ",#追加2
    "yanaka": "ヤナカ",#追加2
    "yanishi": "ヤニシ",#追加2
    "yano": "ヤノ",#追加2
    "yao": "ヤオ",#追加2
    "yara": "ヤラ",#追加2
    "yashige": "ヤシゲ",#追加2
    "yashima": "ヤシマ",#追加2
    "yasu": "ヤス",#追加2
    "yasuda": "ヤスダ",#追加2
    "yasuhara": "ヤスハラ",#追加2
    "yasui": "ヤスイ",#追加2
    "yasumoto": "ヤスモト",#追加2
    "yasumura": "ヤスムラ",#追加2
    "yasunaga": "ヤスナガ",#追加2
    "yatsu": "ヤツ",#追加2
    "yatsuda": "ヤツダ",#追加2
    "yatsuya": "ヤツヤ",#追加2
    "yoda": "ヨダ",#追加2
    "yokoi": "ヨコイ",#追加2
    "yokomatsu": "ヨコマツ",#追加2
    "yokomine": "ヨコミネ",#追加2
    "yokoo": "ヨコオ",#追加2
    "yokota": "ヨコタ",#追加2
    "yokoya": "ヨコヤ",#追加2
    "yokoyama": "ヨコヤマ",#追加2
    "yonamine": "ヨナミネ",#追加2
    "yoneda": "ヨネダ",#追加2
    "yoneoka": "ヨネオカ",#追加2
    "yonetsu": "ヨネツ",#追加2
    "yoneyama": "ヨネヤマ",#追加2
    "yonezawa": "ヨネザワ",#追加2
    "yonezu": "ヨネズ",#追加2
    "yoshida": "ヨシダ",#追加2
    "yoshihara": "ヨシハラ",#追加2
    "yoshihisa": "ヨシヒサ",#追加2
    "yoshijima": "ヨシジマ",#追加2
    "yoshikai": "ヨシカイ",#追加2
    "yoshikawa": "ヨシカワ",#追加2
    "yoshiki": "ヨシキ",#追加2
    "yoshimachi": "ヨシマチ",#追加2
    "yoshimitsu": "ヨシミツ",#追加2
    "yoshimoto": "ヨシモト",#追加2
    "yoshimura": "ヨシムラ",#追加2
    "yoshinaga": "ヨシナガ",#追加2
    "yoshino": "ヨシノ",#追加2
    "yoshioka": "ヨシオカ",#追加2
    "yoshitani": "ヨシタニ",#追加2
    "yoshitomi": "ヨシトミ",#追加2
    "yoshiura": "ヨシウラ",#追加2
    "yoshizaki": "ヨシザキ",#追加2
    "yoshizane": "ヨシザネ",#追加2
    "yoshizawa": "ヨシザワ",#追加2
    "yoshizumi": "ヨシズミ",#追加2
    "yuasa": "ユアサ",#追加2
    "yufu": "ユフ",#追加2
    "yuge": "ユゲ",#追加2
    "yui": "ユイ",#追加2
    "yuki": "ユキ",#追加2
    "yumoto": "ユモト",#追加2
    "yunoki": "ユノキ",#追加2
    "yuri": "ユリ",#追加2
    "yutani": "ユタニ",#追加2
    "yuzawa": "ユザワ",#追加2
    "zaima": "ザイマ",#追加2
    "zuguchi": "ズグチ",#追加2
}
# 個別上書き（名 / FirstName）：ローマ字 -> カタカナ
KATAKANA_FIRSTNAME_OVERRIDE = {
    "ai": "アイ", #追加2
    "aina": "アイナ", #追加2
    "akane": "アカネ", #追加2
    "akashi": "アカシ", #追加2
    "aki": "アキ", #追加2
    "akifumi": "アキフミ", #追加2
    "akihiko": "アキヒコ", #追加2
    "akihiro": "アキヒロ", #追加2
    "akihito": "アキヒト", #追加2
    "akiko": "アキコ", #追加2
    "akimasa": "アキマサ", #追加2
    "akinori": "アキノリ", #追加2
    "akio": "アキオ", #追加2
    "akiomi": "アキオミ", #追加2
    "akira": "アキラ", #追加2
    "akiteru": "アキテル", #追加2
    "akito": "アキト", #追加2
    "akitoshi": "アキトシ", #追加2
    "akiyoshi": "アキヨシ", #追加2
    "amane": "アマネ", #追加2
    "ami": "アミ", #追加2
    "anna": "アンナ", #追加2
    "arano": "アラノ", #追加2
    "arata": "アラタ", #追加2
    "arihiro": "アリヒロ", #追加2
    "aritaka": "アリタカ", #追加2
    "aritomo": "アリトモ", #追加2
    "asahiro": "アサヒロ", #追加2
    "asami": "アサミ", #追加2
    "asataro": "アサタロウ",
    "asataro": "アサタロウ", #追加2
    "atomu": "アトム", #追加2
    "atsuhiko": "アツヒコ", #追加2
    "atsuhiro": "アツヒロ", #追加2
    "atsuki": "アツキ", #追加2
    "atsuko": "アツコ", #追加2
    "atsumi": "アツミ", #追加2
    "atsunori": "アツノリ", #追加2
    "atsuo": "アツオ", #追加2
    "atsushi": "アツシ", #追加2
    "atsutaka": "アツタカ", #追加2
    "atsuya": "アツヤ", #追加2
    "atsuyoshi": "アツヨシ", #追加2
    "atsuyuki": "アツユキ", #追加2
    "aya": "アヤ", #追加2
    "ayaka": "アヤカ", #追加2
    "ayako": "アヤコ", #追加2
    "ayane": "アヤネ", #追加2
    "ayano": "アヤノ", #追加2
    "ayumi": "アユミ", #追加2
    "ayumu": "アユム", #追加2
    "azusa": "アズサ", #追加2
    "chiaki": "チアキ", #追加2
    "chie": "チエ", #追加2
    "chietsugu": "チエツグ", #追加2
    "chiharu": "チハル", #追加2
    "chika": "チカ", #追加2
    "chikai": "チカイ", #追加2
    "chikako": "チカコ", #追加2
    "chikao": "チカオ", #追加2
    "chikara": "チカラ", #追加2
    "chisato": "チサト", #追加2
    "dai": "ダイ", #追加2
    "daichi": "ダイチ", #追加2
    "daigo": "ダイゴ", #追加2
    "daijiro": "ダイジロウ", #追加2
    "daiju": "ダイジュ", #追加2
    "daiki": "ダイキ", #追加2
    "daisaku": "ダイサク", #追加2
    "daishi": "ダイシ", #追加2
    "daisuke": "ダイスケ", #追加2
    "daitaro": "ダイタロウ", #追加2
    "eigo": "エイゴ", #追加2
    "eiichi": "エイイチ", #追加2
    "eiichiro": "エイイチロウ",
    "eiji": "エイジ", #追加2
    "eiki": "エイキ", #追加2
    "eiko": "エイコ", #追加2
    "eiryu": "エイリュウ", #追加2
    "eisaku": "エイサク", #追加2
    "eisei": "エイセイ", #追加2
    "eisuke": "エイスケ", #追加2
    "eizo": "エイゾウ",
    "emi": "エミ", #追加2
    "eri": "エリ", #追加2
    "erika": "エリカ", #追加2
    "etsu": "エツ", #追加2
    "etsuji": "エツジ", #追加2
    "etsuko": "エツコ", #追加2
    "etsumi": "エツミ", #追加2
    "etsuo": "エツオ", #追加2
    "fujio": "フジオ", #追加2
    "fumiaki": "フミアキ", #追加2
    "fumie": "フミエ", #追加2
    "fumiharu": "フミハル", #追加2
    "fumika": "フミカ", #追加2
    "fuminari": "フミナリ", #追加2
    "fuminobu": "フミノブ", #追加2
    "fumitaka": "フミタカ", #追加2
    "fumitoshi": "フミトシ", #追加2
    "fumiya": "フミヤ", #追加2
    "fumiyasu": "フミヤス", #追加2
    "fumiyuki": "フミユキ", #追加2
    "futoshi": "フトシ", #追加2
    "gaku": "ガク", #追加2
    "genki": "ゲンキ", #追加2
    "genryu": "ゲンリュウ", #追加2
    "giichi": "ギイチ", #追加2
    "go": "ゴウ",
    "goro": "ゴロウ",
    "hajime": "ハジメ", #追加2
    "haruhiko": "ハルヒコ", #追加2
    "haruhito": "ハルヒト", #追加2
    "haruka": "ハルカ", #追加2
    "harukazu": "ハルカズ", #追加2
    "haruki": "ハルキ", #追加2
    "harumi": "ハルミ", #追加2
    "haruo": "ハルオ", #追加2
    "harutoshi": "ハルトシ", #追加2
    "haruya": "ハルヤ", #追加2
    "haruyuki": "ハルユキ", #追加2
    "hayashi": "ハヤシ", #追加2
    "hayato": "ハヤト", #追加2
    "heima": "ヘイマ", #追加2
    "heitaro": "ヘイタロウ", #追加2
    "hideaki": "ヒデアキ", #追加2
    "hidefumi": "ヒデフミ", #追加2
    "hideharu": "ヒデハル", #追加2
    "hidehiko": "ヒデヒコ", #追加2
    "hideichi": "ヒデイチ", #追加2
    "hidekazu": "ヒデカズ", #追加2
    "hideki": "ヒデキ", #追加2
    "hidekuni": "ヒデクニ", #追加2
    "hidemaro": "ヒデマロ", #追加2
    "hidemasa": "ヒデマサ", #追加2
    "hidemori": "ヒデモリ", #追加2
    "hidenari": "ヒデナリ", #追加2
    "hidenobu": "ヒデノブ", #追加2
    "hidenori": "ヒデノリ", #追加2
    "hideo": "ヒデオ", #追加2
    "hidesato": "ヒデサト", #追加2
    "hideshi": "ヒデシ", #追加2
    "hidetaka": "ヒデタカ", #追加2
    "hideto": "ヒデト", #追加2
    "hidetomo": "ヒデトモ", #追加2
    "hidetoshi": "ヒデトシ", #追加2
    "hidetsugu": "ヒデツグ", #追加2
    "hideya": "ヒデヤ", #追加2
    "hideyuki": "ヒデユキ", #追加2
    "hikaru": "ヒカル", #追加2
    "himika": "ヒミカ", #追加2
    "hiraku": "ヒラク", #追加2
    "hiroaki": "ヒロアキ", #追加2
    "hirofumi": "ヒロフミ", #追加2
    "hiroharu": "ヒロハル", #追加2
    "hirohide": "ヒロヒデ", #追加2
    "hirohiko": "ヒロヒコ", #追加2
    "hirohisa": "ヒロヒサ", #追加2
    "hirohito": "ヒロヒト", #追加2
    "hirokazu": "ヒロカズ", #追加2
    "hiroki": "ヒロキ", #追加2
    "hirokuni": "ヒロクニ", #追加2
    "hiromasa": "ヒロマサ", #追加2
    "hiromi": "ヒロミ", #追加2
    "hiromichi": "ヒロミチ", #追加2
    "hiromitsu": "ヒロミツ", #追加2
    "hiromoto": "ヒロモト", #追加2
    "hiromu": "ヒロム", #追加2
    "hironaga": "ヒロナガ", #追加2
    "hironobu": "ヒロノブ", #追加2
    "hironori": "ヒロノリ", #追加2
    "hirooki": "ヒロオキ", #追加2
    "hirosada": "ヒロサダ", #追加2
    "hirosato": "ヒロサト", #追加2
    "hiroshi": "ヒロシ", #追加2
    "hirosuke": "ヒロスケ", #追加2
    "hirota": "ヒロタ", #追加2
    "hirotaka": "ヒロタカ", #追加2
    "hirotake": "ヒロタケ", #追加2
    "hiroto": "ヒロト", #追加2
    "hirotoshi": "ヒロトシ", #追加2
    "hirotsugu": "ヒロツグ", #追加2
    "hiroya": "ヒロヤ", #追加2
    "hiroyasu": "ヒロヤス", #追加2
    "hiroyo": "ヒロヨ", #追加2
    "hiroyoshi": "ヒロヨシ", #追加2
    "hiroyuki": "ヒロユキ", #追加2
    "hisaaki": "ヒサアキ", #追加2
    "hisahiko": "ヒサヒコ", #追加2
    "hisahito": "ヒサヒト", #追加2
    "hisaki": "ヒサキ", #追加2
    "hisako": "ヒサコ", #追加2
    "hisanori": "ヒサノリ", #追加2
    "hisao": "ヒサオ", #追加2
    "hisashi": "ヒサシ", #追加2
    "hisateru": "ヒサテル", #追加2
    "hisato": "ヒサト", #追加2
    "hisatomi": "ヒサトミ", #追加2
    "hisayuki": "ヒサユキ", #追加2
    "hitomi": "ヒトミ", #追加2
    "hitoshi": "ヒトシ", #追加2
    "hoshito": "ホシト", #追加2
    "hyuma": "ヒュマ", #追加2
    "ichiro": "イチロウ", #追加2
    "ichitaro": "イチタロウ",
    "iichiro": "イイチロウ", #追加2
    "ikuko": "イクコ", #追加2
    "ikumi": "イクミ", #追加2
    "ikuo": "イクオ", #追加2
    "ippei": "イッペイ", #追加2
    "isamu": "イサム", #追加2
    "isao": "イサオ", #追加2
    "issei": "イッセイ",
    "itaru": "イタル", #追加2
    "itsuro": "イツロウ",
    "itta": "イッタ", #追加2
    "iwao": "イワオ", #追加2
    "jin": "ジン", #追加2
    "jiro": "ジロウ",
    "jo": "ジョウ",
    "joh": "ジョウ", #追加2
    "joji": "ジョウジ", # ← 追加
    "jota": "ジョウタ", #追加2
    "jumpei": "ジュンペイ", #追加2
    "jun": "ジュン",
    "jun-ei": "ジュンエイ", #追加2
    "junichi": "ジュンイチ",
    "jun-ichi": "ジュンイチ",
    "junichiro": "ジュンイチロウ", #追加2
    "junji": "ジュンジ", #追加2
    "junjiro": "ジュンジロウ", #追加2
    "junki": "ジュンキ", #追加2
    "junko": "ジュンコ",
    "junya": "ジュンヤ",
    "juri": "ジュリ", #追加2
    "jyunki": "ユンキ", #追加2
    "kaho": "カホ", #追加2
    "kahori": "カホリ", #追加2
    "kai": "カイ", #追加2
    "kaihara": "カイハラ", #追加2
    "kaito": "カイト", #追加2
    "kakuya": "カクヤ", #追加2
    "kan": "カン", #追加2
    "kana": "カナ", #追加2
    "kanae": "カナエ", #追加2
    "kanako": "カナコ", #追加2
    "kaneto": "カネト", #追加2
    "kanichi": "カンイチ", #追加2
    "kanta": "カンタ", #追加2
    "kantaro": "カンタロウ", #追加2
    "kaori": "カオリ", #追加2
    "kaoru": "カオル", #追加2
    "kasumi": "カスミ", #追加2
    "katsuaki": "カツアキ", #追加2
    "katsuhiko": "カツヒコ", #追加2
    "katsuhiro": "カツヒロ", #追加2
    "katsuhisa": "カツヒサ", #追加2
    "katsuhito": "カツヒト", #追加2
    "katsuji": "カツジ", #追加2
    "katsuki": "カツキ", #追加2
    "katsuko": "カツコ", #追加2
    "katsumi": "カツミ", #追加2
    "katsunori": "カツノリ", #追加2
    "katsuomi": "カツオミ", #追加2
    "katsuro": "カツロウ", #追加2
    "katsushi": "カツシ", #追加2
    "katsutaka": "カツタカ", #追加2
    "katsutoshi": "カツトシ", #追加2
    "katsuya": "カツヤ", #追加2
    "katsuyoshi": "カツヨシ", #追加2
    "katsuyuki": "カツユキ", #追加2
    "kawai": "カワイ", #追加2
    "kayo": "カヨ", #追加2
    "kazuaki": "カズアキ", #追加2
    "kazufumi": "カズフミ", #追加2
    "kazuhide": "カズヒデ", #追加2
    "kazuhiko": "カズヒコ", #追加2
    "kazuhiro": "カズヒロ", #追加2
    "kazuhisa": "カズヒサ", #追加2
    "kazuho": "カズホ", #追加2
    "kazuki": "カズキ", #追加2
    "kazuma": "カズマ", #追加2
    "kazumasa": "カズマサ", #追加2
    "kazumaza": "カズマザ", #追加2
    "kazumi": "カズミ", #追加2
    "kazumiki": "カズミキ", #追加2
    "kazunari": "カズナリ", #追加2
    "kazunori": "カズノリ", #追加2
    "kazuo": "カズオ", #追加2
    "kazuoki": "カズオキ", #追加2
    "kazuomi": "カズオミ", #追加2
    "kazushi": "カズシ", #追加2
    "kazushige": "カズシゲ", #追加2
    "kazutaka": "カズタカ", #追加2
    "kazuteru": "カズテル", #追加2
    "kazuto": "カズト", #追加2
    "kazutoshi": "カズトシ", #追加2
    "kazuya": "カズヤ", #追加2
    "kazuyasu": "カズヤス", #追加2
    "kazuyoshi": "カズヨシ", #追加2
    "kazuyuki": "カズユキ", #追加2
    "kei": "ケイ", #追加2
    "keigo": "ケイゴ", #追加2
    "keiichi": "ケイイチ", #追加2
    "keiichiro": "ケイイチロウ", #追加2
    "keiji": "ケイジ", #追加2
    "keijiro": "ケイジロウ", #追加2
    "keiki": "ケイキ", #追加2
    "keiko": "ケイコ", #追加2
    "keishi": "ケイシ", #追加2
    "keisuke": "ケイスケ", #追加2
    "keita": "ケイタ", #追加2
    "keizo": "ケイゾ", #追加2
    "ken": "ケン", #追加2
    "kengo": "ケンゴ", #追加2
    "kenichi": "ケンイチ",
    "ken-ichi": "ケンイチ",
    "kenichiro": "ケンイチロウ",
    "ken-ichiro": "ケンイチロウ", #追加2
    "kenji": "ケンジ", #追加2
    "kensaku": "ケンサク", #追加2
    "kenshi": "ケンシ", #追加2
    "kensho": "ケンショウ", #追加2
    "kensuke": "ケンスケ", #追加2
    "kenta": "ケンタ", #追加2
    "kentaro": "ケンタロウ",
    "kento": "ケント", #追加2
    "kenya": "ケンヤ",
    "kenzo": "ケンゾウ", #追加2
    "kikuo": "キクオ", #追加2
    "kimiaki": "キミアキ", #追加2
    "kiminori": "キミノリ", #追加2
    "kimito": "キミト", #追加2
    "kimitoshi": "キミトシ", #追加2
    "kinya": "キンヤ", #追加2
    "kinzo": "キンゾウ", #追加2
    "kisaki": "キサキ", #追加2
    "kiu": "キウ", #追加2
    "kiwamu": "キワム", #追加2
    "kiyohide": "キヨヒデ", #追加2
    "kiyohisa": "キヨヒサ", #追加2
    "kiyokazu": "キヨカズ", #追加2
    "kiyoko": "キヨコ", #追加2
    "kiyomitsu": "キヨミツ", #追加2
    "kiyoshi": "キヨシ", #追加2
    "kiyota": "キヨタ", #追加2
    "kiyotaka": "キヨタカ", #追加2
    "kiyotomi": "キヨトミ", #追加2
    "kiyotoshi": "キヨトシ", #追加2
    "ko": "コウ",
    "kodai": "コウダイ", #追加2
    "koh": "コウ",
    "kohei": "コウヘイ",
    "kohki": "コウキ", #追加2
    "koichi": "コウイチ",
    "koichiro": "コウイチロウ",
    "koji": "コウジ",
    "koki": "コウキ",
    "konosuke": "コウノスケ", #追加2
    "korehito": "コレヒト", #追加2
    "kosei": "コウセイ", # ← 追加
    "koshi": "コウシ",
    "koshiro": "コウシロウ",
    "kosuke": "コウスケ",
    "kota": "コウタ", # ← 追加
    "kotaro": "コウタロウ",
    "koto": "コト", #追加2
    "kou": "コウ", #追加2
    "koudai": "コウダイ", #追加2
    "kouhei": "コウヘイ", #追加2
    "kouichi": "コウイチ", #追加2
    "kouji": "コウジ", #追加2
    "kouki": "コウキ", #追加2
    "kousuke": "コウスケ", #追加2
    "kouya": "コウヤ", #追加2
    "koya": "コウヤ", #追加2
    "koyama": "コヤマ", #追加2
    "kozo": "コウゾウ", #追加2
    "kumiko": "クミコ", #追加2
    "kuniaki": "クニアキ", #追加2
    "kunihiko": "クニヒコ", #追加2
    "kunihiro": "クニヒロ", #追加2
    "kunimitsu": "クニミツ", #追加2
    "kuninobu": "クニノブ", #追加2
    "kunio": "クニオ", #追加2
    "kuniya": "クニヤ", #追加2
    "kuniyasu": "クニヤス", #追加2
    "kuniyoshi": "クニヨシ", #追加2
    "kuniyuki": "クニユキ", #追加2
    "kurara": "クララ", #追加2
    "kuya": "クウヤ", #追加2
    "kyo": "キョウ", #追加2
    "kyohei": "キョウヘイ",
    "kyoichi": "キョウイチ", #追加2
    "kyoji": "キョウジ", #追加2
    "kyoko": "キョウコ",
    "machiko": "マチコ", #追加2
    "madoka": "マドカ", #追加2
    "mafumi": "マフミ", #追加2
    "mai": "マイ", #追加2
    "maiko": "マイコ", #追加2
    "maki": "マキ", #追加2
    "makiko": "マキコ", #追加2
    "makio": "マキオ", #追加2
    "makishi": "マキシ", #追加2
    "makoto": "マコト", #追加2
    "mamoru": "マモル", #追加2
    "mana": "マナ", #追加2
    "manabu": "マナブ", #追加2
    "manami": "マナミ", #追加2
    "mao": "マオ", #追加2
    "maoto": "マオト", #追加2
    "mareka": "マレカ", #追加2
    "marenao": "マレナオ", #追加2
    "mari": "マリ", #追加2
    "mariko": "マリコ", #追加2
    "marina": "マリナ", #追加2
    "marohito": "マロヒト", #追加2
    "masaaki": "マサアキ", #追加2
    "masafumi": "マサフミ", #追加2
    "masaharu": "マサハル", #追加2
    "masahide": "マサヒデ", #追加2
    "masahiko": "マサヒコ", #追加2
    "masahiro": "マサヒロ", #追加2
    "masahisa": "マサヒサ", #追加2
    "masahito": "マサヒト", #追加2
    "masakatsu": "マサカツ", #追加2
    "masakazu": "マサカズ", #追加2
    "masaki": "マサキ", #追加2
    "masakiyo": "マサキヨ", #追加2
    "masako": "マサコ", #追加2
    "masami": "マサミ", #追加2
    "masamichi": "マサミチ", #追加2
    "masamitsu": "マサミツ", #追加2
    "masanaga": "マサナガ", #追加2
    "masanao": "マサナオ", #追加2
    "masanari": "マサナリ", #追加2
    "masanobu": "マサノブ", #追加2
    "masanori": "マサノリ", #追加2
    "masao": "マサオ", #追加2
    "masaoki": "マサオキ", #追加2
    "masaomi": "マサオミ", #追加2
    "masaru": "マサル", #追加2
    "masashi": "マサシ", #追加2
    "masashiro": "マサシロ", #追加2
    "masataka": "マサタカ", #追加2
    "masatake": "マサタケ", #追加2
    "masateru": "マサテル", #追加2
    "masato": "マサト", #追加2
    "masatoshi": "マサトシ", #追加2
    "masatsugu": "マサツグ", #追加2
    "masaya": "マサヤ", #追加2
    "masayasu": "マサヤス", #追加2
    "masayoshi": "マサヨシ", #追加2
    "masayuki": "マサユキ", #追加2
    "mashio": "マシオ", #追加2
    "maya": "マヤ", #追加2
    "mayu": "マユ", #追加2
    "mayuko": "マユコ", #追加2
    "megumi": "メグミ", #追加2
    "mei": "メイ", #追加2
    "michiaki": "ミチアキ", #追加2
    "michifumi": "ミチフミ", #追加2
    "michihiko": "ミチヒコ", #追加2
    "michihiro": "ミチヒロ", #追加2
    "michihito": "ミチヒト", #追加2
    "michika": "ミチカ", #追加2
    "michikazu": "ミチカズ", #追加2
    "michiko": "ミチコ", #追加2
    "michinari": "ミチナリ", #追加2
    "michinobu": "ミチノブ", #追加2
    "michio": "ミチオ", #追加2
    "michiro": "ミチロウ", #追加2
    "michiya": "ミチヤ", #追加2
    "michiyo": "ミチヨ", #追加2
    "migaku": "ミガク", #追加2
    "miho": "ミホ", #追加2
    "mikako": "ミカコ", #追加2
    "mike": "マイク", #追加2
    "miki": "ミキ", #追加2
    "mikihiro": "ミキヒロ", #追加2
    "mikihito": "ミキヒト", #追加2
    "mikiko": "ミキコ", #追加2
    "mikio": "ミキオ", #追加2
    "mikizo": "ミキゾウ", #追加2
    "minako": "ミナコ", #追加2
    "minami": "ミナミ", #追加2
    "minao": "ミナオ", #追加2
    "minori": "ミノリ", #追加2
    "minoru": "ミノル", #追加2
    "mio": "ミオ", #追加2
    "mirei": "ミレイ", #追加2
    "miri": "ミリ", #追加2
    "misa": "ミサ", #追加2
    "misaki": "ミサキ", #追加2
    "misato": "ミサト", #追加2
    "mistuhiko": "ミトゥヒコ", #追加2
    "mitsuaki": "ミツアキ", #追加2
    "mitsuhiko": "ミツヒコ", #追加2
    "mitsuhiro": "ミツヒロ", #追加2
    "mitsukuni": "ミツクニ", #追加2
    "mitsumasa": "ミツマサ", #追加2
    "mitsunori": "ミツノリ", #追加2
    "mitsuo": "ミツオ", #追加2
    "mitsuru": "ミツル", #追加2
    "mitsutaka": "ミツタカ", #追加2
    "mitsutoshi": "ミツトシ", #追加2
    "mitsuya": "ミツヤ", #追加2
    "mitsuyoshi": "ミツヨシ", #追加2
    "miwa": "ミワ", #追加2
    "mizuhiko": "ミズヒコ", #追加2
    "mizuho": "ミズホ", #追加2
    "mizuki": "ミズキ", #追加2
    "moeko": "モエコ", #追加2
    "momo": "モモ", #追加2
    "moriaki": "モリアキ", #追加2
    "morihiko": "モリヒコ", #追加2
    "morimasa": "モリマサ", #追加2
    "morio": "モリオ", #追加2
    "motoaki": "モトアキ", #追加2
    "motoharu": "モトハル", #追加2
    "motohiro": "モトヒロ", #追加2
    "motoki": "モトキ", #追加2
    "motomu": "モトム", #追加2
    "motoshi": "モトシ", #追加2
    "motosu": "モトス", #追加2
    "mototsugu": "モトツグ", #追加2
    "motoyoshi": "モトヨシ", #追加2
    "munehiro": "ムネヒロ", #追加2
    "munenori": "ムネノリ", #追加2
    "munetaka": "ムネタカ", #追加2
    "mustumi": "ムツミ", #追加2
    "mutsumi": "ムツミ", #追加2
    "mutsuo": "ムツオ", #追加2
    "myong": "ミョン", #追加2
    "nagataka": "ナガタカ", #追加2
    "nagi": "ナギ", #追加2
    "namio": "ナミオ", #追加2
    "nana": "ナナ", #追加2
    "nao": "ナオ", #追加2
    "naoaki": "ナオアキ", #追加2
    "naoei": "ナオエイ", #追加2
    "naofumi": "ナオフミ", #追加2
    "naohiko": "ナオヒコ", #追加2
    "naohiro": "ナオヒロ", #追加2
    "naohisa": "ナオヒサ", #追加2
    "naohito": "ナオヒト", #追加2
    "naoki": "ナオキ", #追加2
    "naoko": "ナオコ", #追加2
    "naomi": "ナオミ", #追加2
    "naomichi": "ナオミチ", #追加2
    "naonori": "ナオノリ", #追加2
    "naotaka": "ナオタカ", #追加2
    "naotake": "ナオタケ", #追加2
    "naoto": "ナオト", #追加2
    "naotoshi": "ナオトシ", #追加2
    "naoya": "ナオヤ", #追加2
    "naoyuki": "ナオユキ", #追加2
    "naritsugu": "ナリツグ", #追加2
    "naruhiko": "ナルヒコ", #追加2
    "narumi": "ナルミ", #追加2
    "natsuhiko": "ナツヒコ", #追加2
    "natsuhiro": "ナツヒロ", #追加2
    "natsuko": "ナツコ", #追加2
    "natsumi": "ナツミ", #追加2
    "natsuya": "ナツヤ", #追加2
    "nehiro": "ネヒロ", #追加2
    "neiko": "ネイコ", #追加2
    "nobu": "ノブ", #追加2
    "nobuaki": "ノブアキ", #追加2
    "nobuhide": "ノブヒデ", #追加2
    "nobuhiko": "ノブヒコ", #追加2
    "nobuhiro": "ノブヒロ", #追加2
    "nobuhisa": "ノブヒサ", #追加2
    "nobuhito": "ノブヒト", #追加2
    "nobuki": "ノブキ", #追加2
    "nobunari": "ノブナリ", #追加2
    "nobuo": "ノブオ", #追加2
    "nobushige": "ノブシゲ", #追加2
    "nobutaka": "ノブタカ", #追加2
    "nobuyasu": "ノブヤス", #追加2
    "nobuyuki": "ノブユキ", #追加2
    "noriaki": "ノリアキ", #追加2
    "norifumi": "ノリフミ", #追加2
    "norihiko": "ノリヒコ", #追加2
    "norihiro": "ノリヒロ", #追加2
    "norihisa": "ノリヒサ", #追加2
    "norihito": "ノリヒト", #追加2
    "norikazu": "ノリカズ", #追加2
    "noriko": "ノリコ", #追加2
    "norimasa": "ノリマサ", #追加2
    "norimichi": "ノリミチ", #追加2
    "norio": "ノリオ", #追加2
    "noritaka": "ノリタカ", #追加2
    "noritomo": "ノリトモ", #追加2
    "noritoshi": "ノリトシ", #追加2
    "noriya": "ノリヤ", #追加2
    "noriyasu": "ノリヤス", #追加2
    "noriyoshi": "ノリヨシ", #追加2
    "noriyuki": "ノリユキ", #追加2
    "nozomi": "ノゾミ", #追加2
    "nozomu": "ノゾム", #追加2
    "onichi": "オンイチ", #追加2
    "osamu": "オサム", #追加2
    "otohime": "オトヒメ", #追加2
    "raisuke": "ライスケ", #追加2
    "raita": "ライタ", #追加2
    "ran": "ラン", #追加2
    "reiji": "レイジ", #追加2
    "reiko": "レイコ", #追加2
    "reina": "レイナ", #追加2
    "ren": "レン", #追加2
    "rensuke": "レンスケ",
    "reo": "レオ", #追加2
    "rie": "リエ", #追加2
    "rihito": "リヒト", #追加2
    "rika": "リカ", #追加2
    "riku": "リク", #追加2
    "rikuo": "リクオ", #追加2
    "rikuta": "リクタ", #追加2
    "rikuya": "リクヤ", #追加2
    "rin": "リン", #追加2
    "rine": "リネ", #追加2
    "rintaro": "リンタロウ", #追加2
    "rishi": "リシ", #追加2
    "rissei": "リッセイ", #追加2
    "ruiko": "ルイコ", #追加2
    "ruka": "ルカ", #追加2
    "ryo": "リョウ", # ← 追加（Ryo系を辞書で固定したい場合）
    "ryohei": "リョウヘイ", # ← 追加
    "ryoichi": "リョウイチ", # ← 追加
    "ryoji": "リョウジ", # ← 追加
    "ryoko": "リョウコ",
    "ryoma": "リョウマ", # ← 追加
    "ryosuke": "リョウスケ", # ← 追加（大文字→小文字に統一）
    "ryota": "リョウタ", #追加2
    "ryotaro": "リョウタロウ", #追加2
    "ryou": "リョウ", #追加2
    "ryouhei": "リョウヘイ", #追加2
    "ryousuke": "リョウスケ", #追加2
    "ryozo": "リョウゾウ",
    "ryu": "リュウ", #追加2
    "ryuhei": "リュヘイ", #追加2
    "ryuichi": "リュウイチ", # ← 追加（大文字→小文字に統一）
    "ryuichiro": "リュウイチロウ", #追加2
    "ryu-ichiro": "リュウイチロウ", #追加2
    "ryuji": "リュウジ", #追加2
    "ryuki": "リュウキ",
    "ryusuke": "リュウスケ",
    "ryutaro": "リュウタロウ", #追加2
    "ryuusuke": "リュウスケ", #追加2
    "ryuzaburo": "リュウザブロウ", #追加2
    "ryuzo": "リュウゾウ", #追加2
    "saburo": "サブロウ", #追加2
    "sachiko": "サチコ", #追加2
    "sachiro": "サチロ", #追加2
    "sadaharu": "サダハル", #追加2
    "sadako": "サダコ", #追加2
    "sadanori": "サダノリ", #追加2
    "saeko": "サエコ", #追加2
    "saiko": "サイコ", #追加2
    "sakae": "サカエ", #追加2
    "saki": "サキ", #追加2
    "sakiko": "サキコ", #追加2
    "sakura": "サクラ", #追加2
    "sakuramaru": "サクラマル", #追加2
    "sanae": "サナエ", #追加2
    "saori": "サオリ", #追加2
    "sara": "サラ", #追加2
    "satoaki": "サトアキ", #追加2
    "satoki": "サトキ", #追加2
    "satoko": "サトコ", #追加2
    "satomi": "サトミ", #追加2
    "satori": "サトリ", #追加2
    "satoru": "サトル", #追加2
    "satoshi": "サトシ", #追加2
    "satsuki": "サツキ", #追加2
    "saya": "サヤ", #追加2
    "sayaka": "サヤカ", #追加2
    "sayano": "サヤノ", #追加2
    "sayumi": "サユミ", #追加2
    "sei": "セイ", #追加2
    "seigo": "セイゴ", #追加2
    "seiichi": "セイイチ", #追加2
    "seiichiro": "セイイチロウ", #追加2
    "seiji": "セイジ", #追加2
    "seijiro": "セイジロウ", #追加2
    "seita": "セイタ", #追加2
    "seitaro": "セイタロウ", #追加2
    "seiya": "セイヤ", #追加2
    "setsu": "セツ", #追加2
    "setsuyuki": "セツユキ", #追加2
    "shigehiko": "シゲヒコ", #追加2
    "shigehiro": "シゲヒロ", #追加2
    "shigeki": "シゲキ", #追加2
    "shigemitsu": "シゲミツ", #追加2
    "shigenobu": "シゲノブ", #追加2
    "shigenori": "シゲノリ", #追加2
    "shigeo": "シゲオ", #追加2
    "shigeomi": "シゲオミ", #追加2
    "shigeru": "シゲル", #追加2
    "shigeshi": "シゲシ", #追加2
    "shigetaka": "シゲタカ", #追加2
    "shigeto": "シゲト", #追加2
    "shigetoshi": "シゲトシ", #追加2
    "shigeyasu": "シゲヤス", #追加2
    "shigeyoshi": "シゲヨシ", #追加2
    "shiho": "シホ", #追加2
    "shihoko": "シホコ", #追加2
    "shimpei": "シンペイ", #追加2
    "shin": "シン", #追加2
    "shingo": "シンゴ", #追加2
    "shinichi": "シンイチ", #追加2
    "shinichiro": "シンイチロウ", #追加2
    "shin-ichiro": "シンイチロウ", #追加2
    "shinji": "シンジ", #追加2
    "shinjo": "シンジョウ", #追加2
    "shinnosuke": "シンノスケ", #追加2
    "shinrou": "シンロウ", #追加2
    "shinsuke": "シンスケ", #追加2
    "shinta": "シンタ", #追加2
    "shintaro": "シンタロウ", #追加2
    "shinya": "シンヤ", #追加2
    "shiori": "シオリ", #追加2
    "shiro": "シロウ", #追加2
    "shirou": "シロウ", #追加2
    "sho": "ショウ",
    "shodai": "ショウダイ", #追加2
    "shogo": "ショウゴ", # ← 追加
    "shohei": "ショウヘイ", # ← 追加
    "shoichi": "ショウイチ",
    "sho-ichi": "ショウイチ", #追加2
    "shoichiro": "ショウイチロウ", #追加2
    "shoji": "ショウジ",
    "shojiro": "ショウジロウ", #追加2
    "shoko": "ショウコ",
    "shota": "ショウタ",
    "shotaro": "ショウタロウ",
    "shozo": "ショウゾウ", #追加2
    "shu": "シュウ", #追加2
    "shuhei": "シュウヘイ", #追加2
    "shuichi": "シュウイチ", #追加2
    "shu-ichi": "シュウイチ", #追加2
    "shuichiro": "シュウイチロウ", #追加2
    "shuji": "シュウジ", #追加2
    "shujiro": "シュウジロウ", #追加2
    "shuko": "シュウコ", #追加2
    "shumpei": "シュンペイ", #追加2
    "shun": "シュン", #追加2
    "shunbun": "シュンブン", #追加2
    "shungo": "シュンゴ", #追加2
    "shunich": "シュンイチ", #追加2
    "shunichi": "シュンイチ", #追加2
    "shunichiro": "シュンイチロウ",
    "shunji": "シュンジ", #追加2
    "shunsuke": "シュンスケ",
    "shunta": "シュンタ", #追加2
    "shuntaro": "シュンタロウ", #追加2
    "shunya": "シュンヤ", #追加2
    "shuro": "シュウロウ", #追加2
    "shusaku": "シュウサク", #追加2
    "shusuke": "シュウスケ",
    "shuta": "シュウタ", #追加2
    "so": "ソウ", #追加2
    "sohsuke": "ソウスケ",
    "soichi": "ソウイチ",
    "soichiro": "ソウイチロウ",
    "soken": "ソケン", #追加2
    "sonoka": "ソノカ", #追加2
    "soshi": "ソシ", #追加2
    "soshiro": "ソウシロウ", #追加2
    "sota": "ソウタ",
    "sou": "ソウ", #追加2
    "suga": "スガ", #追加2
    "suguru": "スグル", #追加2
    "sumio": "スミオ", #追加2
    "sunao": "スナオ", #追加2
    "susumu": "ススム", #追加2
    "suwako": "スワコ", #追加2
    "suzu": "スズ", #追加2
    "syotaro": "ショウタロウ", #追加2
    "syuntaro": "シュンタロウ", #追加2
    "tadaaki": "タダアキ", #追加2
    "tadahiro": "タダヒロ", #追加2
    "tadakazu": "タダカズ", #追加2
    "tadao": "タダオ", #追加2
    "tadashi": "タダシ", #追加2
    "tadateru": "タダテル", #追加2
    "tadayuki": "タダユキ", #追加2
    "taeko": "タエコ", #追加2
    "taichi": "タイチ", #追加2
    "taiji": "タイジ", #追加2
    "taikan": "タイカン", #追加2
    "taiki": "タイキ", #追加2
    "tairo": "タイロウ", #追加2
    "taishi": "タイシ", #追加2
    "taisuke": "タイスケ", #追加2
    "taito": "タイト", #追加2
    "taiyo": "タイヨウ", #追加2
    "taizo": "タイゾウ", #追加2
    "takaaki": "タカアキ", #追加2
    "taka-aki": "タカアキ", #追加2
    "takafumi": "タカフミ", #追加2
    "takaharu": "タカハル", #追加2
    "takahide": "タカヒデ", #追加2
    "takahiko": "タカヒコ", #追加2
    "takahiro": "タカヒロ", #追加2
    "takahisa": "タカヒサ", #追加2
    "takahito": "タカヒト", #追加2
    "takaki": "タカアキ", #追加2
    "takako": "タカコ", #追加2
    "takamasa": "タカマサ", #追加2
    "takamitsu": "タカミツ", #追加2
    "takanari": "タカナリ", #追加2
    "takanobu": "タカノブ", #追加2
    "takanori": "タカノリ", #追加2
    "takao": "タカオ", #追加2
    "takashi": "タカシ", #追加2
    "takashige": "タカシゲ", #追加2
    "takatomo": "タカトモ", #追加2
    "takatoshi": "タカトシ", #追加2
    "takatoyo": "タカトヨ", #追加2
    "takayo": "タカヨ", #追加2
    "takayoshi": "タカヨシ", #追加2
    "takayuki": "タカユキ", #追加2
    "takeaki": "タケアキ", #追加2
    "takefumi": "タケフミ", #追加2
    "takehiko": "タケヒコ", #追加2
    "takehiro": "タケヒロ", #追加2
    "takenao": "タケナオ", #追加2
    "takenobu": "タケノブ", #追加2
    "takenori": "タケノリ", #追加2
    "takeo": "タケオ", #追加2
    "takeru": "タケル", #追加2
    "takeshi": "タケシ", #追加2
    "takeshige": "タケシゲ", #追加2
    "taketo": "タケト", #追加2
    "taketoshi": "タケトシ", #追加2
    "takeyoshi": "タケヨシ", #追加2
    "takeyuki": "タケユキ", #追加2
    "taku": "タク", #追加2
    "takuji": "タクジ", #追加2
    "takuma": "タクマ", #追加2
    "takumi": "タクミ", #追加2
    "takunori": "タクノリ", #追加2
    "takuo": "タクオ", #追加2
    "takuro": "タクロウ", #追加2
    "takuto": "タクト", #追加2
    "takuya": "タクヤ", #追加2
    "tamahiro": "タマヒロ", #追加2
    "tamaki": "タマキ", #追加2
    "tamiko": "タミコ", #追加2
    "tamio": "タミオ", #追加2
    "tamiyuki": "タミユキ", #追加2
    "tamon": "タモン", #追加2
    "taro": "タロウ",
    "tasuku": "タスク", #追加2
    "tatsuhiko": "タツヒコ", #追加2
    "tatsuhiro": "タツヒロ", #追加2
    "tatsuhisa": "タツヒサ", #追加2
    "tatsuji": "タツジ", #追加2
    "tatsuki": "タツキ", #追加2
    "tatsunori": "タツノリ", #追加2
    "tatsuo": "タツオ", #追加2
    "tatsuro": "タツロウ", #追加2
    "tatsushi": "タツシ", #追加2
    "tatsuto": "タツト", #追加2
    "tatsuya": "タツヤ", #追加2
    "teisuke": "テイスケ", #追加2
    "tenjin": "テンジン", #追加2
    "teruaki": "テルアキ", #追加2
    "teruhiko": "テルヒコ", #追加2
    "teruhiro": "テルヒロ", #追加2
    "teruki": "テルキ", #追加2
    "terumasa": "テルマサ", #追加2
    "terumitsu": "テルミツ", #追加2
    "terumori": "テルモリ", #追加2
    "terumoto": "テルモト", #追加2
    "teruo": "テルオ", #追加2
    "teruyasu": "テルヤス", #追加2
    "teruyoshi": "テルヨシ", #追加2
    "tetsu": "テツ", #追加2
    "tetsuharu": "テツハル", #追加2
    "tetsuhiro": "テツヒロ", #追加2
    "tetsuhisa": "テツヒサ", #追加2
    "tetsuichi": "テツイチ", #追加2
    "tetsuji": "テツジ", #追加2
    "tetsuo": "テツオ", #追加2
    "tetsuro": "テツロウ", #追加2
    "tetsuya": "テツヤ", #追加2
    "tetsuzo": "テツゾウ", #追加2
    "tohru": "トオル",
    "tokuhiro": "トクヒロ", #追加2
    "tokutada": "トクタダ", #追加2
    "tomiharu": "トミハル", #追加2
    "tomihisa": "トミヒサ", #追加2
    "tomo": "トモ", #追加2
    "tomoaki": "トモアキ", #追加2
    "tomofumi": "トモフミ", #追加2
    "tomoharu": "トモハル", #追加2
    "tomohiko": "トモヒコ", #追加2
    "tomohiro": "トモヒロ", #追加2
    "tomohisa": "トモヒサ", #追加2
    "tomokazu": "トモカズ", #追加2
    "tomoki": "トモキ", #追加2
    "tomoko": "トモコ", #追加2
    "tomomi": "トモミ", #追加2
    "tomonori": "トモノリ", #追加2
    "tomoo": "トモオ", #追加2
    "tomotaka": "トモタカ", #追加2
    "tomotsugu": "トモツグ", #追加2
    "tomoya": "トモヤ", #追加2
    "tomoyo": "トモヨ", #追加2
    "tomoyuki": "トモユキ", #追加2
    "toraaki": "トラアキ", #追加2
    "toru": "トオル",
    "toshi": "トシ", #追加2
    "toshiaki": "トシアキ", #追加2
    "toshifumi": "トシフミ", #追加2
    "toshiharu": "トシハル", #追加2
    "toshihide": "トシヒデ", #追加2
    "toshihiko": "トシヒコ", #追加2
    "toshihiro": "トシヒロ", #追加2
    "toshihisa": "トシヒサ", #追加2
    "toshikazu": "トシカズ", #追加2
    "toshiki": "トシキ", #追加2
    "toshiko": "トシコ", #追加2
    "toshimasa": "トシマサ", #追加2
    "toshimitsu": "トシミツ", #追加2
    "toshio": "トシオ", #追加2
    "toshiro": "トシロウ",
    "toshitaka": "トシタカ", #追加2
    "toshiya": "トシヤ", #追加2
    "toshiyuki": "トシユキ", #追加2
    "toyoaki": "トヨアキ", #追加2
    "toyoki": "トヨキ", #追加2
    "tsugumi": "ツグミ", #追加2
    "tsukasa": "ツカサ", #追加2
    "tsunekazu": "ツネカズ", #追加2
    "tsuneki": "ツネキ", #追加2
    "tsunenari": "ツネナリ", #追加2
    "tsuneo": "ツネオ", #追加2
    "tsunetatsu": "ツネタツ", #追加2
    "tsutomu": "ツトム", #追加2
    "tsuyoshi": "ツヨシ", #追加2
    "uichi": "ウイチ", #追加2
    "umihiko": "ウミヒコ", #追加2
    "wakana": "ワカナ", #追加2
    "wakaya": "ワカヤ", #追加2
    "wataru": "ワタル", #追加2
    "yae": "ヤエ", #追加2
    "yamato": "ヤマト", #追加2
    "yasuaki": "ヤスアキ", #追加2
    "yasuchika": "ヤスチカ", #追加2
    "yasufumi": "ヤスフミ", #追加2
    "yasuharu": "ヤスハル", #追加2
    "yasuhide": "ヤスヒデ", #追加2
    "yasuhiko": "ヤスヒコ", #追加2
    "yasuhiro": "ヤスヒロ", #追加2
    "yasuhisa": "ヤスヒサ", #追加2
    "yasuhito": "ヤスヒト", #追加2
    "yasuki": "ヤスキ", #追加2
    "yasuko": "ヤスコ", #追加2
    "yasumasa": "ヤスマサ", #追加2
    "yasumi": "ヤスミ", #追加2
    "yasunari": "ヤスナリ", #追加2
    "yasunobu": "ヤスノブ", #追加2
    "yasunori": "ヤスノリ", #追加2
    "yasuo": "ヤスオ", #追加2
    "yasuomi": "ヤスオミ", #追加2
    "yasushi": "ヤスシ", #追加2
    "yasutaka": "ヤスタカ", #追加2
    "yasutomi": "ヤストミ", #追加2
    "yasutomo": "ヤストモ", #追加2
    "yasutoshi": "ヤストシ", #追加2
    "yasutsugu": "ヤスツグ", #追加2
    "yasuya": "ヤスヤ", #追加2
    "yasuyo": "ヤスヨ", #追加2
    "yasuyuki": "ヤスユキ", #追加2
    "yayoi": "ヤヨイ", #追加2
    "yo": "ヨウ", #追加2
    "yodo": "ヨウドウ", #追加2
    "yohei": "ヨウヘイ",
    "yohsuke": "ヨウスケ", #追加2
    "yoichi": "ヨウイチ", #追加2
    "yoichiro": "ヨウイチロウ",
    "yoji": "ヨウジ", #追加2
    "yoko": "ヨウコ",
    "yoku": "ヨク", #追加2
    "yorihiko": "ヨリヒコ", #追加2
    "yoritaka": "ヨリタカ", #追加2
    "yoriyasu": "ヨリヤス", #追加2
    "yoshiaki": "ヨシアキ", #追加2
    "yoshifumi": "ヨシフミ", #追加2
    "yoshifusa": "ヨシフサ", #追加2
    "yoshiharu": "ヨシハル", #追加2
    "yoshihide": "ヨシヒデ", #追加2
    "yoshihiko": "ヨシヒコ", #追加2
    "yoshihiro": "ヨシヒロ", #追加2
    "yoshihisa": "ヨシヒサ", #追加2
    "yoshihito": "ヨシヒト", #追加2
    "yoshikazu": "ヨシカズ", #追加2
    "yoshiki": "ヨシキ", #追加2
    "yoshiko": "ヨシコ", #追加2
    "yoshimasa": "ヨシマサ", #追加2
    "yoshimi": "ヨシミ", #追加2
    "yoshimitsu": "ヨシミツ", #追加2
    "yoshinari": "ヨシナリ", #追加2
    "yoshinobu": "ヨシノブ", #追加2
    "yoshinori": "ヨシノリ", #追加2
    "yoshio": "ヨシオ", #追加2
    "yoshiro": "ヨシロウ", #追加2
    "yoshisato": "ヨシサト", #追加2
    "yoshisuke": "ヨシスケ", #追加2
    "yoshitaka": "ヨシタカ", #追加2
    "yoshitake": "ヨシタケ", #追加2
    "yoshiteru": "ヨシテル", #追加2
    "yoshito": "ヨシト", #追加2
    "yoshitsugu": "ヨシツグ", #追加2
    "yoshiya": "ヨシヤ", #追加2
    "yoshiyasu": "ヨシヤス", #追加2
    "yoshiyuki": "ヨシユキ", #追加2
    "yosuke": "ヨウスケ",
    "yota": "ヨウタ", #追加2
    "youichi": "ヨウイチ", #追加2
    "yousuke": "ヨウスケ", #追加2
    "yu": "ユウ",
    "yudai": "ユウダイ", # ← 追加
    "yuetsu": "ユウエツ",
    "yugo": "ユウゴ", #追加2
    "yuhei": "ユウヘイ", #追加2
    "yui": "ユイ", #追加2
    "yuichi": "ユウイチ", #追加2
    "yuichiro": "ユウイチロウ", #追加2
    "yuichirou": "ユウイチロウ", #追加2
    "yuji": "ユウジ",
    "yujiro": "ユウジロウ",
    "yu-ki": "ユウキ", #追加2
    "yuki": "ユキ", #追加2
    "yukichi": "ユキチ", #追加2
    "yukie": "ユキエ", #追加2
    "yukihiko": "ユキヒコ", #追加2
    "yukihiro": "ユキヒロ", #追加2
    "yukihito": "ユキヒト", #追加2
    "yukiko": "ユキコ", #追加2
    "yukinori": "ユキノリ", #追加2
    "yukio": "ユキオ", #追加2
    "yuko": "ユウコ",
    "yuma": "ユウマ",
    "yumi": "ユミ", #追加2
    "yumika": "ユミカ", #追加2
    "yumiko": "ユミコ", #追加2
    "yunosuke": "ユウノスケ", #追加2
    "yuriko": "ユリコ", #追加2
    "yusaku": "ユウサク", #追加2
    "yushi": "ユウシ", #追加2
    "yusuke": "ユウスケ",
    "yuta": "ユウタ",
    "yutaka": "ユタカ", #追加2
    "yutaro": "ユウタロウ", #追加2
    "yuto": "ユウト",
    "yuuki": "ユウキ", #追加2
    "yuusuke": "ユウスケ", #追加2
    "yuya": "ユウヤ",
    "yuzo": "ユウゾウ", #追加2
    "zenichi": "ゼンイチ", #追加2
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

st.title("InsighTCROSS® Literature Scorer")
st.caption("Europe PMC（APIキー不要）を利用して文献を検索・著者スコア化します。")
if check_signin():
    # 認証後のみ表示されるサイドバー（ログアウト＋検索条件）
    with st.sidebar:
        if st.button("ログアウト"):
            st.session_state["signed_in"] = False
            st.session_state.pop("user", None)
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
