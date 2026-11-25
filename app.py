import json
from pathlib import Path

import base64

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =====================================================
# 1. KONFIGURASI
# =====================================================

FOLDER = Path(__file__).resolve().parent
FILE_LOGO = FOLDER / "logo.png"

try:
    from PIL import Image
    _ikon = Image.open(FILE_LOGO)
except Exception:
    _ikon = "📊"

st.set_page_config(
    page_title="Churn Analytics | Sinergi Wahana Gemilang",
    page_icon=_ikon,
    layout="wide",
    initial_sidebar_state="expanded",
)

FOLDER = Path(__file__).resolve().parent
FILE_MODEL = FOLDER / "model_churn_m8.joblib"
FILE_KATEGORI = FOLDER / "kategori_model.json"

FITUR_KATEGORIKAL = [
    "End User Category",
    "Product Class",
    "Stock Keeping Unit",
    "Contract Term",
    "Is First Contract",
]
FITUR_NUMERIK = [
    "Qty",
    "Days Between Contracts",
    "Total Contracts per CSN",
]
FITUR_MODEL = FITUR_KATEGORIKAL + FITUR_NUMERIK

# Metrik evaluasi M8 (samakan dengan Bab IV). Tulis sebagai persen.
METRIK_MODEL = {
    "Accuracy": 83.24,
    "Precision": 83.91,
    "Recall": 88.13,
    "F1-Score": 85.97,
}

# Batas pengamanan input jika "rentang" belum ada di kategori_model.json.
RENTANG_DEFAULT = {
    "Qty": [1, 1000],
    "Days Between Contracts": [0, 3650],
    "Total Contracts per CSN": [1, 100],
}

# Ambang kategori risiko (probabilitas churn)
BATAS_RENDAH = 0.40
BATAS_TINGGI = 0.70

# Label tampilan -> nilai asli yang dikirim ke model
LABEL_TERM = {"2-year": "2 Tahun", "3-Year": "3 Tahun", "Annual": "1 Tahun (Annual)"}


def persen(nilai, desimal=2):
    """Format persen dengan koma desimal (gaya Indonesia)."""
    return f"{nilai:.{desimal}f}".replace(".", ",") + "%"


# =====================================================
# 2. LOAD MODEL DAN DATA KATEGORI
# =====================================================

@st.cache_resource
def muat_model():
    return joblib.load(FILE_MODEL)


@st.cache_data
def muat_metadata():
    with open(FILE_KATEGORI, "r", encoding="utf-8") as file:
        isi = json.load(file)
    rentang = {**RENTANG_DEFAULT, **isi.get("rentang", {})}
    return isi["kategori"], rentang


def hapus_hasil():
    st.session_state.pop("hasil_prediksi", None)


# =====================================================
# 3. INPUT, PREDIKSI, DAN PENJELASAN
# =====================================================

def buat_input_model(kategori, qty, pertama, jeda, total):
    data = {
        "End User Category": kategori["End User Category"],
        "Product Class": kategori["Product Class"],
        "Stock Keeping Unit": kategori["Stock Keeping Unit"],
        "Contract Term": kategori["Contract Term"],
        "Is First Contract": pertama,
        "Qty": float(qty),
        "Days Between Contracts": (
            np.nan if pertama == "Yes" else float(jeda)
        ),
        "Total Contracts per CSN": float(total),
    }
    return pd.DataFrame([data], columns=FITUR_MODEL)


def _float_numerik(data):
    data = data.copy()
    for kolom in FITUR_NUMERIK:
        data[kolom] = pd.to_numeric(data[kolom], errors="coerce").astype(float)
    return data


def peluang_churn(model, data):
    data = _float_numerik(data)
    posisi_churn = list(model.classes_).index(1)
    return float(model.predict_proba(data)[0, posisi_churn])


def peluang_banyak(model, data):
    data = _float_numerik(data)
    posisi_churn = list(model.classes_).index(1)
    return model.predict_proba(data)[:, posisi_churn]


def kategori_risiko(p):
    if p >= BATAS_TINGGI:
        return "Tinggi"
    if p >= BATAS_RENDAH:
        return "Sedang"
    return "Rendah"


def cek_konsistensi(pertama, total, jeda):
    """Kembalikan (daftar_error, daftar_peringatan) untuk kombinasi input."""
    error, peringatan = [], []
    if pertama == "Yes" and total is not None and total > 1:
        error.append(
            "Kontrak pertama (Is First Contract = Yes) tidak mungkin "
            "memiliki Total Contracts per CSN lebih dari 1."
        )
    if pertama == "No" and total is not None and total < 2:
        error.append(
            "Kontrak bukan yang pertama (Is First Contract = No) "
            "berarti Total Contracts per CSN minimal 2."
        )
    return error, peringatan


def cek_rentang(rentang, qty, jeda, total):
    peringatan = []
    for nama, nilai in [
        ("Qty", qty),
        ("Days Between Contracts", jeda),
        ("Total Contracts per CSN", total),
    ]:
        if nilai is None:
            continue
        low, high = rentang[nama]
        if nilai < low or nilai > high:
            peringatan.append(
                f"{nama} = {nilai:,.0f} berada di luar rentang data "
                f"yang wajar ({low:,.0f} – {high:,.0f}). Hasil prediksi "
                "kurang dapat diandalkan."
            )
    return peringatan


def hitung_kontribusi(pipeline, data):
    """
    Kontribusi tiap fitur terhadap satu prediksi (skala log-odds),
    memakai pred_contribs XGBoost (setara nilai SHAP untuk pohon).
    Nilai positif mendorong ke CHURN, negatif ke NON-CHURN.
    """
    import xgboost as xgb

    persiapan = pipeline.named_steps["persiapan"]
    encoding = pipeline.named_steps["encoding"]
    xgb_model = pipeline.named_steps["model"]

    # SMOTE hanya dipakai saat training, tidak diterapkan pada data baru.
    matriks = encoding.transform(persiapan.transform(data))
    matriks = matriks.toarray() if hasattr(matriks, "toarray") else matriks

    kontribusi = xgb_model.get_booster().predict(
        xgb.DMatrix(np.asarray(matriks, dtype=float)),
        pred_contribs=True,
    )[0]

    encoder = encoding.named_transformers_["kategori"]
    hasil, posisi = [], 0
    for nama, daftar in zip(FITUR_KATEGORIKAL, encoder.categories_):
        hasil.append((nama, float(kontribusi[posisi:posisi + len(daftar)].sum())))
        posisi += len(daftar)
    for nama in FITUR_NUMERIK:
        hasil.append((nama, float(kontribusi[posisi])))
        posisi += 1

    tabel = pd.DataFrame(hasil, columns=["Fitur", "Kontribusi"])
    tabel["Nilai Input"] = [
        "-" if pd.isna(data.at[0, f]) else str(data.at[0, f])
        for f in tabel["Fitur"]
    ]
    tabel["Arah"] = np.where(
        tabel["Kontribusi"] >= 0, "Menaikkan risiko", "Menurunkan risiko"
    )
    tabel["abs"] = tabel["Kontribusi"].abs()
    return tabel.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def hitung_bobot_fitur(pipeline):
    xgb_model = pipeline.named_steps["model"]
    encoder = pipeline.named_steps["encoding"].named_transformers_["kategori"]
    importance = np.asarray(xgb_model.feature_importances_, dtype=float)

    hasil, posisi = [], 0
    for nama, daftar in zip(FITUR_KATEGORIKAL, encoder.categories_):
        hasil.append([nama, float(importance[posisi:posisi + len(daftar)].sum())])
        posisi += len(daftar)
    for nama in FITUR_NUMERIK:
        hasil.append([nama, float(importance[posisi])])
        posisi += 1

    if posisi != len(importance) or len(hasil) != len(FITUR_MODEL):
        raise ValueError("Struktur fitur tidak sesuai dengan model.")

    tabel = pd.DataFrame(hasil, columns=["Fitur", "Bobot"])
    total = tabel["Bobot"].sum()
    tabel["Bobot (%)"] = tabel["Bobot"] / total * 100 if total > 0 else 0.0
    return tabel.sort_values("Bobot (%)", ascending=False).reset_index(drop=True)


# =====================================================
# 4. TOKEN DESAIN (palet diambil dari logo perusahaan)
# =====================================================

BIRU = "#3A97E0"       # biru logo, versi terang untuk grafik di latar gelap
BIRU_LOGO = "#006CB0"
ORANYE = "#F07229"     # oranye logo = satu-satunya aksen interaktif
CANVAS = "#0A111C"     # L0
SURFACE = "#111B2B"    # L1
SURFACE2 = "#172338"   # L2
GARIS = "#24344C"
TEKS = "#E8EEF6"
TEKS_REDAM = "#93A4BC"
STATUS = {"Rendah": "#3DB88A", "Sedang": "#E8B03A", "Tinggi": "#E5484D"}

SLIDER_MAX = {"Qty": 200, "Days Between Contracts": 1095, "Total Contracts per CSN": 30}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600&display=swap');

:root {{
  --canvas:{CANVAS}; --s1:{SURFACE}; --s2:{SURFACE2}; --line:{GARIS};
  --text:{TEKS}; --muted:{TEKS_REDAM}; --accent:{ORANYE}; --blue:{BIRU}; --blue-logo:{BIRU_LOGO};
}}
html, body, .stApp, [class*="css"] {{ font-family:'Manrope', system-ui, sans-serif; }}
.stApp {{ background:var(--canvas); }}
header[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ max-width:1280px; padding-top:1.5rem; padding-bottom:3rem; }}
#MainMenu, footer {{ visibility:hidden; }}

/* Tipografi: bobot 400/500/600 saja, 16/14/12 px */
b, strong, h1, h2, h3, h4, h5 {{ font-weight:600 !important; }}
h1, h2, h3 {{ line-height:1.2 !important; letter-spacing:-0.01em; color:var(--text); }}
p, li {{ line-height:1.55; }}
[data-testid="stCaptionContainer"] {{ color:var(--muted); font-size:0.875rem; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background:var(--s1); border-right:1px solid var(--line); }}
section[data-testid="stSidebar"] .block-container {{ padding-top:1.25rem; }}
.brand {{ display:flex; gap:.75rem; align-items:center; padding-bottom:1.1rem; margin-bottom:.9rem; border-bottom:1px solid var(--line); }}
.brand .tile {{ flex:0 0 44px; width:44px; height:44px; border-radius:10px; background:#fff; display:grid; place-items:center; padding:5px; }}
.brand .tile img {{ width:100%; height:100%; object-fit:contain; }}
.brand b {{ display:block; color:var(--text); font-size:.875rem; line-height:1.25; }}
.brand span {{ color:var(--muted); font-size:.75rem; }}
.navtitle {{ color:var(--muted); font-size:.75rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; margin:.2rem 0 .5rem; }}

.st-key-navwrap {{ gap:.2rem; }}
.st-key-navwrap button {{ width:100%; justify-content:flex-start; min-height:2.6rem; padding:.45rem .85rem; border:0;
  border-radius:8px; background:transparent; color:#B7C5DA; box-shadow:none; transition:background .15s, color .15s; }}
.st-key-navwrap button > div {{ width:100%; justify-content:flex-start; }}
.st-key-navwrap button p {{ font-size:.875rem; font-weight:500; text-align:left; width:100%; }}
.st-key-navwrap button:hover {{ background:var(--s2); color:#fff; }}
.st-key-navwrap button[kind="primary"] {{ background:rgba(240,114,41,.12); color:#fff; box-shadow:inset 3px 0 0 var(--accent); }}
.st-key-navwrap button[kind="primary"] p {{ font-weight:600; }}
.sidenote {{ margin-top:1.4rem; padding:.85rem; border:1px solid var(--line); border-radius:10px; background:var(--canvas); }}
.sidenote b {{ font-size:.875rem; color:var(--text); }}
.sidenote p {{ margin:.25rem 0 0; font-size:.75rem; line-height:1.5; color:var(--muted); }}

/* Header halaman */
.pagehead {{ position:relative; background:var(--s1); border:1px solid var(--line); border-radius:14px;
  padding:1.6rem 1.8rem 1.5rem; margin-bottom:1.4rem; overflow:hidden; }}
.pagehead:before {{ content:""; position:absolute; left:0; right:0; top:0; height:3px;
  background:linear-gradient(90deg, var(--blue-logo) 50%, var(--accent) 50%); }}
.pagehead small {{ display:block; color:var(--muted); font-size:.875rem; margin-bottom:.35rem; }}
.pagehead h1 {{ margin:0 0 .4rem !important; padding:0 !important; font-size:2rem !important; font-weight:600 !important; }}
.pagehead p {{ margin:0; color:#B7C5DA; font-size:1rem; max-width:62ch; }}

/* Seksi */
.sec {{ margin:1.6rem 0 .75rem; }}
.sec h3 {{ margin:0 !important; padding:0 !important; font-size:1.25rem !important; font-weight:600 !important; }}
.sec .sub {{ display:block; color:var(--muted); font-size:.875rem; margin-top:.2rem; }}

/* Kartu */
div[data-testid="stVerticalBlockBorderWrapper"] {{ background:var(--s1); border:1px solid var(--line) !important; border-radius:12px; }}
.kpi {{ background:var(--s1); border:1px solid var(--line); border-radius:12px; padding:1rem 1.1rem; height:100%; }}
.kpi small {{ display:block; color:var(--muted); font-size:.875rem; }}
.kpi b {{ display:block; color:#fff; font-size:1.75rem; font-weight:600; line-height:1.2; margin:.2rem 0; }}
.kpi em {{ color:var(--muted); font-size:.75rem; font-style:normal; }}

/* Hasil */
.result {{ text-align:center; padding:.4rem 0 .2rem; }}
.gauge {{ --p:50; --c:{ORANYE}; width:184px; height:184px; border-radius:50%; margin:.2rem auto 1rem;
  background:conic-gradient(var(--c) calc(var(--p)*1%), var(--s2) 0); display:grid; place-items:center; }}
.gauge .in {{ width:146px; height:146px; border-radius:50%; background:var(--s1); display:grid; place-content:center; }}
.gauge b {{ color:#fff; font-size:2rem; font-weight:600; line-height:1; }}
.gauge span {{ color:var(--muted); font-size:.75rem; margin-top:.3rem; }}
.verdict {{ font-size:1.125rem; font-weight:600; letter-spacing:.02em; }}
.verdict small {{ display:block; margin-top:.15rem; font-size:.875rem; font-weight:500; }}
.result p.hint {{ display:block; width:100%; color:var(--muted); font-size:.875rem; line-height:1.55; margin:.9rem 0 0 !important; padding:0 .6rem; text-align:center !important; }}
.result {{ text-align:center; }}
.result .verdict, .result .verdict small {{ text-align:center; }}

/* Empty state */
.empty {{ padding:1.6rem .6rem; text-align:left; }}
.empty h4 {{ margin:0 0 .3rem; font-size:1.125rem; color:var(--text); }}
.empty p {{ margin:0 0 .2rem; color:var(--muted); font-size:.875rem; }}

/* Widget */
[data-testid="stMain"] button[kind="primary"] {{ background:var(--accent); color:{CANVAS}; border:0; border-radius:8px;
  font-weight:600; min-height:2.7rem; transition:filter .15s; }}
[data-testid="stMain"] button[kind="primary"] p {{ font-weight:600; }}
[data-testid="stMain"] button[kind="primary"]:hover {{ filter:brightness(1.08); color:{CANVAS}; }}
[data-testid="stMain"] button[kind="secondary"] {{ background:transparent; border:1px solid var(--line); border-radius:8px; color:var(--text); min-height:2.7rem; }}
[data-testid="stMain"] button[kind="secondary"]:hover {{ border-color:var(--accent); color:#fff; }}
[data-testid="stDownloadButton"] button {{ border-radius:8px; }}
div[data-baseweb="select"] > div, div[data-testid="stNumberInput"] input, div[data-testid="stNumberInput"] > div > div {{ border-radius:8px; }}
button[data-baseweb="tab"] p {{ font-size:.875rem; font-weight:500; }}
button[data-baseweb="tab"][aria-selected="true"] p {{ font-weight:600; color:#fff; }}
div[data-baseweb="tab-highlight"] {{ background:var(--accent) !important; }}
div[data-testid="stExpander"] {{ border-radius:10px; border-color:var(--line); }}
.step {{ display:flex; gap:.9rem; padding:.9rem 1rem; border:1px solid var(--line); border-radius:10px; background:var(--s1); margin-bottom:.6rem; }}
.step .n {{ flex:0 0 28px; height:28px; border-radius:50%; background:var(--s2); border:1px solid var(--line); color:var(--text);
  display:grid; place-items:center; font-weight:600; font-size:.875rem; }}
.step b {{ color:var(--text); font-size:1rem; }} .step p {{ margin:.1rem 0 0; color:var(--muted); font-size:.875rem; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def logo_base64():
    try:
        return base64.b64encode(FILE_LOGO.read_bytes()).decode()
    except Exception:
        return ""


def img_logo():
    data = logo_base64()
    return f'<img src="data:image/png;base64,{data}" alt="Logo Sinergi Wahana Gemilang">' if data else "SWG"


def seksi(judul, ket=""):
    ket_html = f'<span class="sub">{ket}</span>' if ket else ""
    st.markdown(f'<div class="sec"><h3>{judul}</h3>{ket_html}</div>', unsafe_allow_html=True)


def kpi(label, nilai, sub=""):
    return f'<div class="kpi"><small>{label}</small><b>{nilai}</b><em>{sub}</em></div>'


def kepala(judul, deskripsi, konteks="Sinergi Wahana Gemilang"):
    st.markdown(
        f'<div class="pagehead"><small>{konteks}</small><h1>{judul}</h1><p>{deskripsi}</p></div>',
        unsafe_allow_html=True,
    )


def tampil(kolom, nilai):
    return LABEL_TERM.get(nilai, nilai) if kolom == "Contract Term" else nilai


# =====================================================
# 5. GRAFIK (Altair, mengikuti aturan dataviz)
# =====================================================

def _tema(chart):
    return (
        chart.configure(background="transparent", font="Manrope, sans-serif")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=TEKS_REDAM, titleColor=TEKS_REDAM, labelFontSize=12, titleFontSize=12,
            titleFontWeight=500, gridColor="#1B2A42", domainColor=GARIS, tickColor=GARIS,
        )
        .configure_legend(labelColor=TEKS, labelFontSize=12, titleColor=TEKS_REDAM, symbolType="square")
    )


def grafik_bobot(tabel):
    urut = tabel["Fitur"].tolist()
    dasar = alt.Chart(tabel).encode(
        y=alt.Y("Fitur:N", sort=urut, title=None, axis=alt.Axis(labelColor=TEKS, labelFontSize=13, labelLimit=220)),
        x=alt.X("Bobot (%):Q", title="Bobot (%)"),
    )
    batang = dasar.mark_bar(size=16, cornerRadiusEnd=4, color=BIRU).encode(
        tooltip=[alt.Tooltip("Fitur:N"), alt.Tooltip("Bobot (%):Q", format=".2f")]
    )
    label = dasar.mark_text(align="left", dx=6, color=TEKS, fontSize=12).encode(
        text=alt.Text("Bobot (%):Q", format=".1f")
    )
    return _tema((batang + label).properties(height=34 * len(tabel) + 30))


def grafik_kontribusi(tabel):
    urut = tabel["Fitur"].tolist()
    dasar = alt.Chart(tabel).encode(
        y=alt.Y("Fitur:N", sort=urut, title=None, axis=alt.Axis(labelColor=TEKS, labelFontSize=13, labelLimit=220)),
        x=alt.X("Kontribusi:Q", title="Kontribusi terhadap log-odds churn"),
    )
    batang = dasar.mark_bar(size=16, cornerRadiusEnd=4).encode(
        color=alt.Color(
            "Arah:N",
            scale=alt.Scale(domain=["Menaikkan risiko", "Menurunkan risiko"], range=[ORANYE, BIRU]),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=[
            alt.Tooltip("Fitur:N"), alt.Tooltip("Nilai Input:N"),
            alt.Tooltip("Kontribusi:Q", format="+.3f"), alt.Tooltip("Arah:N"),
        ],
    )
    nol = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=TEKS_REDAM, strokeWidth=1).encode(x="x:Q")
    return _tema((batang + nol).properties(height=34 * len(tabel) + 50))


def grafik_peta(df, kx, ky):
    urut_x = list(dict.fromkeys(df[kx]))
    urut_y = list(dict.fromkeys(df[ky]))
    dasar = alt.Chart(df).encode(
        x=alt.X(f"{kx}:N", sort=urut_x, title=None, axis=alt.Axis(labelAngle=0, labelColor=TEKS, labelFontSize=12, orient="top", labelLimit=160)),
        y=alt.Y(f"{ky}:N", sort=urut_y, title=None, axis=alt.Axis(labelColor=TEKS, labelFontSize=12, labelLimit=200)),
    )
    sel = dasar.mark_rect(stroke=CANVAS, strokeWidth=3, cornerRadius=4).encode(
        color=alt.Color(
            "Probabilitas:Q",
            scale=alt.Scale(domain=[0, 1], range=[SURFACE2, ORANYE]),
            legend=alt.Legend(title="Probabilitas churn", format="%", orient="bottom", gradientLength=240, tickCount=5),
        ),
        tooltip=[alt.Tooltip(f"{kx}:N"), alt.Tooltip(f"{ky}:N"), alt.Tooltip("Probabilitas:Q", format=".1%")],
    )
    teks = dasar.mark_text(fontSize=13, fontWeight=600).encode(
        text=alt.Text("Probabilitas:Q", format=".0%"),
        color=alt.condition(alt.datum.Probabilitas > 0.6, alt.value(CANVAS), alt.value(TEKS)),
    )
    return _tema((sel + teks).properties(height=max(120, 56 * len(urut_y))))


def grafik_kurva(df, kolom, nilai_kini, prob_kini):
    garis = alt.Chart(df).mark_line(color=BIRU, strokeWidth=2, interpolate="monotone").encode(
        x=alt.X(f"{kolom}:Q", title=kolom),
        y=alt.Y("Probabilitas:Q", title="Probabilitas churn", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
    )
    sentuh = alt.Chart(df).mark_circle(size=220, opacity=0).encode(
        x=f"{kolom}:Q", y="Probabilitas:Q",
        tooltip=[alt.Tooltip(f"{kolom}:Q", format=",.0f"), alt.Tooltip("Probabilitas:Q", format=".1%")],
    )
    ambang = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(color=TEKS_REDAM, strokeDash=[5, 4]).encode(y="y:Q")
    label_ambang = alt.Chart(pd.DataFrame({"y": [0.5], "t": ["Ambang klasifikasi 50%"]})).mark_text(
        align="left", dx=6, dy=-8, color=TEKS_REDAM, fontSize=12, x=4).encode(y="y:Q", text="t:N")
    kini = alt.Chart(pd.DataFrame({kolom: [nilai_kini], "Probabilitas": [prob_kini], "t": ["Nilai saat ini"]})).mark_point(
        filled=True, size=150, color=ORANYE, stroke=CANVAS, strokeWidth=2).encode(
        x=f"{kolom}:Q", y="Probabilitas:Q",
        tooltip=[alt.Tooltip("t:N", title=""), alt.Tooltip(f"{kolom}:Q", format=",.0f"), alt.Tooltip("Probabilitas:Q", format=".1%")],
    )
    return _tema((garis + sentuh + ambang + label_ambang + kini).properties(height=300))


# =====================================================
# 6. PEMUATAN FILE DAN STATE
# =====================================================

if not FILE_MODEL.exists() or not FILE_KATEGORI.exists():
    st.error("File model atau kategori tidak ditemukan.")
    st.info("Letakkan model_churn_m8.joblib dan kategori_model.json di folder yang sama dengan app.py.")
    st.stop()

try:
    model = muat_model()
    pilihan_kategori, rentang = muat_metadata()
except Exception as error:
    st.error(f"Gagal memuat model: {error}")
    st.info("Versi scikit-learn, xgboost, atau imbalanced-learn kemungkinan berbeda dengan versi saat model dilatih. "
            "Samakan versinya di requirements.txt.")
    st.stop()

st.session_state.setdefault("halaman", "Beranda")
st.session_state.setdefault("id_prediksi", 0)
st.session_state.setdefault("riwayat", [])

KUNCI_FORM = ["f_eu", "f_pc", "f_sku", "f_term", "f_qty", "f_first", "f_total", "f_jeda"]
for _k in KUNCI_FORM:
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]

CONTOH = {
    "tinggi": dict(eu="Construction", pc="AutoCAD IST", sku="02JI1-005995-L403", term="3-Year",
                   qty=10, first="No", total=6, jeda=700),
    "rendah": dict(eu="Food and beverage", pc="Navisworks", sku="02HI1-005995-L403", term="2-year",
                   qty=50, first="Yes", total=1, jeda=None),
}


# =====================================================
# 7. LOGIKA PREDIKSI DAN NAVIGASI
# =====================================================

def ke(halaman):
    st.session_state["halaman"] = halaman


def simpan_prediksi(data_input):
    kelas = int(model.predict(data_input)[0])
    p = peluang_churn(model, data_input)
    st.session_state["id_prediksi"] += 1
    st.session_state["hasil_prediksi"] = {
        "id": st.session_state["id_prediksi"], "kelas": kelas, "probabilitas": p, "data": data_input,
    }
    baris = data_input.iloc[0]
    st.session_state["riwayat"].insert(0, {
        "No": st.session_state["id_prediksi"],
        "End User": baris["End User Category"],
        "Product Class": baris["Product Class"],
        "Masa kontrak": tampil("Contract Term", baris["Contract Term"]),
        "Qty": int(baris["Qty"]),
        "Kontrak pertama": baris["Is First Contract"],
        "Probabilitas churn (%)": round(p * 100, 1),
        "Prediksi": "CHURN" if kelas == 1 else "NON-CHURN",
        "Risiko": kategori_risiko(p),
    })
    del st.session_state["riwayat"][50:]


def isi_contoh(nama):
    c = CONTOH[nama]
    st.session_state.update({
        "f_eu": c["eu"], "f_pc": c["pc"], "f_sku": c["sku"], "f_term": c["term"],
        "f_qty": c["qty"], "f_first": c["first"], "f_total": c["total"],
        "f_jeda": c["jeda"],
    })
    data = buat_input_model(
        {"End User Category": c["eu"], "Product Class": c["pc"],
         "Stock Keeping Unit": c["sku"], "Contract Term": c["term"]},
        c["qty"], c["first"], c["jeda"], c["total"],
    )
    simpan_prediksi(data)
    st.session_state["halaman"] = "Prediksi Kontrak"


def hapus_riwayat():
    st.session_state["riwayat"] = []


def tampilkan_hasil(p, kelas=None, ringkas=False):
    """Kartu hasil: lingkaran probabilitas + status."""
    risiko = kategori_risiko(p)
    warna = STATUS[risiko]
    churn = (p >= 0.5) if kelas is None else kelas == 1
    label = "CHURN" if churn else "NON-CHURN"
    ket = "Kontrak berpotensi tidak diperpanjang." if churn else "Kontrak diprediksi berlanjut."
    ekstra = "" if ringkas else (
        f'<p class="hint">{ket} Peluang non-churn {persen((1 - p) * 100, 1)}. '
        f"Kategori risiko: rendah di bawah {BATAS_RENDAH:.0%}, tinggi mulai {BATAS_TINGGI:.0%}. "
        "Hasil adalah estimasi model, bukan kepastian.</p>"
    )
    st.markdown(
        f'<div class="result"><div class="gauge" style="--p:{p*100:.1f};--c:{warna}">'
        f'<div class="in"><b>{persen(p*100, 1)}</b><span>probabilitas churn</span></div></div>'
        f'<div class="verdict" style="color:{warna}">{label}<small>Risiko {risiko.lower()}</small></div>{ekstra}</div>',
        unsafe_allow_html=True,
    )


def grid_prediksi(basis, kol_x, kol_y):
    """Probabilitas churn untuk semua kombinasi dua fitur kategorikal, fitur lain tetap."""
    baris = []
    for y in pilihan_kategori[kol_y]:
        for x in pilihan_kategori[kol_x]:
            d = dict(basis)
            d[kol_x], d[kol_y] = x, y
            baris.append(d)
    data = pd.DataFrame(baris, columns=FITUR_MODEL)
    hasil = data[[kol_x, kol_y]].copy()
    hasil["Probabilitas"] = peluang_banyak(model, data)
    hasil[kol_x] = hasil[kol_x].map(lambda v: tampil(kol_x, v))
    hasil[kol_y] = hasil[kol_y].map(lambda v: tampil(kol_y, v))
    return hasil


# =====================================================
# 8. SIDEBAR
# =====================================================

HALAMAN = [
    "Beranda", "Prediksi Kontrak", "Simulator", "Prediksi Massal", "Performa Model", "Panduan",
]

with st.sidebar:
    st.markdown(
        f'<div class="brand"><div class="tile">{img_logo()}</div>'
        "<div><b>Sinergi Wahana Gemilang</b><span>Churn analytics</span></div></div>"
        '<div class="navtitle">Menu</div>',
        unsafe_allow_html=True,
    )
    with st.container(key="navwrap"):
        for nama in HALAMAN:
            st.button(
                nama, key=f"nav_{nama}", width="stretch",
                type="primary" if st.session_state["halaman"] == nama else "secondary",
                on_click=ke, args=(nama,),
            )

halaman = st.session_state["halaman"]


# =====================================================
# 9. HALAMAN: BERANDA
# =====================================================

def halaman_beranda():
    kepala(
        "Prediksi churn kontrak lisensi",
        "Perkirakan kontrak mana yang berisiko tidak diperpanjang, pahami alasannya, "
        "dan uji skenario sebelum menghubungi pelanggan.",
    )

    seksi("Kualitas model", "Hasil evaluasi model M8 pada data testing")
    kolom = st.columns(len(METRIK_MODEL))
    for k, (nama, nilai) in zip(kolom, METRIK_MODEL.items()):
        k.markdown(kpi(nama, persen(nilai), "data testing"), unsafe_allow_html=True)

    seksi("Mulai dari sini")
    a, b, c = st.columns(3, gap="medium")
    with a:
        with st.container(border=True):
            st.markdown("**Prediksi satu kontrak**")
            st.caption("Isi data kontrak, lihat probabilitas churn dan alasan di baliknya.")
            st.button("Buka prediksi kontrak", key="go_pred", on_click=ke, args=("Prediksi Kontrak",), width="stretch")
    with b:
        with st.container(border=True):
            st.markdown("**Uji skenario**")
            st.caption("Geser Qty, jarak kontrak, atau jumlah kontrak dan lihat risikonya berubah.")
            st.button("Buka simulator", key="go_sim", on_click=ke, args=("Simulator",), width="stretch")
    with c:
        with st.container(border=True):
            st.markdown("**Prediksi banyak kontrak**")
            st.caption("Unggah CSV, dapatkan prediksi tiap baris, lalu unduh hasilnya.")
            st.button("Buka prediksi massal", key="go_batch", on_click=ke, args=("Prediksi Massal",), width="stretch")

    kiri, kanan = st.columns([1.25, 0.75], gap="large")
    with kiri:
        seksi("Faktor yang paling memengaruhi churn", "Tingkat kepentingan fitur menurut model")
        try:
            with st.container(border=True):
                st.altair_chart(grafik_bobot(hitung_bobot_fitur(model)), width="stretch")
        except Exception as error:
            st.warning(f"Grafik belum dapat ditampilkan: {error}")
    with kanan:
        seksi("Coba contoh kontrak", "Prediksi langsung terisi")
        with st.container(border=True):
            st.caption("Dua kontrak contoh dengan hasil yang berlawanan, untuk melihat cara kerja dashboard.")
            st.button("Isi contoh risiko tinggi", key="ex_hi", on_click=isi_contoh, args=("tinggi",), type="primary", width="stretch")
            st.button("Isi contoh risiko rendah", key="ex_lo", on_click=isi_contoh, args=("rendah",), width="stretch")


# =====================================================
# 10. HALAMAN: PREDIKSI KONTRAK
# =====================================================

def halaman_prediksi():
    kepala(
        "Prediksi kontrak",
        "Masukkan data satu kontrak untuk melihat peluang churn dan faktor yang mendorongnya.",
    )

    kolom_input, kolom_hasil = st.columns([1.25, 0.75], gap="large")

    with kolom_input:
        seksi("Data kontrak", "Kolom bertanda * wajib diisi")
        with st.container(border=True):
            st.markdown("**Produk dan pelanggan**")
            kategori = {}
            kiri, kanan = st.columns(2)
            with kiri:
                kategori["End User Category"] = st.selectbox(
                    "End User Category *", pilihan_kategori["End User Category"],
                    index=None, placeholder="Pilih kategori", key="f_eu", on_change=hapus_hasil)
                kategori["Product Class"] = st.selectbox(
                    "Product Class *", pilihan_kategori["Product Class"],
                    index=None, placeholder="Pilih kelas produk", key="f_pc", on_change=hapus_hasil)
            with kanan:
                kategori["Stock Keeping Unit"] = st.selectbox(
                    "Stock Keeping Unit *", pilihan_kategori["Stock Keeping Unit"],
                    index=None, placeholder="Pilih SKU", key="f_sku", on_change=hapus_hasil)
                kategori["Contract Term"] = st.selectbox(
                    "Contract Term *", pilihan_kategori["Contract Term"],
                    index=None, placeholder="Pilih masa kontrak", key="f_term",
                    format_func=lambda v: LABEL_TERM.get(v, v), on_change=hapus_hasil)

            st.markdown("**Riwayat kontrak**")
            kiri2, kanan2 = st.columns(2)
            with kiri2:
                qty = st.number_input("Qty *", min_value=1, step=1, value=None,
                                      placeholder="Jumlah lisensi", key="f_qty", on_change=hapus_hasil)
                pertama = st.selectbox("Is First Contract *", ["Yes", "No"], index=None,
                                       placeholder="Pilih Yes atau No", key="f_first", on_change=hapus_hasil)
            with kanan2:
                total = st.number_input("Total Contracts per CSN *", min_value=1, step=1, value=None,
                                        placeholder="Jumlah kontrak", key="f_total", on_change=hapus_hasil)
                jeda = None
                if pertama == "No":
                    jeda = st.number_input("Days Between Contracts *", min_value=0, step=1, value=None,
                                           placeholder="Selisih hari", key="f_jeda", on_change=hapus_hasil)
                elif pertama == "Yes":
                    st.caption("Days Between Contracts tidak berlaku untuk kontrak pertama.")

            galat, _ = cek_konsistensi(pertama, total, jeda)
            for pesan in galat:
                st.error(pesan)
            for pesan in cek_rentang(rentang, qty, jeda, total):
                st.warning(pesan)

            if st.button("Prediksi kontrak", type="primary", width="stretch", key="btn_pred"):
                belum_lengkap = (
                    any(v is None for v in kategori.values())
                    or pertama is None or qty is None or total is None
                    or (pertama == "No" and jeda is None)
                )
                if belum_lengkap:
                    st.warning("Lengkapi semua kolom bertanda * sebelum memprediksi.")
                    hapus_hasil()
                elif galat:
                    st.warning("Perbaiki input yang tidak konsisten terlebih dahulu.")
                    hapus_hasil()
                else:
                    try:
                        simpan_prediksi(buat_input_model(kategori, qty, pertama, jeda, total))
                    except Exception as error:
                        hapus_hasil()
                        st.error(f"Prediksi gagal: {error}")

    with kolom_hasil:
        seksi("Hasil prediksi")
        with st.container(border=True):
            hasil = st.session_state.get("hasil_prediksi")
            if hasil is None:
                st.markdown(
                    '<div class="empty"><h4>Belum ada prediksi</h4>'
                    "<p>Isi data kontrak lalu pilih Prediksi kontrak. Hasil, risiko, dan alasannya muncul di sini.</p></div>",
                    unsafe_allow_html=True,
                )
                st.button("Isi contoh risiko tinggi", key="ex_hi2", on_click=isi_contoh, args=("tinggi",), width="stretch")
            else:
                tampilkan_hasil(hasil["probabilitas"], hasil["kelas"])

    seksi("Analisis prediksi", "Mengapa model memberi hasil ini")
    tab_alasan, tab_sim = st.tabs(["Alasan prediksi", "What-if"])

    with tab_alasan:
        hasil = st.session_state.get("hasil_prediksi")
        if hasil is None:
            st.markdown(
                '<div class="empty"><h4>Alasan prediksi belum tersedia</h4>'
                "<p>Jalankan prediksi untuk melihat faktor yang menaikkan atau menurunkan risiko kontrak ini.</p></div>",
                unsafe_allow_html=True,
            )
        else:
            try:
                tabel_k = hitung_kontribusi(model, hasil["data"])
                atas = tabel_k.iloc[0]
                st.markdown(
                    f"**{atas['Fitur']}** (nilai `{atas['Nilai Input']}`) adalah faktor terkuat pada kontrak ini "
                    f"dan **{atas['Arah'].lower()}**."
                )
                st.altair_chart(grafik_kontribusi(tabel_k), width="stretch")
                with st.expander("Lihat tabel kontribusi"):
                    st.dataframe(tabel_k.style.format({"Kontribusi": "{:+.3f}"}), hide_index=True, width="stretch")
                st.caption("Berlaku untuk satu kontrak dan menunjukkan pola yang dipelajari model, "
                           "bukan hubungan sebab-akibat.")
            except Exception as error:
                st.warning(f"Alasan prediksi belum dapat ditampilkan: {error}")

    with tab_sim:
        render_whatif()

    seksi("Riwayat sesi", "Prediksi yang Anda jalankan selama sesi ini")
    riwayat = st.session_state["riwayat"]
    if not riwayat:
        with st.container(border=True):
            st.markdown(
                '<div class="empty"><h4>Belum ada riwayat</h4>'
                "<p>Setiap prediksi tersimpan di sini sehingga Anda bisa membandingkan beberapa kontrak.</p></div>",
                unsafe_allow_html=True,
            )
    else:
        st.dataframe(pd.DataFrame(riwayat), hide_index=True, width="stretch")
        st.button("Hapus riwayat", key="clr_hist", on_click=hapus_riwayat)


def render_whatif():
    hasil = st.session_state.get("hasil_prediksi")
    if hasil is None:
        st.markdown(
            '<div class="empty"><h4>What-if belum tersedia</h4>'
            "<p>Jalankan prediksi, lalu ubah satu fitur untuk melihat pergeseran probabilitasnya.</p></div>",
            unsafe_allow_html=True,
        )
        return

    data_asli = hasil["data"]
    probabilitas_awal = hasil["probabilitas"]
    id_hasil = hasil["id"]

    fitur = st.selectbox("Fitur yang diubah", FITUR_MODEL, index=FITUR_MODEL.index("Qty"),
                         key=f"sim_fitur_{id_hasil}")
    data_simulasi = data_asli.copy()
    nilai_asli = data_asli.at[0, fitur]
    boleh_dihitung = True
    kunci = f"sim_{id_hasil}_{fitur}"

    if fitur in pilihan_kategori:
        pilihan = pilihan_kategori[fitur]
        nilai_baru = st.selectbox(
            "Nilai untuk simulasi", pilihan, index=pilihan.index(str(nilai_asli)),
            format_func=(lambda v: LABEL_TERM.get(v, v)) if fitur == "Contract Term" else str, key=kunci)
        data_simulasi.at[0, fitur] = nilai_baru
    elif fitur == "Is First Contract":
        nilai_baru = st.selectbox("Nilai untuk simulasi", ["Yes", "No"],
                                  index=["Yes", "No"].index(nilai_asli), key=kunci)
        data_simulasi.at[0, fitur] = nilai_baru
        if nilai_baru == "Yes":
            data_simulasi.at[0, "Days Between Contracts"] = np.nan
            data_simulasi.at[0, "Total Contracts per CSN"] = 1.0
            st.caption("Kontrak pertama: Total Contracts per CSN otomatis 1.")
        elif nilai_asli == "Yes":
            hari = st.number_input("Days Between Contracts untuk simulasi", min_value=0, value=0, step=1, key=kunci + "_hari")
            tot = st.number_input("Total Contracts per CSN untuk simulasi", min_value=2, value=2, step=1, key=kunci + "_total")
            data_simulasi.at[0, "Days Between Contracts"] = float(hari)
            data_simulasi.at[0, "Total Contracts per CSN"] = float(tot)
    elif fitur == "Days Between Contracts" and pd.isna(nilai_asli):
        st.info("Fitur ini tidak berlaku untuk kontrak pertama.")
        boleh_dihitung = False
    elif fitur == "Total Contracts per CSN" and data_asli.at[0, "Is First Contract"] == "Yes":
        st.info("Pada kontrak pertama nilainya selalu 1. Ubah Is First Contract lebih dulu untuk mensimulasikan perpanjangan.")
        boleh_dihitung = False
    else:
        batas = 0 if fitur == "Days Between Contracts" else (2 if fitur == "Total Contracts per CSN" else 1)
        nilai_baru = st.number_input("Nilai untuk simulasi", min_value=batas,
                                     value=max(batas, int(nilai_asli)), step=1, key=kunci)
        data_simulasi.at[0, fitur] = float(nilai_baru)

    if not boleh_dihitung:
        return
    try:
        p_baru = peluang_churn(model, data_simulasi)
        selisih = (p_baru - probabilitas_awal) * 100
        a, b, c = st.columns(3)
        a.markdown(kpi("Probabilitas awal", persen(probabilitas_awal * 100)), unsafe_allow_html=True)
        b.markdown(kpi("Setelah perubahan", persen(p_baru * 100)), unsafe_allow_html=True)
        c.markdown(kpi("Selisih", f"{selisih:+.2f}".replace(".", ",") + " poin %"), unsafe_allow_html=True)
        st.write("")
        if selisih > 0:
            st.warning("Perubahan ini menaikkan probabilitas churn menurut model.")
        elif selisih < 0:
            st.success("Perubahan ini menurunkan probabilitas churn menurut model.")
        else:
            st.caption("Tidak ada perubahan probabilitas yang terukur.")
        st.caption("Simulasi menghitung ulang hasil model dengan satu nilai diganti. Ini bukan bukti sebab-akibat.")
    except Exception as error:
        st.warning(f"Simulasi gagal: {error}")


# =====================================================
# 11. HALAMAN: SIMULATOR
# =====================================================

def halaman_simulator():
    kepala(
        "Simulator skenario",
        "Geser nilai kontrak dan lihat probabilitas churn berubah langsung. "
        "Peta risiko membandingkan semua kombinasi produk dan masa kontrak sekaligus.",
    )

    hasil = st.session_state.get("hasil_prediksi")
    awal = hasil["data"].iloc[0] if hasil is not None else None

    def awalnya(kolom, default):
        if awal is None:
            return default
        nilai = awal[kolom]
        return default if (isinstance(nilai, float) and np.isnan(nilai)) else nilai

    kiri, kanan = st.columns([0.8, 1.2], gap="large")

    with kiri:
        seksi("Atur skenario")
        with st.container(border=True):
            sim = {}
            sim["End User Category"] = st.selectbox(
                "End User Category", pilihan_kategori["End User Category"],
                index=pilihan_kategori["End User Category"].index(awalnya("End User Category", pilihan_kategori["End User Category"][0])),
                key="s_eu")
            sim["Product Class"] = st.selectbox(
                "Product Class", pilihan_kategori["Product Class"],
                index=pilihan_kategori["Product Class"].index(awalnya("Product Class", pilihan_kategori["Product Class"][0])),
                key="s_pc")
            sim["Stock Keeping Unit"] = st.selectbox(
                "Stock Keeping Unit", pilihan_kategori["Stock Keeping Unit"],
                index=pilihan_kategori["Stock Keeping Unit"].index(awalnya("Stock Keeping Unit", pilihan_kategori["Stock Keeping Unit"][0])),
                key="s_sku")
            sim["Contract Term"] = st.selectbox(
                "Contract Term", pilihan_kategori["Contract Term"],
                index=pilihan_kategori["Contract Term"].index(awalnya("Contract Term", pilihan_kategori["Contract Term"][0])),
                format_func=lambda v: LABEL_TERM.get(v, v), key="s_term")
            pertama = st.radio("Jenis kontrak", ["Kontrak pertama", "Perpanjangan"],
                               index=0 if awalnya("Is First Contract", "No") == "Yes" else 1,
                               horizontal=True, key="s_first")
            sim["Is First Contract"] = "Yes" if pertama == "Kontrak pertama" else "No"
            sim["Qty"] = st.slider("Qty", 1, SLIDER_MAX["Qty"],
                                   int(min(max(1, awalnya("Qty", 5)), SLIDER_MAX["Qty"])), key="s_qty")
            if sim["Is First Contract"] == "No":
                sim["Days Between Contracts"] = st.slider(
                    "Days Between Contracts", 0, SLIDER_MAX["Days Between Contracts"],
                    int(min(awalnya("Days Between Contracts", 180), SLIDER_MAX["Days Between Contracts"])), step=5, key="s_days")
                sim["Total Contracts per CSN"] = st.slider(
                    "Total Contracts per CSN", 2, SLIDER_MAX["Total Contracts per CSN"],
                    int(min(max(2, awalnya("Total Contracts per CSN", 3)), SLIDER_MAX["Total Contracts per CSN"])), key="s_total")
            else:
                sim["Days Between Contracts"] = np.nan
                sim["Total Contracts per CSN"] = 1
                st.caption("Kontrak pertama: Days Between Contracts tidak berlaku dan Total Contracts per CSN bernilai 1.")
            st.caption(f"Rentang slider dibatasi: Qty sampai {SLIDER_MAX['Qty']}, jarak kontrak sampai "
                       f"{SLIDER_MAX['Days Between Contracts']:,} hari.".replace(",", "."))

        data_sim = pd.DataFrame([sim], columns=FITUR_MODEL)
        p = peluang_churn(model, data_sim)
        seksi("Hasil skenario")
        with st.container(border=True):
            tampilkan_hasil(p, ringkas=True)

    with kanan:
        tab_kurva, tab_peta = st.tabs(["Kurva sensitivitas", "Peta risiko"])

        with tab_kurva:
            st.caption("Ubah satu fitur sepanjang rentangnya sambil menjaga yang lain tetap.")
            pilihan_fitur = ["Qty"]
            if sim["Is First Contract"] == "No":
                pilihan_fitur += ["Days Between Contracts", "Total Contracts per CSN"]
            fitur = st.selectbox("Fitur yang divariasikan", pilihan_fitur, key="s_kurva")
            if fitur == "Qty":
                xs = np.arange(1, SLIDER_MAX["Qty"] + 1, 2)
            elif fitur == "Days Between Contracts":
                xs = np.arange(0, SLIDER_MAX[fitur] + 1, 15)
            else:
                xs = np.arange(2, SLIDER_MAX[fitur] + 1)
            baris = []
            for x in xs:
                d = dict(sim)
                d[fitur] = float(x)
                baris.append(d)
            df_kurva = pd.DataFrame(baris, columns=FITUR_MODEL)
            df_plot = pd.DataFrame({fitur: xs.astype(float), "Probabilitas": peluang_banyak(model, df_kurva)})
            with st.container(border=True):
                st.altair_chart(grafik_kurva(df_plot, fitur, float(sim[fitur]), p), width="stretch")
            st.caption("Titik oranye adalah nilai skenario Anda. Garis putus-putus menandai ambang klasifikasi 50%.")

        with tab_peta:
            st.caption("Probabilitas churn untuk setiap kombinasi, dengan fitur lain mengikuti skenario di kiri.")
            pasangan = {
                "Product Class dan Contract Term": ("Contract Term", "Product Class"),
                "End User Category dan Product Class": ("Product Class", "End User Category"),
                "Stock Keeping Unit dan Contract Term": ("Contract Term", "Stock Keeping Unit"),
            }
            pilih = st.selectbox("Bandingkan", list(pasangan), key="s_peta")
            kx, ky = pasangan[pilih]
            df_peta = grid_prediksi(sim, kx, ky)
            with st.container(border=True):
                st.altair_chart(grafik_peta(df_peta, kx, ky), width="stretch")
            tertinggi = df_peta.loc[df_peta["Probabilitas"].idxmax()]
            terendah = df_peta.loc[df_peta["Probabilitas"].idxmin()]
            st.markdown(
                f"Risiko tertinggi: **{tertinggi[ky]}** dengan **{tertinggi[kx]}** "
                f"({persen(tertinggi['Probabilitas']*100, 1)}). "
                f"Terendah: **{terendah[ky]}** dengan **{terendah[kx]}** "
                f"({persen(terendah['Probabilitas']*100, 1)})."
            )


# =====================================================
# 12. HALAMAN: PREDIKSI MASSAL
# =====================================================

def halaman_batch():
    kepala("Prediksi massal",
           "Unggah banyak kontrak sekaligus, dapatkan prediksi dan kategori risiko tiap baris, lalu unduh hasilnya.")

    seksi("Format file", "Kolom wajib: " + ", ".join(FITUR_MODEL))
    contoh = pd.DataFrame([{
        "End User Category": pilihan_kategori["End User Category"][0],
        "Product Class": pilihan_kategori["Product Class"][0],
        "Stock Keeping Unit": pilihan_kategori["Stock Keeping Unit"][0],
        "Contract Term": pilihan_kategori["Contract Term"][0],
        "Is First Contract": "Yes", "Qty": 1,
        "Days Between Contracts": np.nan, "Total Contracts per CSN": 1,
    }])
    st.download_button("Unduh template CSV", contoh.to_csv(index=False).encode("utf-8"),
                       file_name="template_prediksi_churn.csv", mime="text/csv")
    st.caption("Kosongkan Days Between Contracts untuk kontrak pertama. Maksimal 5.000 baris.")

    seksi("Unggah data")
    berkas = st.file_uploader("Unggah file CSV", type="csv", label_visibility="collapsed")
    if berkas is None:
        with st.container(border=True):
            st.markdown(
                '<div class="empty"><h4>Belum ada file</h4>'
                "<p>Unggah CSV sesuai template untuk memprediksi banyak kontrak sekaligus dan mengunduh hasilnya.</p></div>",
                unsafe_allow_html=True,
            )
        return

    try:
        df = pd.read_csv(berkas)
        kurang = [k for k in FITUR_MODEL if k not in df.columns]
        if kurang:
            st.error("Kolom belum lengkap: " + ", ".join(kurang) + ". Gunakan template CSV di atas."); return
        if len(df) == 0:
            st.error("File tidak berisi data."); return
        if len(df) > 5000:
            st.error("Maksimal 5.000 baris per unggahan. Pecah file menjadi beberapa bagian."); return

        df = df.reset_index(drop=True)
        data_batch = df[FITUR_MODEL].copy()
        for k in FITUR_KATEGORIKAL:
            data_batch[k] = data_batch[k].astype(str).str.strip()
        for k in FITUR_NUMERIK:
            data_batch[k] = pd.to_numeric(data_batch[k], errors="coerce")

        masalah = []
        for k in FITUR_KATEGORIKAL[:4]:
            asing = ~data_batch[k].isin(pilihan_kategori[k])
            if asing.any():
                masalah.append(f"{k}: nilai tidak dikenal pada baris "
                               + ", ".join(str(i + 2) for i in data_batch.index[asing][:10]))
        if (~data_batch["Is First Contract"].isin(["Yes", "No"])).any():
            masalah.append("Is First Contract harus Yes atau No.")
        if data_batch[["Qty", "Total Contracts per CSN"]].isna().any().any():
            masalah.append("Qty dan Total Contracts per CSN wajib berupa angka.")
        pertama_ya = data_batch["Is First Contract"] == "Yes"
        data_batch.loc[pertama_ya, "Days Between Contracts"] = np.nan
        if data_batch.loc[~pertama_ya, "Days Between Contracts"].isna().any():
            masalah.append("Days Between Contracts wajib diisi untuk kontrak non-pertama.")
        if masalah:
            for pesan in masalah:
                st.error(pesan)
            return

        prob = peluang_banyak(model, data_batch)
        keluaran = df.copy()
        keluaran["Prediksi"] = np.where(prob >= 0.5, "CHURN", "NON-CHURN")
        keluaran["Probabilitas Churn (%)"] = (prob * 100).round(2)
        keluaran["Kategori Risiko"] = [kategori_risiko(p) for p in prob]

        n = len(keluaran)
        n_churn = int((keluaran["Prediksi"] == "CHURN").sum())
        tinggi = int((keluaran["Kategori Risiko"] == "Tinggi").sum())
        seksi("Ringkasan hasil")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(kpi("Total kontrak", f"{n:,}", "baris diproses"), unsafe_allow_html=True)
        c2.markdown(kpi("Diprediksi churn", f"{n_churn:,}", persen(n_churn / n * 100, 1) + " dari total"), unsafe_allow_html=True)
        c3.markdown(kpi("Diprediksi non-churn", f"{n - n_churn:,}", persen((n - n_churn) / n * 100, 1) + " dari total"), unsafe_allow_html=True)
        c4.markdown(kpi("Risiko tinggi", f"{tinggi:,}", "prioritas tindak lanjut"), unsafe_allow_html=True)

        seksi("Detail prediksi")
        st.dataframe(keluaran, hide_index=True, width="stretch")
        st.download_button("Unduh hasil prediksi", keluaran.to_csv(index=False).encode("utf-8"),
                           file_name="hasil_prediksi_churn.csv", mime="text/csv", type="primary")
    except Exception as error:
        st.error(f"File tidak dapat diproses: {error}")


# =====================================================
# 13. HALAMAN: PERFORMA MODEL
# =====================================================

def halaman_performa():
    kepala("Performa model",
           "Hasil evaluasi pada data testing, tingkat kepentingan fitur, dan konfigurasi model M8.")

    seksi("Metrik evaluasi", "Dataset testing penelitian")
    kolom = st.columns(len(METRIK_MODEL))
    for k, (nama, nilai) in zip(kolom, METRIK_MODEL.items()):
        k.markdown(kpi(nama, persen(nilai), "data testing"), unsafe_allow_html=True)
    st.write("")
    st.caption("Performa pada kontrak baru dapat berbeda dari hasil pengujian.")

    seksi("Feature importance", "Kepentingan delapan fitur menurut XGBoost")
    try:
        tabel_bobot = hitung_bobot_fitur(model)
        with st.container(border=True):
            st.altair_chart(grafik_bobot(tabel_bobot), width="stretch")
        with st.expander("Lihat tabel bobot"):
            st.dataframe(tabel_bobot[["Fitur", "Bobot (%)"]].style.format({"Bobot (%)": "{:.2f}%"}),
                         hide_index=True, width="stretch")
        st.caption("Bobot menunjukkan kepentingan fitur secara keseluruhan. Nilainya bukan probabilitas churn "
                   "dan tidak menunjukkan arah pengaruh pada satu kontrak.")
    except Exception as error:
        st.warning(f"Feature importance belum dapat ditampilkan: {error}")

    seksi("Konfigurasi model")
    kiri, kanan = st.columns(2, gap="large")
    with kiri:
        with st.container(border=True):
            st.markdown("**Hyperparameter hasil Optuna**")
            try:
                prm = model.named_steps["model"].get_params()
                st.dataframe(pd.DataFrame({
                    "Parameter": ["n_estimators", "max_depth", "learning_rate", "subsample", "random_state"],
                    "Nilai": [str(prm["n_estimators"]), str(prm["max_depth"]),
                              f"{prm['learning_rate']:.4f}", f"{prm['subsample']:.4f}", str(prm["random_state"])],
                }), hide_index=True, width="stretch")
            except Exception:
                st.caption("Hyperparameter tidak tersedia.")
    with kanan:
        with st.container(border=True):
            st.markdown("**Alur pemrosesan**")
            st.markdown(
                "1. Imputasi dan *ordinal encoding* fitur kategorikal, imputasi fitur numerik.\n"
                "2. **SMOTENC** menyeimbangkan kelas pada data latih. Tidak dijalankan saat prediksi.\n"
                "3. *One-hot encoding* fitur kategorikal.\n"
                "4. **XGBoost** menghasilkan probabilitas churn."
            )


# =====================================================
# 14. HALAMAN: PANDUAN
# =====================================================

def halaman_panduan():
    kepala("Panduan", "Cara membaca hasil dan batasan yang perlu diketahui sebelum dipakai untuk mengambil keputusan.")

    seksi("Cara memakai")
    langkah = [
        ("Isi data kontrak", "Pilih kategori, produk, SKU, dan masa kontrak, lalu isi Qty dan riwayat kontrak pelanggan."),
        ("Jalankan prediksi", "Model menghitung probabilitas churn dan mengelompokkannya ke risiko rendah, sedang, atau tinggi."),
        ("Baca alasan prediksi", "Lihat fitur mana yang menaikkan dan menurunkan risiko pada kontrak tersebut."),
        ("Uji skenario", "Di Simulator, geser nilai kontrak atau bandingkan kombinasi produk dan masa kontrak lewat peta risiko."),
    ]
    for i, (j, d) in enumerate(langkah, 1):
        st.markdown(f'<div class="step"><div class="n">{i}</div><div><b>{j}</b><p>{d}</p></div></div>',
                    unsafe_allow_html=True)

    seksi("Batasan")
    with st.container(border=True):
        st.markdown(
            "- Satu prediksi mewakili **satu kontrak lisensi**, bukan seluruh pelanggan.\n"
            "- Hasil adalah **estimasi model**, bukan kepastian churn.\n"
            "- Alasan prediksi, what-if, dan simulator menunjukkan **pola yang dipelajari model**, bukan sebab-akibat.\n"
            "- Input di luar rentang data training dapat menghasilkan prediksi yang kurang andal.\n"
            f"- Kategori risiko memakai ambang {BATAS_RENDAH:.0%} dan {BATAS_TINGGI:.0%}. "
            "Label churn atau non-churn memakai ambang klasifikasi model, yaitu 50%."
        )


# =====================================================
# 15. ROUTER
# =====================================================

{
    "Beranda": halaman_beranda,
    "Prediksi Kontrak": halaman_prediksi,
    "Simulator": halaman_simulator,
    "Prediksi Massal": halaman_batch,
    "Performa Model": halaman_performa,
    "Panduan": halaman_panduan,
}[halaman]()
