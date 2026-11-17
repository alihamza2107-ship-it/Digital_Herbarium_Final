
# Digital Herbarium - Streamlit App
# Developed by NanStack.tech
# https://www.nanstack.tech/
#
# Run:
# python -m streamlit run app_herbarium.py
#
# Required:
# herbarium_model.keras
# class_names.json
#
# Optional:
# metrics.json

import os
import json
import html
import hmac
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image, ImageOps

import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================= Configuration =======================

Image.MAX_IMAGE_PIXELS = None

st.set_page_config(
    page_title="Digital Herbarium",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = os.environ.get(
    "HERBARIUM_KERAS",
    "herbarium_model.keras"
)

CLASSES_PATH = os.environ.get(
    "HERBARIUM_CLASSES",
    "class_names.json"
)

METRICS_PATH = os.environ.get(
    "HERBARIUM_METRICS",
    "metrics.json"
)

NAN_URL = "https://www.nanstack.tech/"
NAN_CONTACT = "https://www.nanstack.tech/contact"

DISCLAIMER = (
    "This information is for educational purposes only "
    "and is not medical advice."
)

APP_USER = os.environ.get("HERBARIUM_USER", "NanStack")
APP_PASS = os.environ.get("HERBARIUM_PASS", "AliHamza")


# ======================= Plant Information =======================

SPECIES = [
    ('Achyranthes aspera', "Prickly Chaff Flower / Devil's Horsewhip",
     'Amaranthaceae',
     'Tropical and subtropical Asia and Africa; weed of roadsides and wasteland',
     "Widely used in traditional medicine (Ayurvedic 'Apamarga'). Also a very common weed"),

    ('Ageratum conyzoides', 'Billy Goat Weed',
     'Asteraceae',
     'Native to tropical America; invasive in Asia and Africa',
     'Used in folk medicine for wounds. ⚠️ Contains pyrrolizidine alkaloids that can be toxic to the liver. Do not consume'),

    ('Amaranthus viridis', 'Slender Amaranth / Green Amaranth',
     'Amaranthaceae',
     'Pantropical weed, common in Asia and Africa',
     'Edible leafy vegetable in many regions and used in folk medicine'),

    ('Anredera cordifolia', 'Madeira Vine / Mignonette Vine',
     'Basellaceae',
     'Native to South America; invasive climber in many warm regions',
     'Ornamental, leaves edible in some regions. Considered an invasive weed'),

    ('Chenopodium album', "Fat Hen / Lamb's Quarters (Bathua)",
     'Amaranthaceae',
     'Cosmopolitan weed of crops, common in South Asia',
     'Edible leafy vegetable (Bathua). Contains oxalates, so eat in moderation'),

    ('Cynodon dactylon', 'Bermuda Grass (Doob)',
     'Poaceae',
     'Cosmopolitan, common in warm regions including South Asia',
     'Lawn and fodder grass, used in traditional medicine. Can be a persistent weed and pollen allergen'),

    ('Cyperus rotundus', 'Purple Nutsedge / Nut Grass',
     'Cyperaceae',
     'Cosmopolitan, one of the most widespread weeds in the world',
     "Tubers used in traditional medicine (Ayurvedic 'Musta'). A very persistent weed"),

    ('Dentella repens', 'Dentella (creeping herb)',
     'Rubiaceae',
     'Damp ground in tropical Asia, Australasia and the Pacific',
     'Small creeping herb of moist places, with limited documented use'),

    ('Oenothera laciniata', 'Cutleaf Evening Primrose',
     'Onagraceae',
     'Native to North America; naturalised in other regions',
     'Roadside and waste-ground herb, occasionally grown as an ornamental'),

    ('Parthenium hysterophorus', 'Parthenium / Congress Grass',
     'Asteraceae',
     'Native to tropical America; highly invasive weed in South Asia and Australia',
     '⚠️ Invasive weed that can cause skin allergy and asthma in sensitive people. Avoid handling without protection'),
]


INFO_BY_SCI = {
    r[0]: {
        "common": r[1],
        "scientific": r[0],
        "family": r[2],
        "habitat": r[3],
        "uses": r[4]
    }
    for r in SPECIES
}


def read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


# ======================= Model Classes =======================

NAMES = [
    c.replace("_", " ")
    for c in read_json(
        CLASSES_PATH,
        sorted(r[0].replace(" ", "_") for r in SPECIES)
    )
]

INFO = [
    INFO_BY_SCI.get(
        n,
        {
            "common": n,
            "scientific": n,
            "family": "-",
            "habitat": "-",
            "uses": "-"
        }
    )
    for n in NAMES
]

METRICS = read_json(METRICS_PATH, {})


# ======================= Styling =======================

CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    display: none !important;
}

.block-container {
    padding-top: 1.1rem;
    padding-bottom: 1rem;
    max-width: 1180px;
}

.hero {
    background: linear-gradient(135deg, #1B5E20, #2E7D32, #43A047);
    color: white;
    border-radius: 22px;
    padding: 26px 28px;
    box-shadow: 0 10px 30px rgba(27,94,32,.28);
    position: relative;
    overflow: hidden;
}

.hero-top {
    display: flex;
    align-items: center;
    gap: 14px;
}

.logo {
    width: 54px;
    height: 54px;
    border-radius: 16px;
    background: rgba(255,255,255,.18);
    display: flex;
    align-items: center;
    justify-content: center;
}

.hero h1 {
    margin: 0;
    font-size: 1.9rem;
    font-weight: 800;
    color: white;
}

.hero p {
    margin: 4px 0 0;
    color: white;
}

.chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 16px;
}

.chip {
    background: rgba(255,255,255,.17);
    border: 1px solid rgba(255,255,255,.28);
    color: white;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: .8rem;
    font-weight: 600;
}

.card {
    background: rgba(46,125,50,.07);
    border: 1px solid rgba(46,125,50,.25);
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 14px;
}

.plant-name {
    font-size: 1.6rem;
    font-weight: 800;
}

.sci {
    font-style: italic;
    opacity: .8;
    margin-bottom: 10px;
}

.muted {
    opacity: .65;
    font-size: .78rem;
    text-transform: uppercase;
    letter-spacing: .8px;
    font-weight: 600;
}

.badge {
    display: inline-block;
    background: rgba(46,125,50,.16);
    color: #2E7D32;
    border: 1px solid rgba(46,125,50,.35);
    padding: 3px 11px;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 700;
    margin: 0 6px 6px 0;
}

.badge.warn {
    background: rgba(211,47,47,.12);
    color: #D32F2F;
    border-color: rgba(211,47,47,.4);
}

.row-k {
    font-weight: 700;
    font-size: .8rem;
    opacity: .7;
    text-transform: uppercase;
    margin-top: 12px;
}

.row-v {
    font-size: .98rem;
    margin-top: 2px;
}

.notice {
    border-radius: 14px;
    padding: 11px 14px;
    font-size: .9rem;
    margin: 8px 0 14px;
}

.notice.warn {
    background: rgba(249,168,37,.14);
    border: 1px solid rgba(249,168,37,.5);
}

.notice.danger {
    background: rgba(211,47,47,.1);
    border: 1px solid rgba(211,47,47,.45);
}

.conf-wrap {
    margin: 12px 0 4px;
}

.conf-head {
    display: flex;
    justify-content: space-between;
    font-weight: 700;
    font-size: .85rem;
    margin-bottom: 5px;
}

.track {
    height: 12px;
    background: rgba(128,128,128,.22);
    border-radius: 999px;
    overflow: hidden;
}

.fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #66BB6A, #2E7D32);
}

.fill.low {
    background: linear-gradient(90deg, #FFCA28, #F57C00);
}

.bar-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 8px 0;
}

.bar-label {
    flex: 0 0 38%;
    font-size: .85rem;
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.bar-track {
    flex: 1;
    height: 10px;
    background: rgba(128,128,128,.22);
    border-radius: 999px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #81C784, #388E3C);
    border-radius: 999px;
}

.bar-val {
    flex: 0 0 52px;
    text-align: right;
    font-size: .82rem;
    font-weight: 700;
}

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(270px, 1fr));
    gap: 14px;
}

.pc-name {
    font-weight: 800;
    font-size: 1.08rem;
}

.pc-sci {
    font-style: italic;
    opacity: .75;
    font-size: .88rem;
    margin-bottom: 8px;
}

.pc-uses {
    font-size: .88rem;
    margin-top: 6px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    overflow-x: auto;
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    border-radius: 12px;
    padding: 0 16px;
    font-weight: 600;
    white-space: nowrap;
}

.stTabs [aria-selected="true"] {
    background: rgba(46,125,50,.15);
    color: #2E7D32 !important;
}

[data-testid="stForm"] {
    background: rgba(46,125,50,.07);
    border: 1px solid rgba(46,125,50,.25) !important;
    border-radius: 22px;
    padding: 24px;
    margin-top: 18px;
}

.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    border-radius: 12px;
    font-weight: 700;
    padding: .55rem 1.1rem;
    min-height: 46px;
}

button[kind^="primary"] {
    background: linear-gradient(135deg, #2E7D32, #43A047) !important;
    border: none !important;
    color: white !important;
}

.stFormSubmitButton > button {
    width: 100%;
}

.stTextInput input {
    border-radius: 12px;
    min-height: 46px;
}

[data-testid="stFileUploader"] section {
    border-radius: 16px;
    border: 2px dashed rgba(46,125,50,.5);
}

.login-title {
    font-size: 1.5rem;
    font-weight: 800;
}

.login-sub {
    opacity: .7;
    margin-bottom: 8px;
    font-size: .92rem;
}

.nan-footer {
    text-align: center;
    margin-top: 30px;
    padding: 18px 8px;
    border-top: 1px solid rgba(128,128,128,.25);
}

.nan-footer a {
    color: #2E7D32;
    font-weight: 800;
    text-decoration: none;
}

@media (max-width: 640px) {
    .block-container {
        padding: .7rem .75rem 1rem;
    }

    .hero {
        padding: 18px 16px;
    }

    .hero h1 {
        font-size: 1.4rem;
    }

    .plant-name {
        font-size: 1.35rem;
    }

    .bar-label {
        flex-basis: 44%;
    }
}

</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ======================= Logo =======================

LOGO = (
    '<svg width="30" height="30" viewBox="0 0 24 24" '
    'fill="none" stroke="#fff" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>'
    '<path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>'
    '</svg>'
)


# ======================= Common Functions =======================

def footer():
    st.markdown(
        '<div class="nan-footer">'
        'Developed by <a href="' + NAN_URL + '" target="_blank">'
        'NanStack.tech</a>'
        '<div>Digital Herbarium · ' + str(datetime.now().year) + '</div>'
        '</div>',
        unsafe_allow_html=True
    )


def hero(subtitle, chips):

    chip_html = "".join(
        '<span class="chip">' + html.escape(c) + '</span>'
        for c in chips
    )

    st.markdown(
        '<div class="hero">'
        '<div class="hero-top">'
        '<div class="logo">' + LOGO + '</div>'
        '<div>'
        '<h1>Digital Herbarium</h1>'
        '<p>' + html.escape(subtitle) + '</p>'
        '</div>'
        '</div>'
        '<div class="chips">' + chip_html + '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ======================= Authentication =======================

def check_login(u, p):

    ok_u = hmac.compare_digest(
        u.strip().lower().encode(),
        APP_USER.lower().encode()
    )

    ok_p = hmac.compare_digest(
        p.strip().encode(),
        APP_PASS.encode()
    )

    return ok_u and ok_p


def login_screen():

    hero(
        "Identify plant species from herbarium sheet images",
        ["AI Powered", "10 Species", "Herbarium Sheets"]
    )

    _, mid, _ = st.columns([1, 1.5, 1])

    with mid:

        with st.form("login_form"):

            st.markdown(
                '<div class="login-title">Welcome back 👋</div>'
                '<div class="login-sub">'
                'Sign in or continue as a guest'
                '</div>',
                unsafe_allow_html=True
            )

            u = st.text_input(
                "Username",
                placeholder="Enter username"
            )

            p = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password"
            )

            go = st.form_submit_button(
                "Admin Login →",
                type="primary"
            )

        lock = st.session_state.get("lock_until", 0)

        if go:

            if time.time() < lock:

                st.error(
                    "Too many failed attempts. Please wait and try again."
                )

            elif check_login(u, p):

                st.session_state.auth = True
                st.session_state.role = "admin"
                st.session_state.fails = 0

                st.rerun()

            else:

                st.session_state.fails = (
                    st.session_state.get("fails", 0) + 1
                )

                if st.session_state.fails >= 5:

                    st.session_state.lock_until = time.time() + 30
                    st.session_state.fails = 0

                st.error("Incorrect username or password.")

        st.markdown(
            "<div style='text-align:center; margin:20px 0;'>"
            "──────── OR ────────"
            "</div>",
            unsafe_allow_html=True
        )

        if st.button(
            "👤 Continue as Guest",
            use_container_width=True
        ):

            st.session_state.auth = True
            st.session_state.role = "guest"

            st.rerun()

    footer()


# ======================= Authentication Check =======================

if not st.session_state.get("auth", False):

    login_screen()
    st.stop()


# ======================= Model Loading =======================

@st.cache_resource(show_spinner="Loading model...")
def load_model(path):

    return tf.keras.models.load_model(
        path,
        compile=False
    )


def predict(img, model):

    h, w = model.input_shape[1:3]

    arr = np.asarray(
        img.convert("RGB"),
        dtype="float32"
    )[None]

    arr = tf.image.resize(
        arr,
        (h, w)
    ).numpy()

    return model.predict(
        arr,
        verbose=0
    )[0]


if not os.path.exists(MODEL_PATH):

    st.error(
        f"'{MODEL_PATH}' not found. "
        "Place herbarium_model.keras in the same folder."
    )

    st.stop()


try:

    model = load_model(MODEL_PATH)

except Exception as e:

    st.error(
        "Could not load the model. "
        "Check your TensorFlow version."
    )

    st.code(str(e)[:1500])
    st.stop()


if model.output_shape[-1] != len(NAMES):

    st.error(
        f"The model has {model.output_shape[-1]} classes "
        f"but class_names.json lists {len(NAMES)}."
    )

    st.stop()


# ======================= Session Storage =======================

if "log" not in st.session_state:
    st.session_state.log = []


# ======================= UI Helpers =======================

def result_card(d, conf, thr):

    toxic = "⚠️" in d["uses"]

    uses = html.escape(
        d["uses"].replace("⚠️", "").strip()
    )

    low = conf < thr

    s = (
        '<div class="card">'
        '<div class="muted">'
        + (
            "Best guess (low confidence)"
            if low else "Identified species"
        )
        + '</div>'
        '<div class="plant-name">'
        + html.escape(d["common"])
        + '</div>'
        '<div class="sci">'
        + html.escape(d["scientific"])
        + '</div>'
        '<span class="badge">'
        + html.escape(d["family"])
        + '</span>'
    )

    if toxic:

        s += '<span class="badge warn">⚠ Caution</span>'

    s += (
        '<div class="conf-wrap">'
        '<div class="conf-head">'
        '<span>Confidence</span>'
        '<span>' + f"{conf:.1f}%" + '</span>'
        '</div>'
        '<div class="track">'
        '<div class="fill'
        + (" low" if low else "")
        + '" style="width:'
        + f"{min(conf, 100):.1f}"
        + '%"></div>'
        '</div>'
        '</div>'
        '<div class="row-k">Distribution</div>'
        '<div class="row-v">'
        + html.escape(d["habitat"])
        + '</div>'
        '<div class="row-k">Uses and notes</div>'
        '<div class="row-v">'
        + uses
        + '</div>'
        '</div>'
    )

    return s


def top_bars(top, probs):

    s = (
        '<div class="card">'
        '<div class="muted">Top 3 predictions</div>'
    )

    for i in top:

        pct = float(probs[int(i)]) * 100

        s += (
            '<div class="bar-row">'
            '<div class="bar-label">'
            + html.escape(NAMES[int(i)])
            + '</div>'
            '<div class="bar-track">'
            '<div class="bar-fill" style="width:'
            + f"{pct:.1f}"
            + '%"></div>'
            '</div>'
            '<div class="bar-val">'
            + f"{pct:.1f}%"
            + '</div>'
            '</div>'
        )

    return s + '</div>'


def plant_grid(items):

    s = '<div class="grid">'

    for d in items:

        toxic = "⚠️" in d["uses"]

        s += (
            '<div class="card">'
            '<div class="pc-name">'
            + html.escape(d["common"])
            + '</div>'
            '<div class="pc-sci">'
            + html.escape(d["scientific"])
            + '</div>'
            '<span class="badge">'
            + html.escape(d["family"])
            + '</span>'
            + (
                '<span class="badge warn">⚠ Caution</span>'
                if toxic else ''
            )
            + '<div class="pc-uses"><b>Distribution:</b> '
            + html.escape(d["habitat"])
            + '</div>'
            '<div class="pc-uses"><b>Uses and notes:</b> '
            + html.escape(
                d["uses"].replace("⚠️", "").strip()
            )
            + '</div>'
            '</div>'
        )

    return s + '</div>'


# ======================= Main Page =======================

role = st.session_state.get("role", "guest")

display_user = (
    "Admin"
    if role == "admin"
    else "Guest User"
)

hero(
    "Identify plant species from digitised herbarium sheets",
    [
        "👤 " + display_user,
        str(len(NAMES)) + " Species",
        "MobileNetV2",
        "Herbarium Sheets"
    ]
)


# ======================= Logout =======================

if st.button("🚪 Logout", key="logout_btn"):

    st.session_state.auth = False
    st.session_state.pop("role", None)
    st.session_state.log = []

    st.rerun()


# ======================= Navigation Tabs =======================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🔍 Identify",
        "📚 My Herbarium",
        "🌱 Species",
        "📊 Model",
        "ℹ️ About"
    ]
)


# ======================= Identify =======================

with tab1:

    left, right = st.columns(2)

    with left:

        file = st.file_uploader(
            "Upload a herbarium sheet image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp",
                "tif",
                "tiff"
            ]
        )

        img = (
            ImageOps.exif_transpose(Image.open(file))
            if file else None
        )

        with st.expander("⚙️ Settings"):

            threshold = st.slider(
                "Low-confidence warning (%)",
                10,
                95,
                60
            )

        if img is not None:

            st.image(
                img,
                caption="Your image"
            )

        st.caption(
            "Tip: Use a full herbarium sheet image. "
            "Photos of living plants are not supported."
        )

    with right:

        if img is None:

            st.markdown(
                '<div class="card">'
                '<div class="plant-name">'
                'Waiting for an image 🌿'
                '</div>'
                '<div class="row-v">'
                'Upload an image and the app will identify '
                'the plant species.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            probs = predict(img, model)

            top = np.argsort(probs)[::-1][:3]

            idx = int(top[0])

            conf = float(probs[idx]) * 100

            d = INFO[idx]

            st.markdown(
                result_card(d, conf, threshold),
                unsafe_allow_html=True
            )

            if "⚠️" in d["uses"]:

                st.markdown(
                    '<div class="notice danger">'
                    '⚠️ Handle this plant with care.'
                    '</div>',
                    unsafe_allow_html=True
                )

            if conf < threshold:

                st.markdown(
                    '<div class="notice warn">'
                    'Low confidence. This may be a species '
                    'outside the supported list.'
                    '</div>',
                    unsafe_allow_html=True
                )

            st.markdown(
                top_bars(top, probs),
                unsafe_allow_html=True
            )

            if st.button(
                "➕ Save to herbarium",
                type="primary"
            ):

                thumb = img.convert("RGB")

                thumb.thumbnail((220, 220))

                st.session_state.log.append(
                    {
                        "Date": datetime.now().strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                        "Species": d["scientific"],
                        "Common name": d["common"],
                        "Family": d["family"],
                        "Confidence": f"{conf:.1f}%",
                        "thumb": thumb
                    }
                )

                st.success(
                    "Saved! Check the My Herbarium tab."
                )

    with st.expander(
        "🌱 Supported species (" + str(len(NAMES)) + ")"
    ):

        st.markdown(
            "".join(
                '<span class="badge">'
                + html.escape(n)
                + '</span>'
                for n in NAMES
            ),
            unsafe_allow_html=True
        )

        st.caption(
            "The model can only recognise these species. "
            "Other plants may be matched to the closest supported species."
        )


# ======================= My Herbarium =======================

with tab2:

    log = st.session_state.log

    if not log:

        st.markdown(
            '<div class="card">'
            '<div class="plant-name">'
            'Your herbarium is empty'
            '</div>'
            '<div class="row-v">'
            'Upload an image in the Identify tab and press '
            'Save to herbarium.'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="muted">'
            + str(len(log))
            + ' entries</div>',
            unsafe_allow_html=True
        )

        cols = st.columns(4)

        for i, e in enumerate(log):

            with cols[i % 4]:

                st.image(
                    e["thumb"],
                    caption=(
                        e["Species"]
                        + " ("
                        + e["Confidence"]
                        + ")"
                    )
                )

        df = pd.DataFrame(
            [
                {
                    k: v
                    for k, v in e.items()
                    if k != "thumb"
                }
                for e in log
            ]
        )

        st.dataframe(
            df,
            hide_index=True
        )

        c1, c2 = st.columns(2)

        with c1:

            st.download_button(
                "⬇️ CSV download",
                df.to_csv(index=False).encode("utf-8"),
                file_name="my_herbarium.csv",
                mime="text/csv"
            )

        with c2:

            if st.button("🗑️ Clear herbarium"):

                st.session_state.log = []

                st.rerun()


# ======================= Species =======================

with tab3:

    q = st.text_input(
        "🔎 Search",
        placeholder="Type a name, family or use"
    )

    fams = st.multiselect(
        "Family filter",
        sorted({d["family"] for d in INFO})
    )

    items = INFO

    if q:

        items = [
            d
            for d in items
            if q.lower() in " ".join(d.values()).lower()
        ]

    if fams:

        items = [
            d
            for d in items
            if d["family"] in fams
        ]

    st.markdown(
        '<div class="muted" style="margin:6px 0 10px 0;">'
        + str(len(items))
        + ' species</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        plant_grid(items)
        if items
        else '<div class="card">No species found.</div>',
        unsafe_allow_html=True
    )

    st.caption("⚠️ " + DISCLAIMER)


# ======================= Model =======================

with tab4:

    st.markdown(
        '<div class="card">'
        '<div class="plant-name">Model details</div>'
        '<div class="row-v">'
        'MobileNetV2 (ImageNet pretrained) with a new classifier head, '
        'trained on herbarium sheet images from GBIF. '
        'The top layer was trained first, then the last layers '
        'were fine-tuned. Preprocessing is built into the model.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    if not METRICS:

        st.info(
            "Place metrics.json in the app folder to see "
            "accuracy, reports and confusion matrix."
        )

    else:

        rep = METRICS.get("report", {})

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Test accuracy",
            f"{METRICS.get('test_acc', 0) * 100:.1f}%"
        )

        c2.metric(
            "Validation accuracy",
            f"{METRICS.get('val_acc', 0) * 100:.1f}%"
        )

        if "macro avg" in rep:

            c3.metric(
                "Macro F1",
                f"{rep['macro avg']['f1-score']:.3f}"
            )

        st.caption(
            f"Test images: {METRICS.get('n_test', '?')} | "
            f"Training images: {METRICS.get('n_train', '?')} | "
            f"Trained: {METRICS.get('created', '?')}"
        )

        hist = METRICS.get("history")

        if hist:

            fig, ax = plt.subplots(
                1,
                2,
                figsize=(11, 3.4)
            )

            for a, (k1, k2, t) in zip(
                ax,
                [
                    ("accuracy", "val_accuracy", "Accuracy"),
                    ("loss", "val_loss", "Loss")
                ]
            ):

                a.plot(hist[k1], label="Train")
                a.plot(hist[k2], label="Validation")
                a.set_title(t)
                a.legend()

            fig.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        if rep:

            st.subheader(
                "Per-species report (test set)"
            )

            rows = [
                {
                    "Species": n,
                    "Precision": rep[n]["precision"],
                    "Recall": rep[n]["recall"],
                    "F1": rep[n]["f1-score"],
                    "Images": int(rep[n]["support"])
                }
                for n in NAMES
                if n in rep
            ]

            st.dataframe(
                pd.DataFrame(rows).round(3),
                hide_index=True
            )

        cm = METRICS.get("cm")

        if cm:

            st.subheader(
                "Confusion matrix (test set)"
            )

            cm = np.array(cm)

            fig, ax = plt.subplots(
                figsize=(7.5, 6.5)
            )

            im = ax.imshow(
                cm,
                cmap="Greens"
            )

            ax.set_xticks(range(len(NAMES)))
            ax.set_yticks(range(len(NAMES)))

            ax.set_xticklabels(
                NAMES,
                rotation=60,
                ha="right",
                fontsize=8
            )

            ax.set_yticklabels(
                NAMES,
                fontsize=8
            )

            for i in range(cm.shape[0]):

                for j in range(cm.shape[1]):

                    if cm[i, j]:

                        ax.text(
                            j,
                            i,
                            str(cm[i, j]),
                            ha="center",
                            va="center",
                            fontsize=8,
                            color=(
                                "white"
                                if cm[i, j] > cm.max() / 2
                                else "black"
                            )
                        )

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            fig.colorbar(im)

            fig.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


# ======================= About =======================

with tab5:

    st.markdown(
        '<div class="card">'
        '<div class="plant-name">'
        'What is a Digital Herbarium?'
        '</div>'
        '<div class="row-v">'
        'A herbarium is a collection of pressed and preserved '
        'plant specimens mounted on sheets. This app identifies '
        'the species shown on a digitised herbarium sheet.'
        '</div>'
        '</div>'

        '<div class="grid">'

        '<div class="card">'
        '<b>1. Upload</b>'
        '<div class="row-v">Upload an image of a herbarium sheet.</div>'
        '</div>'

        '<div class="card">'
        '<b>2. AI Identification</b>'
        '<div class="row-v">A MobileNetV2 model matches it against '
        + str(len(NAMES))
        + ' species.</div>'
        '</div>'

        '<div class="card">'
        '<b>3. Herbarium Entry</b>'
        '<div class="row-v">View details and save the identification.</div>'
        '</div>'

        '</div>'

        '<div class="card" style="margin-top:14px;">'
        '<div class="muted">Developed by</div>'
        '<div class="plant-name">NanStack</div>'
        '<div class="row-v">'
        'Web · Mobile · AI · SaaS. '
        'Turning concepts into digital products.'
        '</div>'
        '<a class="btn-link" href="' + NAN_URL + '" target="_blank">'
        'Visit NanStack.tech</a>'
        '<a class="btn-link alt" href="' + NAN_CONTACT + '" target="_blank">'
        'Contact</a>'
        '</div>'

        '<div class="notice warn">'
        '⚠️ ' + DISCLAIMER
        + ' Data: specimen images from GBIF (gbif.org). '
        'Image licences vary by contributing herbarium. '
        'The model works on herbarium sheet images only.'
        '</div>',

        unsafe_allow_html=True
    )


# ======================= Footer =======================

footer()
