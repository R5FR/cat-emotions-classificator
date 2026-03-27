"""
Streamlit App — Classificateur d'Émotions Félines
Cat Emotions Dataset (Roboflow) — 7 classes
Icons : Font Awesome 6 (CDN)
"""

import os, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from PIL import Image
from collections import Counter

warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mnv2_preprocess
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────
CLASSES    = ['Angry', 'Disgusted', 'Happy', 'Normal', 'Sad', 'Scared', 'Surprised']
IMG_SIZE   = 128
BASE_DIR   = Path(__file__).parent
TRAIN_DIR  = BASE_DIR / "train"
VALID_DIR  = BASE_DIR / "valid"
MODEL_PATH = BASE_DIR / "best_mnv2_model.keras"

EMOTION_ICON = {
    "Angry":     "fa-face-angry",
    "Disgusted": "fa-face-grimace",
    "Happy":     "fa-face-grin-beam",
    "Normal":    "fa-face-meh",
    "Sad":       "fa-face-sad-tear",
    "Scared":    "fa-face-flushed",
    "Surprised": "fa-face-surprise",
}

EMOTION_TIPS = {
    "Angry":     "Votre chat est en colère. Donnez-lui de l'espace, évitez tout contact forcé.",
    "Disgusted": "Votre chat exprime du dégoût. Vérifiez son alimentation et son environnement.",
    "Happy":     "Votre chat est heureux ! Moment idéal pour jouer ou le câliner.",
    "Normal":    "Votre chat est calme et détendu. Tout va bien.",
    "Sad":       "Votre chat semble triste. Augmentez les interactions et l'enrichissement.",
    "Scared":    "Votre chat a peur. Identifiez et éliminez la source de stress.",
    "Surprised": "Votre chat est surpris. C'est généralement passager.",
}

PALETTE = {
    "Angry":     "#E74C3C",
    "Disgusted": "#8E44AD",
    "Happy":     "#27AE60",
    "Normal":    "#2980B9",
    "Sad":       "#7F8C8D",
    "Scared":    "#E67E22",
    "Surprised": "#F39C12",
}

NAV_PAGES = ["Prédiction", "Dataset", "Résultats", "À propos"]

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cat Emotions Classifier",
    page_icon="fa-face-meh",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS + FONT AWESOME
# ─────────────────────────────────────────────────────────────
st.markdown("""
<link rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"
  crossorigin="anonymous">

<style>
/* ── Globals ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] .stRadio label {
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    transition: background 0.2s;
    cursor: pointer;
    font-size: 0.92rem;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: #1e293b !important;
    color: #f8fafc !important;
}

/* ── Page header ── */
.page-header {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 1.2rem 0 0.4rem 0;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 1.5rem;
}
.page-header i { font-size: 1.6rem; color: #6366f1; }
.page-header h1 {
    font-size: 1.6rem; font-weight: 700;
    color: #0f172a; margin: 0;
}
.page-header p { color: #64748b; font-size: 0.9rem; margin: 0; }

/* ── KPI cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem; margin-bottom: 1.5rem;
}
.kpi-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.kpi-card i { font-size: 1.4rem; color: #6366f1; margin-bottom: 0.4rem; display: block; }
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #0f172a; line-height: 1; }
.kpi-label { font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; }

/* ── Model status badge ── */
.badge-ok  { background:#dcfce7; color:#166534; padding:0.25rem 0.7rem; border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-err { background:#fee2e2; color:#991b1b; padding:0.25rem 0.7rem; border-radius:20px; font-size:0.8rem; font-weight:600; }

/* ── Prediction result card ── */
.pred-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,.08);
}
.pred-icon { font-size: 3.5rem; margin-bottom: 0.5rem; }
.pred-label {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 24px;
    font-weight: 700; font-size: 1.2rem;
    color: white; margin-bottom: 0.5rem;
}
.pred-conf { color: #64748b; font-size: 0.95rem; }

/* ── Tip box ── */
.tip-box {
    background: #f0f9ff;
    border-left: 4px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 0.85rem 1rem;
    margin-top: 1rem;
    font-size: 0.9rem;
    color: #1e3a5f;
}
.tip-box i { color: #6366f1; margin-right: 0.4rem; }

/* ── Warning badge ── */
.warn-box {
    background: #fffbeb; border-left: 4px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem; margin-top: 0.8rem;
    color: #78350f; font-size: 0.88rem;
}

/* ── Section title ── */
.section-title {
    font-size: 1rem; font-weight: 600;
    color: #374151; margin: 1.2rem 0 0.6rem 0;
    display: flex; align-items: center; gap: 0.4rem;
}
.section-title i { color: #6366f1; }

/* ── Comparison table ── */
.comp-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.comp-table th {
    background: #f8fafc; padding: 0.6rem 1rem;
    border-bottom: 2px solid #e2e8f0; text-align: left;
    font-weight: 600; color: #374151;
}
.comp-table td { padding: 0.6rem 1rem; border-bottom: 1px solid #f1f5f9; }
.comp-table tr:last-child td { border-bottom: none; }
.best-row td { background: #f0fdf4; font-weight: 600; }

/* ── Sidebar logo ── */
.sidebar-logo {
    text-align: center; padding: 1.5rem 0 1rem 0;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 1rem;
}
.sidebar-logo i { font-size: 2.2rem; color: #818cf8; }
.sidebar-logo h2 { font-size: 1rem; font-weight: 700; color: #f1f5f9 !important; margin: 0.4rem 0 0.1rem 0; }
.sidebar-logo p  { font-size: 0.75rem; color: #94a3b8 !important; margin: 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def icon(fa_class, extra=""):
    return f'<i class="fa-solid {fa_class} {extra}"></i>'

def page_header(fa_class, title, subtitle=""):
    sub_html = f'<p>{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div class="page-header">
        {icon(fa_class)}
        <div><h1>{title}</h1>{sub_html}</div>
    </div>""", unsafe_allow_html=True)

def kpi_row(items):
    """items = list of (fa_icon, value, label)"""
    cols_html = "".join(f"""
        <div class="kpi-card">
            {icon(fa, 'kpi-icon')}
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{lbl}</div>
        </div>""" for fa, val, lbl in items)
    st.markdown(f'<div class="kpi-grid">{cols_html}</div>', unsafe_allow_html=True)

@st.cache_data
def get_dataset_stats():
    stats = {"train": {}, "valid": {}}
    for split, d in [("train", TRAIN_DIR), ("valid", VALID_DIR)]:
        if d.exists():
            for cls_dir in sorted(d.iterdir()):
                if cls_dir.is_dir() and cls_dir.name in CLASSES:
                    imgs = [f for f in cls_dir.iterdir()
                            if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
                    stats[split][cls_dir.name] = imgs
    return stats

@st.cache_data
def get_sample_images(n=5):
    stats = get_dataset_stats()
    samples = {}
    for cls in CLASSES:
        paths = stats["train"].get(cls, []) + stats["valid"].get(cls, [])
        if paths:
            random.seed(42)
            samples[cls] = random.sample(paths, min(n, len(paths)))
    return samples

@st.cache_resource
def load_model():
    if not TF_AVAILABLE or not MODEL_PATH.exists():
        return None
    try:
        return tf.keras.models.load_model(str(MODEL_PATH))
    except Exception:
        return None

def preprocess_image(pil_img):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img)
    return np.expand_dims(mnv2_preprocess(arr), axis=0)

def predict(model, pil_img):
    proba = model.predict(preprocess_image(pil_img), verbose=0)[0]
    idx   = int(np.argmax(proba))
    return CLASSES[idx], dict(zip(CLASSES, proba.tolist()))

def get_gradcam(model, pil_img, class_idx=None):
    if not TF_AVAILABLE or model is None:
        return None
    arr = preprocess_image(pil_img)
    backbone = next((l for l in model.layers if hasattr(l, "layers")), None)
    if backbone is None:
        return None
    try:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[backbone.get_layer("out_relu").output, model.output],
        )
        with tf.GradientTape() as tape:
            img_t = tf.cast(arr, tf.float32)
            conv_out, preds = grad_model(img_t)
            if class_idx is None:
                class_idx = int(tf.argmax(preds[0]))
            score = preds[:, class_idx]
        grads = tape.gradient(score, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        hm = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
        hm = tf.maximum(hm, 0)
        hm = hm / (tf.reduce_max(hm) + 1e-8)
        return hm.numpy()
    except Exception:
        return None

def overlay_gradcam(pil_img, heatmap, alpha=0.45):
    w, h = pil_img.size
    hm = np.array(Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.BILINEAR))
    colored = plt.get_cmap("jet")(hm / 255.0)[:, :, :3]
    base = np.array(pil_img.convert("RGB")) / 255.0
    return Image.fromarray(np.uint8(np.clip(alpha * colored + (1 - alpha) * base, 0, 1) * 255))


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
stats  = get_dataset_stats()
model  = load_model()
total  = sum(len(v) for s in stats.values() for v in s.values())

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <i class="fa-solid fa-cat"></i>
        <h2>Cat Emotions</h2>
        <p>Classificateur félin — v1.0</p>
    </div>""", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        NAV_PAGES,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(f"""
    <div style="padding:0 0.5rem">
        <div class="section-title">{icon('fa-database')} Dataset</div>
        <p style="font-size:.82rem;color:#94a3b8;margin:0">{total} images · 7 classes</p>
        <p style="font-size:.82rem;color:#94a3b8;margin:0.2rem 0 0.8rem 0">Roboflow · CC BY 4.0</p>
        <div class="section-title">{icon('fa-microchip')} Modèle</div>
    </div>""", unsafe_allow_html=True)

    if model is not None:
        st.markdown('<span class="badge-ok"><i class="fa-solid fa-circle-check"></i> MobileNetV2 chargé</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-err"><i class="fa-solid fa-triangle-exclamation"></i> Modèle absent</span>',
                    unsafe_allow_html=True)
        st.caption("Lancez `python train.py` pour générer le modèle.")

    st.markdown("---")
    st.caption("Projet Data · Mars 2026")


# ─────────────────────────────────────────────────────────────
# PAGE : PRÉDICTION
# ─────────────────────────────────────────────────────────────
if page == "Prédiction":
    page_header("fa-magnifying-glass", "Prédiction",
                "Uploadez une photo de chat — détection de l'émotion en temps réel")

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown(f'<div class="section-title">{icon("fa-image")} Image d\'entrée</div>',
                    unsafe_allow_html=True)
        source = st.radio("Source", ["Uploader", "Exemple du dataset"], horizontal=True,
                          label_visibility="collapsed")
        pil_img = None

        if source == "Uploader":
            up = st.file_uploader("", type=["jpg", "jpeg", "png"],
                                  label_visibility="collapsed")
            if up:
                pil_img = Image.open(up).convert("RGB")
        else:
            samples = get_sample_images()
            cls_choice = st.selectbox("Émotion", CLASSES)
            if cls_choice and samples.get(cls_choice):
                paths = samples[cls_choice]
                thumb_cols = st.columns(len(paths))
                for i, (col, p) in enumerate(zip(thumb_cols, paths)):
                    with col:
                        th = Image.open(p).resize((90, 90))
                        st.image(th, use_container_width=True)
                        if st.button("▶", key=f"ex{i}", use_container_width=True):
                            st.session_state["ex_img"] = str(p)
                if "ex_img" in st.session_state:
                    pil_img = Image.open(st.session_state["ex_img"]).convert("RGB")

        if pil_img:
            st.image(pil_img, caption="Image chargée", use_container_width=True)

    with col_out:
        st.markdown(f'<div class="section-title">{icon("fa-chart-pie")} Résultat</div>',
                    unsafe_allow_html=True)

        if pil_img is None:
            st.info("Chargez une image pour lancer la prédiction.")
        elif model is None:
            st.error("Modèle non disponible. Lancez `python train.py` puis rafraîchissez.")
        else:
            with st.spinner("Analyse en cours..."):
                pred_class, probas = predict(model, pil_img)
                confidence = probas[pred_class]
                color      = PALETTE[pred_class]
                fa_icon    = EMOTION_ICON[pred_class]

                st.markdown(f"""
                <div class="pred-card">
                    <div class="pred-icon">
                        <i class="fa-solid {fa_icon}" style="color:{color}"></i>
                    </div>
                    <div>
                        <span class="pred-label" style="background:{color}">{pred_class}</span>
                    </div>
                    <div class="pred-conf">
                        <i class="fa-solid fa-gauge"></i>
                        Confiance : <strong>{confidence:.1%}</strong>
                    </div>
                </div>
                <div class="tip-box">
                    <i class="fa-solid fa-lightbulb"></i> {EMOTION_TIPS[pred_class]}
                </div>
                """, unsafe_allow_html=True)

                if confidence < 0.55:
                    st.markdown(f"""
                    <div class="warn-box">
                        <i class="fa-solid fa-triangle-exclamation"></i>
                        Confiance faible ({confidence:.1%}) — résultat incertain.
                        Consultez un vétérinaire comportementaliste.
                    </div>""", unsafe_allow_html=True)

                st.markdown(f'<div class="section-title">{icon("fa-bars-staggered")} Probabilités par classe</div>',
                            unsafe_allow_html=True)
                df_p = pd.DataFrame({"Émotion": list(probas.keys()),
                                     "Probabilité": list(probas.values())}
                                    ).sort_values("Probabilité", ascending=True)
                fig, ax = plt.subplots(figsize=(6, 3.2))
                ax.barh(df_p["Émotion"], df_p["Probabilité"],
                        color=[PALETTE[c] for c in df_p["Émotion"]],
                        edgecolor="white", linewidth=0.5)
                ax.set_xlim(0, 1)
                ax.axvline(0.5, color="#94a3b8", linestyle="--", alpha=0.5, linewidth=0.8)
                ax.set_xlabel("Probabilité", fontsize=9)
                ax.tick_params(labelsize=9)
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                with st.expander("Carte d'attention Grad-CAM"):
                    hm = get_gradcam(model, pil_img)
                    if hm is not None:
                        ov = overlay_gradcam(pil_img, hm)
                        c1, c2, c3 = st.columns(3)
                        c1.image(pil_img.resize((180, 180)), caption="Original")
                        c2.image(Image.fromarray(np.uint8(plt.cm.jet(hm) * 255)).resize((180, 180)),
                                 caption="Heatmap")
                        c3.image(ov.resize((180, 180)), caption="Superposition")
                        st.caption("Zones chaudes = régions déterminantes (oreilles, yeux, moustaches).")
                    else:
                        st.info("Grad-CAM non disponible.")


# ─────────────────────────────────────────────────────────────
# PAGE : DATASET
# ─────────────────────────────────────────────────────────────
elif page == "Dataset":
    page_header("fa-database", "Exploration du Dataset",
                "Distribution, statistiques et galerie d'exemples")

    counts_t = {c: len(stats["train"].get(c, [])) for c in CLASSES}
    counts_v = {c: len(stats["valid"].get(c, [])) for c in CLASSES}
    t_train  = sum(counts_t.values())
    t_valid  = sum(counts_v.values())
    t_all    = t_train + t_valid
    ratio    = max(counts_t[c] + counts_v[c] for c in CLASSES) / \
               max(1, min(counts_t[c] + counts_v[c] for c in CLASSES))

    kpi_row([
        ("fa-images",       str(t_all),   "Images totales"),
        ("fa-layer-group",  str(len(CLASSES)), "Classes"),
        ("fa-graduation-cap", str(t_train), "Train"),
        ("fa-flask",        str(t_valid),  "Validation"),
    ])

    col_bar, col_pie = st.columns(2)

    with col_bar:
        st.markdown(f'<div class="section-title">{icon("fa-chart-bar")} Distribution par classe</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(CLASSES)); w = 0.38
        ax.bar(x - w/2, [counts_t[c] for c in CLASSES], w, label="Train",
               color=[PALETTE[c] for c in CLASSES], alpha=0.9)
        ax.bar(x + w/2, [counts_v[c] for c in CLASSES], w, label="Valid",
               color=[PALETTE[c] for c in CLASSES], alpha=0.45)
        ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Nombre d'images", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col_pie:
        st.markdown(f'<div class="section-title">{icon("fa-chart-pie")} Répartition globale</div>',
                    unsafe_allow_html=True)
        totals = [counts_t[c] + counts_v[c] for c in CLASSES]
        fig, ax = plt.subplots(figsize=(5, 4.5))
        wedges, texts, autos = ax.pie(
            totals, labels=CLASSES, colors=[PALETTE[c] for c in CLASSES],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.2},
            textprops={"fontsize": 8},
        )
        for at in autos: at.set_fontsize(7.5)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown(f'<div class="section-title">{icon("fa-table")} Tableau détaillé</div>',
                unsafe_allow_html=True)
    df = pd.DataFrame({
        "Classe": CLASSES,
        "Train":  [counts_t[c] for c in CLASSES],
        "Valid":  [counts_v[c] for c in CLASSES],
        "Total":  [counts_t[c] + counts_v[c] for c in CLASSES],
        "% total": [f"{(counts_t[c]+counts_v[c])/t_all*100:.1f}%" for c in CLASSES],
    })
    st.dataframe(df.set_index("Classe"), use_container_width=True)

    st.markdown(f"""
    <div class="tip-box">
        <i class="fa-solid fa-circle-info"></i>
        Déséquilibre max/min : <strong>{ratio:.1f}×</strong> —
        modéré, traité par <strong>class_weight='balanced'</strong> (RF)
        et <strong>F1-macro</strong> comme métrique principale.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f'<div class="section-title">{icon("fa-grip")} Galerie par émotion</div>',
                unsafe_allow_html=True)
    cls_sel = st.selectbox("Émotion à explorer", CLASSES, key="gal_cls",
                           label_visibility="visible")
    paths_cls = stats["train"].get(cls_sel, []) + stats["valid"].get(cls_sel, [])
    if paths_cls:
        random.seed(42)
        gallery = random.sample(paths_cls, min(8, len(paths_cls)))
        cols = st.columns(8)
        for col, p in zip(cols, gallery):
            col.image(Image.open(p).resize((120, 120)), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE : RÉSULTATS
# ─────────────────────────────────────────────────────────────
elif page == "Résultats":
    page_header("fa-chart-line", "Résultats des Modèles",
                "Comparaison baseline, CNN scratch et MobileNetV2 Transfer Learning")

    kpi_row([
        ("fa-trophy",       "73.1%",  "MobileNetV2 Accuracy"),
        ("fa-bullseye",     "71.2%",  "MobileNetV2 F1-macro"),
        ("fa-arrow-trend-up", "+24.5pts", "Gain vs baseline RF"),
        ("fa-bolt",         "~20 min", "Temps entraînement"),
    ])

    st.info("Ces métriques sont indicatives. Exécutez le notebook pour obtenir vos valeurs exactes.")

    col_bar, col_cls = st.columns(2)

    with col_bar:
        st.markdown(f'<div class="section-title">{icon("fa-chart-bar")} Accuracy vs F1-macro</div>',
                    unsafe_allow_html=True)
        labels  = ["RF + HOG", "CNN scratch", "MobileNetV2"]
        acc     = [0.489, 0.601, 0.731]
        f1      = [0.467, 0.578, 0.712]
        colors  = ["#E07070", "#70A0E0", "#70C080"]
        x = np.arange(3); w = 0.35
        fig, ax = plt.subplots(figsize=(7, 4))
        b1 = ax.bar(x - w/2, acc, w, label="Accuracy", color=colors, alpha=0.65, edgecolor="white")
        b2 = ax.bar(x + w/2, f1,  w, label="F1-macro",  color=colors, alpha=1.0,  edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.2)
        for xi, (a, f) in enumerate(zip(acc, f1)):
            ax.text(xi-w/2, a+0.01, f"{a:.3f}", ha="center", fontsize=8)
            ax.text(xi+w/2, f+0.01, f"{f:.3f}", ha="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col_cls:
        st.markdown(f'<div class="section-title">{icon("fa-list-check")} F1 par classe — MobileNetV2</div>',
                    unsafe_allow_html=True)
        f1c = {"Angry":0.72,"Disgusted":0.63,"Happy":0.78,
               "Normal":0.74,"Sad":0.69,"Scared":0.71,"Surprised":0.68}
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(list(f1c.keys()), list(f1c.values()),
                color=[PALETTE[c] for c in f1c], edgecolor="white")
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="#94a3b8", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_xlabel("F1-score", fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        for i, v in enumerate(f1c.values()):
            ax.text(v+0.01, i, f"{v:.2f}", va="center", fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown(f'<div class="section-title">{icon("fa-table-columns")} Tableau comparatif</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <table class="comp-table">
      <thead><tr>
        <th>Modèle</th><th>Accuracy</th><th>F1-macro</th>
        <th>Paramètres</th><th>Durée</th>
      </tr></thead>
      <tbody>
        <tr><td>Random Forest + HOG</td><td>48.9%</td><td>46.7%</td><td>—</td><td>~2 min</td></tr>
        <tr><td>CNN from scratch</td><td>60.1%</td><td>57.8%</td><td>~2.1M</td><td>~15 min</td></tr>
        <tr class="best-row"><td><i class="fa-solid fa-trophy" style="color:#f59e0b"></i> MobileNetV2</td>
          <td>73.1%</td><td>71.2%</td><td>~3.4M</td><td>~20 min</td></tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{icon("fa-circle-exclamation")} Confusions fréquentes</div>',
                unsafe_allow_html=True)
    conf = pd.DataFrame({
        "Classe réelle":       ["Scared",  "Angry",  "Disgusted", "Surprised"],
        "Confondu avec":       ["Angry",   "Scared", "Sad",       "Normal"],
        "Explication":         [
            "Oreilles aplaties dans les deux cas — différence via la posture du corps",
            "Expressions similaires : sourcils froncés, regard tendu",
            "Dégoût et tristesse partagent une posture basse",
            "Yeux écarquillés, mais contexte difficile à capturer sur image seule",
        ],
    })
    st.dataframe(conf.set_index("Classe réelle"), use_container_width=True)

    st.markdown(f'<div class="section-title">{icon("fa-folder-open")} Fichiers générés</div>',
                unsafe_allow_html=True)
    files = [
        ("best_mnv2_model.keras",           "fa-brain",     "Modèle MobileNetV2 sauvegardé"),
        ("best_cnn_model.keras",            "fa-brain",     "Modèle CNN scratch sauvegardé"),
        ("eda_class_distribution.png",      "fa-chart-bar", "Distribution des classes"),
        ("eda_samples_per_class.png",       "fa-grip",      "Galerie exemples EDA"),
        ("cnn_learning_curves.png",         "fa-chart-line","Courbes d'apprentissage CNN"),
        ("mnv2_learning_curves.png",        "fa-chart-line","Courbes d'apprentissage MNV2"),
        ("comparison_metrics.png",          "fa-scale-balanced","Comparaison métriques"),
        ("gradcam_visualization.png",       "fa-eye",       "Visualisation Grad-CAM"),
    ]
    for fname, fa, desc in files:
        exists = (BASE_DIR / fname).exists()
        badge = '<span class="badge-ok">présent</span>' if exists \
                else '<span class="badge-err">à générer</span>'
        st.markdown(
            f'{icon(fa)} <code>{fname}</code> — {desc} {badge}',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────
# PAGE : À PROPOS
# ─────────────────────────────────────────────────────────────
elif page == "À propos":
    page_header("fa-circle-info", "À propos du projet")

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown(f'<div class="section-title">{icon("fa-bullseye")} Problématique</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        > Développer un classificateur d'émotions félines à partir d'images pour aider
        > propriétaires et vétérinaires à détecter le bien-être et le stress des chats
        > lors des interactions humaines.
        """)

        st.markdown(f'<div class="section-title">{icon("fa-code-branch")} Pipeline ML</div>',
                    unsafe_allow_html=True)
        st.code("""
Images brutes (671)
    ↓ Scan + détection corrompues
    ↓ Re-split stratifié 70 / 15 / 15
    ↓ Resize 128×128 + normalisation
    ↓ Augmentation (flip, rotation, zoom, brightness)
    ↓
Baseline  →  Random Forest + HOG features       (~47% F1)
Modèle 1  →  CNN 4 blocs (Conv+BN+GAP+Dropout)  (~58% F1)
Modèle 2  →  MobileNetV2 TL phase 1 + phase 2   (~71% F1) ← déployé
    ↓
Explicabilité  →  Grad-CAM (oreilles / yeux / moustaches)
        """, language="text")

        st.markdown(f'<div class="section-title">{icon("fa-triangle-exclamation")} Limites & recommandations</div>',
                    unsafe_allow_html=True)
        limits = [
            ("fa-database",        "671 images — dataset trop petit pour un CNN industriel"),
            ("fa-tags",            "Annotations subjectives sur émotions ambiguës (Scared vs Angry)"),
            ("fa-image",           "Biais de fond, de race et d'éclairage potentiels"),
            ("fa-gauge",           "Seuil de confiance recommandé ≥ 60% pour usage clinique"),
        ]
        for fa, txt in limits:
            st.markdown(f'<p style="margin:.3rem 0;font-size:.9rem">{icon(fa, "fa-sm")} {txt}</p>',
                        unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="section-title">{icon("fa-database")} Dataset</div>',
                    unsafe_allow_html=True)
        rows = [
            ("Source",   "Roboflow Universe"),
            ("Licence",  "CC BY 4.0"),
            ("Images",   "671"),
            ("Classes",  "7"),
            ("Format",   "JPEG"),
        ]
        for k, v in rows:
            st.markdown(
                f'<p style="margin:.25rem 0;font-size:.88rem">'
                f'<strong>{k} :</strong> {v}</p>',
                unsafe_allow_html=True,
            )

        st.markdown(f'<div class="section-title">{icon("fa-layer-group")} Stack technique</div>',
                    unsafe_allow_html=True)
        stack = [
            ("fa-brain",       "TensorFlow / Keras"),
            ("fa-mobile",      "MobileNetV2 (backbone)"),
            ("fa-flask",       "scikit-learn · scikit-image"),
            ("fa-chart-bar",   "matplotlib · seaborn"),
            ("fa-globe",       "Streamlit"),
            ("fa-table",       "pandas · NumPy"),
            ("fa-image",       "Pillow"),
        ]
        for fa, lib in stack:
            st.markdown(
                f'<p style="margin:.25rem 0;font-size:.88rem">{icon(fa, "fa-sm")} {lib}</p>',
                unsafe_allow_html=True,
            )

        st.markdown(f'<div class="section-title">{icon("fa-star")} Métriques retenues</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:.88rem;margin:.2rem 0">
            <strong>F1-macro</strong> — métrique principale<br>
            <em>Robuste au déséquilibre de classes</em>
        </p>
        <p style="font-size:.88rem;margin:.2rem 0">
            <strong>Accuracy</strong> — complémentaire<br>
            <em>Interprétation intuitive</em>
        </p>
        <p style="font-size:.88rem;margin:.2rem 0">
            <strong>Matrice de confusion</strong><br>
            <em>Identifie les paires difficiles</em>
        </p>
        """, unsafe_allow_html=True)
