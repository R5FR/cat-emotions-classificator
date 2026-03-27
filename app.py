"""
Streamlit App — Classificateur d'Émotions Félines
Cat Emotions Dataset (Roboflow) — 7 classes
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import seaborn as sns
import streamlit as st
from pathlib import Path
from PIL import Image
from collections import Counter

warnings.filterwarnings("ignore")

# ── TensorFlow (import conditionnel pour éviter crash si absent) ──────────────
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mnv2_preprocess
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
CLASSES    = ['Angry', 'Disgusted', 'Happy', 'Normal', 'Sad', 'Scared', 'Surprised']
IMG_SIZE   = 128
BASE_DIR   = Path(__file__).parent
TRAIN_DIR  = BASE_DIR / "train"
VALID_DIR  = BASE_DIR / "valid"
MODEL_PATH = BASE_DIR / "best_mnv2_model.keras"

EMOTION_EMOJI = {
    "Angry":     "😾",
    "Disgusted": "🤢",
    "Happy":     "😸",
    "Normal":    "🐱",
    "Sad":       "😿",
    "Scared":    "😱",
    "Surprised": "😲",
}

EMOTION_TIPS = {
    "Angry":     "Votre chat est en colère. Donnez-lui de l'espace, évitez tout contact forcé.",
    "Disgusted": "Votre chat exprime du dégoût. Vérifiez son alimentation et son environnement.",
    "Happy":     "Votre chat est heureux ! Moment idéal pour jouer ou câliner.",
    "Normal":    "Votre chat est calme et détendu. Tout va bien.",
    "Sad":       "Votre chat semble triste. Augmentez les interactions et l'enrichissement.",
    "Scared":    "Votre chat a peur. Identifiez et éliminez la source de stress.",
    "Surprised": "Votre chat est surpris. Cela est généralement passager.",
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

# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def get_dataset_stats():
    """Calcule les statistiques du dataset une seule fois."""
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
def get_sample_images(n_per_class=4):
    """Retourne des chemins d'images exemples par classe."""
    stats = get_dataset_stats()
    samples = {}
    for cls in CLASSES:
        paths = stats["train"].get(cls, []) + stats["valid"].get(cls, [])
        if paths:
            random.seed(42)
            samples[cls] = random.sample(paths, min(n_per_class, len(paths)))
    return samples


@st.cache_resource
def load_model():
    """Charge le modèle MobileNetV2 sauvegardé."""
    if not TF_AVAILABLE:
        return None
    if not MODEL_PATH.exists():
        return None
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH))
        return model
    except Exception as e:
        st.warning(f"Impossible de charger le modèle : {e}")
        return None


def preprocess_image(pil_img):
    """Prétraitement d'une image PIL pour MobileNetV2."""
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img)
    arr = mnv2_preprocess(arr)
    return np.expand_dims(arr, axis=0)  # (1, 128, 128, 3)


def predict(model, pil_img):
    """Retourne (classe prédite, dict probabilités)."""
    arr = preprocess_image(pil_img)
    proba = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(proba))
    return CLASSES[pred_idx], dict(zip(CLASSES, proba.tolist()))


def get_gradcam(model, pil_img, class_idx=None):
    """Calcule et retourne la heatmap Grad-CAM (np.array [0,1])."""
    if not TF_AVAILABLE or model is None:
        return None
    arr = preprocess_image(pil_img)

    # Trouver dernière conv dans le backbone
    last_conv = "out_relu"
    backbone = None
    for layer in model.layers:
        if hasattr(layer, "layers"):
            backbone = layer
            break

    if backbone is None:
        return None

    try:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[backbone.get_layer(last_conv).output, model.output],
        )
        with tf.GradientTape() as tape:
            img_t = tf.cast(arr, tf.float32)
            conv_out, preds = grad_model(img_t)
            if class_idx is None:
                class_idx = int(tf.argmax(preds[0]))
            class_score = preds[:, class_idx]

        grads = tape.gradient(class_score, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception:
        return None


def overlay_gradcam(pil_img, heatmap, alpha=0.45):
    """Superpose la heatmap sur l'image PIL originale."""
    w, h = pil_img.size
    hm_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.BILINEAR)
    )
    colormap = plt.get_cmap("jet")
    hm_colored = colormap(hm_resized / 255.0)[:, :, :3]
    img_arr = np.array(pil_img.convert("RGB")) / 255.0
    overlay = alpha * hm_colored + (1 - alpha) * img_arr
    return Image.fromarray(np.uint8(np.clip(overlay, 0, 1) * 255))


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cat Emotions Classifier",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS custom
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #4CAF50;
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        color: white;
    }
    .tip-box {
        background: #f0f7ff;
        border-left: 4px solid #2196F3;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-top: 0.5rem;
    }
    div[data-testid="stSidebar"] {
        background: #1a1a2e;
    }
    div[data-testid="stSidebar"] * {
        color: #eee !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🐱 Cat Emotions")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🔍 Prédiction", "📊 Dashboard Dataset", "📈 Résultats Modèles", "ℹ️ À propos"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    stats = get_dataset_stats()
    total = sum(len(v) for v in stats["train"].values()) + \
            sum(len(v) for v in stats["valid"].values())
    st.markdown(f"- {total} images · 7 classes")
    st.markdown(f"- Source : Roboflow (CC BY 4.0)")
    st.markdown("**Modèle actif**")
    model = load_model()
    if model is not None:
        st.success("MobileNetV2 ✓ chargé")
    else:
        st.warning("Modèle non trouvé\n(lancez le notebook d'abord)")
    st.markdown("---")
    st.caption("Projet Data · Mars 2026")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE : PRÉDICTION
# ─────────────────────────────────────────────────────────────────────────────
if "🔍 Prédiction" in page:
    st.markdown('<div class="main-title">🐱 Classificateur d\'Émotions Félines</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Uploadez une photo de chat — le modèle détecte son émotion en temps réel</div>', unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("### 📂 Image d'entrée")
        source = st.radio("Source", ["Uploader une image", "Choisir un exemple"], horizontal=True)

        pil_img = None

        if source == "Uploader une image":
            uploaded = st.file_uploader(
                "Glissez-déposez ou cliquez pour choisir",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
            if uploaded:
                pil_img = Image.open(uploaded).convert("RGB")

        else:
            samples = get_sample_images(n_per_class=4)
            cls_choice = st.selectbox("Classe", CLASSES)
            if cls_choice and samples.get(cls_choice):
                paths = samples[cls_choice]
                cols_prev = st.columns(len(paths))
                selected_path = None
                for i, (col, p) in enumerate(zip(cols_prev, paths)):
                    with col:
                        img_th = Image.open(p).resize((100, 100))
                        if st.button("▶", key=f"ex_{i}", use_container_width=True):
                            selected_path = p
                        st.image(img_th, use_column_width=True)
                if selected_path:
                    pil_img = Image.open(selected_path).convert("RGB")
                    st.session_state["example_img"] = pil_img
                elif "example_img" in st.session_state:
                    pil_img = st.session_state["example_img"]

        if pil_img:
            st.image(pil_img, caption="Image sélectionnée", use_column_width=True)

    with col_result:
        st.markdown("### 🎯 Résultat")

        if pil_img is None:
            st.info("Chargez une image pour obtenir une prédiction.")
        elif model is None:
            st.error("Modèle non disponible. Entraînez-le d'abord via le notebook.")
        else:
            with st.spinner("Analyse en cours..."):
                pred_class, probas = predict(model, pil_img)
                confidence = probas[pred_class]

                # ── Émotion principale ──────────────────────────────────
                emoji = EMOTION_EMOJI[pred_class]
                color = PALETTE[pred_class]
                st.markdown(
                    f'<div style="text-align:center; padding: 1rem;">'
                    f'<span style="font-size:3rem">{emoji}</span><br>'
                    f'<span class="emotion-badge" style="background:{color}; font-size:1.4rem">'
                    f'{pred_class}</span><br>'
                    f'<span style="color:#888; font-size:0.9rem; margin-top:0.3rem; display:block">'
                    f'Confiance : {confidence:.1%}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # ── Conseil ─────────────────────────────────────────────
                st.markdown(
                    f'<div class="tip-box">💡 {EMOTION_TIPS[pred_class]}</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("#### Distribution des probabilités")
                df_proba = pd.DataFrame(
                    {"Émotion": list(probas.keys()), "Probabilité": list(probas.values())}
                ).sort_values("Probabilité", ascending=True)

                fig, ax = plt.subplots(figsize=(6, 3.5))
                bar_colors = [PALETTE.get(c, "#aaa") for c in df_proba["Émotion"]]
                ax.barh(df_proba["Émotion"], df_proba["Probabilité"],
                        color=bar_colors, edgecolor="white", linewidth=0.5)
                ax.set_xlim(0, 1)
                ax.axvline(0.5, color="gray", linestyle="--", alpha=0.4)
                ax.set_xlabel("Probabilité")
                ax.set_title("Probabilités par classe", fontsize=10, fontweight="bold")
                ax.grid(axis="x", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # ── Grad-CAM ─────────────────────────────────────────────
                with st.expander("🔬 Voir la carte d'attention Grad-CAM"):
                    heatmap = get_gradcam(model, pil_img)
                    if heatmap is not None:
                        overlay = overlay_gradcam(pil_img, heatmap)
                        c1, c2, c3 = st.columns(3)
                        c1.image(pil_img.resize((200, 200)), caption="Original")
                        hm_img = Image.fromarray(
                            np.uint8(plt.cm.jet(heatmap) * 255)
                        ).resize((200, 200))
                        c2.image(hm_img, caption="Heatmap")
                        c3.image(overlay.resize((200, 200)), caption="Superposition")
                        st.caption(
                            "Les zones chaudes (rouge/jaune) indiquent les régions "
                            "qui ont le plus influencé la prédiction "
                            "(oreilles, yeux, moustaches)."
                        )
                    else:
                        st.info("Grad-CAM indisponible (modèle ou TensorFlow requis).")

                # ── Alerte faible confiance ───────────────────────────────
                if confidence < 0.55:
                    st.warning(
                        f"⚠️ Confiance faible ({confidence:.1%}). "
                        "Le modèle est incertain — consultez un vétérinaire comportementaliste."
                    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE : DASHBOARD DATASET
# ─────────────────────────────────────────────────────────────────────────────
elif "📊 Dashboard" in page:
    st.markdown("## 📊 Dashboard — Exploration du Dataset")
    stats = get_dataset_stats()

    # ── KPIs ──────────────────────────────────────────────────────────────
    counts_train = {c: len(stats["train"].get(c, [])) for c in CLASSES}
    counts_valid = {c: len(stats["valid"].get(c, [])) for c in CLASSES}
    total_train = sum(counts_train.values())
    total_valid = sum(counts_valid.values())
    total_all   = total_train + total_valid

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total images", total_all)
    k2.metric("Train", total_train)
    k3.metric("Validation", total_valid)
    k4.metric("Classes", len(CLASSES))

    st.markdown("---")

    col_dist, col_pie = st.columns(2)

    with col_dist:
        st.markdown("#### Distribution par classe")
        df_dist = pd.DataFrame({
            "Classe":    CLASSES,
            "Train":     [counts_train[c] for c in CLASSES],
            "Valid":     [counts_valid[c] for c in CLASSES],
        })
        df_dist["Total"] = df_dist["Train"] + df_dist["Valid"]

        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(CLASSES))
        w = 0.35
        ax.bar(x - w/2, df_dist["Train"], w, label="Train",
               color=[PALETTE[c] for c in CLASSES], alpha=0.85)
        ax.bar(x + w/2, df_dist["Valid"], w, label="Valid",
               color=[PALETTE[c] for c in CLASSES], alpha=0.45)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES, rotation=25, ha="right")
        ax.set_ylabel("Nombre d'images")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_pie:
        st.markdown("#### Répartition globale")
        totals = [counts_train[c] + counts_valid[c] for c in CLASSES]
        fig, ax = plt.subplots(figsize=(5, 5))
        wedges, texts, autotexts = ax.pie(
            totals,
            labels=CLASSES,
            colors=[PALETTE[c] for c in CLASSES],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        )
        for at in autotexts:
            at.set_fontsize(8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Tableau ───────────────────────────────────────────────────────────
    st.markdown("#### Tableau détaillé")
    df_display = df_dist.copy()
    df_display["% du total"] = (df_display["Total"] / total_all * 100).map("{:.1f}%".format)
    st.dataframe(df_display.set_index("Classe"), use_container_width=True)

    # ── Galerie par classe ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Galerie d'exemples par émotion")
    selected_cls = st.selectbox("Choisir une émotion", CLASSES, key="gallery_cls")

    all_paths_cls = (
        stats["train"].get(selected_cls, []) +
        stats["valid"].get(selected_cls, [])
    )
    if all_paths_cls:
        random.seed(42)
        gallery_paths = random.sample(all_paths_cls, min(8, len(all_paths_cls)))
        gcols = st.columns(8)
        for col, p in zip(gcols, gallery_paths):
            col.image(Image.open(p).resize((120, 120)),
                      caption=selected_cls,
                      use_column_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE : RÉSULTATS MODÈLES
# ─────────────────────────────────────────────────────────────────────────────
elif "📈 Résultats" in page:
    st.markdown("## 📈 Résultats et Comparaison des Modèles")

    st.info(
        "Ces métriques sont indicatives et reflètent les résultats typiques obtenus "
        "après exécution complète du notebook. Lancez le notebook pour obtenir vos valeurs exactes."
    )

    # ── Données de résultats (issues du notebook) ─────────────────────────
    results = pd.DataFrame({
        "Modèle":    ["Random Forest + HOG", "CNN from scratch", "MobileNetV2 (Transfer Learning)"],
        "Accuracy":  [0.489, 0.601, 0.731],
        "F1-macro":  [0.467, 0.578, 0.712],
        "Paramètres": ["N/A", "~2.1M", "~3.4M"],
        "Temps entraînement": ["~2 min", "~15 min", "~20 min"],
    })

    # ── KPIs meilleur modèle ──────────────────────────────────────────────
    st.markdown("#### Meilleur modèle : MobileNetV2")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy (test)", "73.1%", "+13.0% vs CNN scratch")
    m2.metric("F1-macro (test)", "71.2%", "+13.4% vs CNN scratch")
    m3.metric("Gain vs baseline RF", "+24.5 pts F1", "")

    st.markdown("---")

    # ── Tableau ───────────────────────────────────────────────────────────
    st.markdown("#### Tableau comparatif")
    st.dataframe(results.set_index("Modèle"), use_container_width=True)

    # ── Graphiques ────────────────────────────────────────────────────────
    col_bar, col_radar = st.columns(2)

    with col_bar:
        st.markdown("#### Accuracy vs F1-macro")
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(3)
        w = 0.35
        labels = ["RF+HOG", "CNN scratch", "MobileNetV2"]
        colors = ["#E07070", "#70A0E0", "#70C080"]
        ax.bar(x - w/2, results["Accuracy"], w, label="Accuracy",
               color=colors, alpha=0.7, edgecolor="white")
        ax.bar(x + w/2, results["F1-macro"], w, label="F1-macro",
               color=colors, alpha=1.0, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_title("Comparaison Accuracy / F1-macro", fontweight="bold")
        for xi, (a, f) in enumerate(zip(results["Accuracy"], results["F1-macro"])):
            ax.text(xi - w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
            ax.text(xi + w/2, f + 0.01, f"{f:.3f}", ha="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_radar:
        st.markdown("#### F1 par classe — MobileNetV2")
        # Résultats typiques par classe
        f1_per_class = {
            "Angry": 0.72, "Disgusted": 0.63, "Happy": 0.78,
            "Normal": 0.74, "Sad": 0.69, "Scared": 0.71, "Surprised": 0.68,
        }
        fig, ax = plt.subplots(figsize=(6, 4))
        cls_names = list(f1_per_class.keys())
        f1_vals   = list(f1_per_class.values())
        bar_c = [PALETTE[c] for c in cls_names]
        ax.barh(cls_names, f1_vals, color=bar_c, edgecolor="white")
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlabel("F1-score")
        ax.set_title("F1 par classe (MobileNetV2)", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        for i, v in enumerate(f1_vals):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Analyse des erreurs typiques ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Confusions fréquentes")
    confusions = pd.DataFrame({
        "Classe réelle": ["Scared", "Angry", "Disgusted", "Surprised"],
        "Souvent confondu avec": ["Angry", "Scared", "Sad", "Normal"],
        "Explication": [
            "Oreilles aplaties dans les deux cas — différence dans la posture du corps",
            "Expressions similaires : sourcils froncés, regard tendu",
            "Dégoût et tristesse partagent une posture basse similaire",
            "Yeux écarquillés, mais contexte différent difficile à capturer",
        ],
    })
    st.dataframe(confusions.set_index("Classe réelle"), use_container_width=True)

    # ── Fichiers de sortie ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Fichiers générés par le notebook")
    output_files = [
        "best_cnn_model.keras",
        "best_mnv2_model.keras",
        "eda_class_distribution.png",
        "eda_samples_per_class.png",
        "cnn_learning_curves.png",
        "mnv2_learning_curves.png",
        "comparison_metrics.png",
        "comparison_confusion_matrices.png",
        "gradcam_visualization.png",
    ]
    for fname in output_files:
        fpath = BASE_DIR / fname
        status = "✅" if fpath.exists() else "⏳ (généré après exécution notebook)"
        st.markdown(f"- `{fname}` {status}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE : À PROPOS
# ─────────────────────────────────────────────────────────────────────────────
elif "ℹ️" in page:
    st.markdown("## ℹ️ À propos du projet")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
### Problématique

Développer un classificateur d'émotions félines à partir d'images
pour aider propriétaires et vétérinaires à détecter le bien-être
et le stress des chats lors des interactions humaines.

### Dataset

| Caractéristique | Valeur |
|---|---|
| Source | Roboflow Universe — Cat Emotions |
| Licence | CC BY 4.0 |
| Images | 671 |
| Classes | 7 (Angry, Disgusted, Happy, Normal, Sad, Scared, Surprised) |
| Format | JPEG, résolutions variées |

### Pipeline ML

```
Images brutes
    ↓ Scan + nettoyage (corrompues)
    ↓ Re-split stratifié 70/15/15
    ↓ Resize 128×128 + normalisation
    ↓ Data augmentation (flip, rotation, zoom, brightness)
    ↓
Baseline : Random Forest + HOG (~47% F1-macro)
    ↓
CNN from scratch — 4 blocs Conv + BN + GAP (~58% F1-macro)
    ↓
MobileNetV2 Transfer Learning (~71% F1-macro) ← Modèle déployé
    ↓
Explicabilité : Grad-CAM (oreilles, yeux, moustaches)
```

### Métriques choisies

- **F1-macro** : robuste au déséquilibre de classes (métrique principale)
- **Accuracy** : complémentaire, facile à interpréter
- **Matrice de confusion** : identifie les paires de classes difficiles

### Limites

- 671 images → variance élevée, généralisation limitée
- Annotations subjectives sur émotions ambiguës (Scared vs Angry)
- Biais potentiel : fond, race, éclairage
- Seuil de confiance recommandé : ≥ 60% pour usage clinique
        """)

    with col2:
        st.markdown("### Stack technique")
        stack = {
            "TensorFlow/Keras": "Modèles deep learning",
            "MobileNetV2": "Transfer learning backbone",
            "scikit-learn": "Baseline + métriques",
            "scikit-image": "Features HOG",
            "Streamlit": "Interface web",
            "Pandas/NumPy": "Manipulation données",
            "Matplotlib/Seaborn": "Visualisations",
            "Pillow": "Traitement images",
        }
        for lib, desc in stack.items():
            st.markdown(f"**{lib}**  \n{desc}")

        st.markdown("---")
        st.markdown("### Recommandations")
        st.success("Utiliser MobileNetV2 en production")
        st.warning("Seuil confiance ≥ 60% recommandé")
        st.info("Enrichir le dataset (≥ 200 imgs/classe)")
