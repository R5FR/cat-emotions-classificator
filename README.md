# 🐱 Cat Emotions Classifier — Projet Data

Classificateur d'émotions félines par deep learning, développé dans le cadre du **Projet Data** (évaluation 2026).

## Problématique

> Développer un classificateur d'émotions félines à partir d'images pour aider propriétaires et vétérinaires à détecter le bien-être et le stress des chats lors des interactions humaines.

## Dataset

**Cat Emotions Dataset** (Roboflow Universe) — licence CC BY 4.0
671 images · 7 classes : `Angry`, `Disgusted`, `Happy`, `Normal`, `Sad`, `Scared`, `Surprised`

## Structure du projet

```
.
├── GroupeX_Evaluation_ProjetData.ipynb   # Notebook complet (pipeline ML)
├── app.py                                 # Application Streamlit
├── requirements.txt
├── README.md
├── .gitignore
├── train/                                 # Dataset (non versionné)
│   ├── Angry/
│   ├── Disgusted/
│   └── ...
└── valid/                                 # Dataset validation (non versionné)
```

> **Note :** Les dossiers `train/` et `valid/` et les modèles `.keras` ne sont pas versionnés (voir `.gitignore`).
> Téléchargez le dataset depuis [Roboflow Universe](https://universe.roboflow.com/cats-xofvm/cat-emotions) et placez-le à la racine.

## Installation

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd <nom-du-repo>

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### 1. Entraîner les modèles

Ouvrir et exécuter **toutes les cellules** de `GroupeX_Evaluation_ProjetData.ipynb`.

Ce notebook :
- Scanne et nettoie le dataset
- Effectue le re-split stratifié 70/15/15
- Réalise l'EDA complète
- Entraîne 3 modèles (Random Forest + HOG, CNN scratch, MobileNetV2)
- Compare les performances (accuracy, F1-macro, matrices de confusion)
- Génère les visualisations Grad-CAM
- Sauvegarde `best_mnv2_model.keras` (utilisé par l'app Streamlit)

### 2. Lancer l'application Streamlit

```bash
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501` avec 4 pages :
- **Prédiction** — upload d'image + prédiction temps réel + Grad-CAM
- **Dashboard Dataset** — EDA interactive, galerie par classe
- **Résultats Modèles** — comparaison des performances
- **À propos** — contexte, pipeline, limites

## Résultats

| Modèle | Accuracy | F1-macro |
|--------|----------|----------|
| Random Forest + HOG (baseline) | ~48.9% | ~46.7% |
| CNN from scratch | ~60.1% | ~57.8% |
| **MobileNetV2 (Transfer Learning)** | **~73.1%** | **~71.2%** |

*Les valeurs exactes dépendent du run — seeds fixées à 42 pour la reproductibilité.*

## Pipeline ML

```
Images brutes (671)
    ↓ Nettoyage (images corrompues)
    ↓ Re-split stratifié 70 / 15 / 15
    ↓ Resize 128×128 + normalisation
    ↓ Data augmentation (flip H, rotation ±15°, zoom ±10%, brightness ±20%)
    ↓
Baseline  →  Random Forest + HOG features
Modèle 1  →  CNN 4 blocs (Conv + BatchNorm + GAP) + Dropout
Modèle 2  →  MobileNetV2 : phase 1 (feature extraction) + phase 2 (fine-tuning)
    ↓
Explicabilité  →  Grad-CAM (zones : oreilles, yeux, moustaches)
```

## Dépendances principales

- Python ≥ 3.9
- TensorFlow ≥ 2.13
- Streamlit ≥ 1.32
- scikit-learn, scikit-image, pandas, matplotlib, seaborn, Pillow

## Auteurs

*(Noms et prénoms des membres du groupe)*

## Licence

Code : MIT — Dataset : CC BY 4.0 (Roboflow)
