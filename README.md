# Cat Emotions Classifier

Classificateur d'émotions félines par deep learning, développé dans le cadre du **Projet Data** (évaluation 2026).

## Problématique

> Développer un classificateur d'émotions félines à partir d'images pour aider propriétaires et vétérinaires à détecter le bien-être et le stress des chats lors des interactions humaines.

## Dataset

**Cat Emotions Dataset** — [Roboflow Universe](https://universe.roboflow.com/cats-xofvm/cat-emotions) · licence CC BY 4.0

| Propriété | Valeur |
|---|---|
| Images | 671 |
| Classes | 7 : `Angry` `Disgusted` `Happy` `Normal` `Sad` `Scared` `Surprised` |
| Format | JPEG, résolutions variées |
| Split | Re-splitté 70 / 15 / 15 (stratifié) dans le notebook |

## Résultats

| Modèle | Accuracy | F1-macro |
|---|---|---|
| Random Forest + HOG (baseline) | ~48.9% | ~46.7% |
| CNN from scratch | ~60.1% | ~57.8% |
| **MobileNetV2 Transfer Learning** | **~73.1%** | **~71.2%** |

*Seeds fixées à 42 — résultats reproductibles.*

---

## Installation

```bash
git clone <url-du-repo>
cd <nom-du-repo>

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Mac

pip install -r requirements.txt
```

> **Dataset :** placez les dossiers `train/` et `valid/` à la racine (non versionnés, voir `.gitignore`).

---

## Utilisation

### 1. Entraîner le modèle

**Option A — Script autonome** (recommandé, pas besoin de Jupyter) :
```bash
python train.py
```
Durée : 15–30 min CPU. Génère `best_mnv2_model.keras`.

**Option B — Notebook complet** (EDA + 3 modèles + Grad-CAM) :
```bash
python -m jupyter lab GroupeX_Evaluation_ProjetData.ipynb
# Menu Run → Run All Cells
```

### 2. Lancer l'application Streamlit

```bash
python -m streamlit run app.py
# ou sur un port spécifique :
python -m streamlit run app.py --server.port 8080
```

Ouvrir `http://localhost:8501` dans le navigateur.

---

## Structure du projet

```
.
├── GroupeX_Evaluation_ProjetData.ipynb   # Notebook ML complet (46 cellules)
├── app.py                                 # Application Streamlit 4 pages
├── train.py                               # Script d'entraînement autonome
├── requirements.txt                       # Dépendances Python
├── README.md
├── .gitignore
│
├── train/                                 # Dataset train (non versionné)
│   ├── Angry/
│   ├── Disgusted/
│   ├── Happy/
│   ├── Normal/
│   ├── Sad/
│   ├── Scared/
│   └── Surprised/
│
└── valid/                                 # Dataset validation (non versionné)
    └── ...
```

Fichiers générés après entraînement (non versionnés) :

```
best_mnv2_model.keras          # Modèle MobileNetV2 (utilisé par Streamlit)
best_cnn_model.keras           # Modèle CNN scratch
eda_*.png                      # Figures EDA
cnn_*.png / mnv2_*.png         # Courbes d'apprentissage
comparison_*.png               # Comparaisons
gradcam_*.png                  # Visualisations Grad-CAM
```

---

## Pipeline ML

```
Images brutes (671)
    ↓ Scan + détection images corrompues
    ↓ Re-split stratifié  70 % train  /  15 % valid  /  15 % test
    ↓ Resize 128×128 px
    ↓ Data augmentation : flip H · rotation ±15° · zoom ±10% · brightness ±20%
    ↓
Baseline  →  Random Forest + HOG (64×64, 9 orientations)
Modèle 1  →  CNN 4 blocs Conv + BatchNorm + GAP + Dropout(0.5)
Modèle 2  →  MobileNetV2 :
               phase 1 — feature extraction  (backbone gelé, LR=1e-3)
               phase 2 — fine-tuning 30 dernières couches  (LR=1e-5)
    ↓
Explicabilité  →  Grad-CAM via tf.GradientTape (couche out_relu)
```

---

## Application Streamlit

| Page | Contenu |
|---|---|
| **Prédiction** | Upload ou sélection d'exemple · prédiction temps réel · probabilités · Grad-CAM |
| **Dataset** | KPIs · distribution train/valid · galerie filtrée par émotion |
| **Résultats** | Tableau comparatif · graphiques · confusions fréquentes · fichiers générés |
| **À propos** | Contexte · pipeline · limites · stack technique |

---

## Dépendances

| Package | Rôle |
|---|---|
| `tensorflow >= 2.13` | CNN + MobileNetV2 + Grad-CAM |
| `scikit-learn` | Random Forest · métriques · split stratifié |
| `scikit-image` | Extraction features HOG |
| `streamlit >= 1.32` | Interface web |
| `pandas · numpy` | Manipulation données |
| `matplotlib · seaborn` | Visualisations |
| `Pillow` | Chargement et traitement images |

---

## Limites connues

- **671 images** : volume insuffisant pour un CNN industriel (variance élevée entre runs)
- **Annotations subjectives** : frontière floue entre `Scared` / `Angry` et `Normal` / `Disgusted`
- **Biais** : fond, race du chat, conditions d'éclairage
- **Seuil de confiance** : recommandé ≥ 60% pour usage clinique — en dessous, résultat incertain

## Pistes d'amélioration

- Enrichir à ≥ 200 images par classe
- Tester EfficientNetB0 / Vision Transformer (ViT)
- Ensemble CNN + MobileNetV2
- Validation croisée k-fold stratifiée
- Déploiement avec seuil de confiance automatique

---

## Auteurs

*(Noms et prénoms des membres du groupe)*

## Licence

Code : MIT · Dataset : CC BY 4.0 (Roboflow)
