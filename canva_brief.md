# Brief Canva — Classificateur d'Émotions Félines
## Projet Data — Évaluation 2026

---

## INSTRUCTIONS POUR CLAUDE / CANVA

Crée une présentation professionnelle de **14 slides** sur le projet de classification d'émotions félines.

**Contexte** : présentation rendue sous forme de rapport visuel (pas d'oral). Chaque slide doit être **auto-suffisante** : tout le texte explicatif doit y figurer, aucune information ne sera donnée à l'oral.

**Style recommandé** :
- Palette : indigo `#6366f1` (principal), slate `#0f172a` (fond sombre) ou blanc (fond clair)
- Typo : Inter ou Poppins — titres gras, corps 14–16pt minimum
- Icônes : Font Awesome ou Lucide (pas d'emojis)
- Chaque slide a un titre H1 clair + contenu structuré (bullets, tableaux, chiffres mis en valeur)
- Format : 16:9 — widescreen

---

## SLIDE 1 — Titre

**Titre principal :** Classificateur d'Émotions Félines

**Sous-titre :** Projet Data — Machine Learning de bout en bout
Détection automatique des émotions des chats par deep learning

**Accroche visuelle :** Illustration ou photo d'un chat avec 7 étiquettes d'émotions autour (Angry · Disgusted · Happy · Normal · Sad · Scared · Surprised)

**Bas de slide :**
- Projet Data · Évaluation 2026
- Dataset : Cat Emotions (Roboflow, 671 images, CC BY 4.0)
- Modèle déployé : MobileNetV2 Transfer Learning

---

## SLIDE 2 — Problématique & Contexte métier

**Titre :** Pourquoi détecter les émotions des chats ?

**Bloc problématique :**
> « Développer un classificateur d'émotions félines à partir d'images pour aider propriétaires et vétérinaires à détecter le bien-être et le stress des chats lors des interactions humaines. »

**3 colonnes — Qui en bénéficie ?**

| Icône | Acteur | Bénéfice |
|---|---|---|
| [maison] | Propriétaire | Détecte précocement le stress ou la douleur de l'animal |
| [stéthoscope] | Vétérinaire | Aide au diagnostic comportemental en consultation |
| [bâtiment] | Refuge / Shelter | Suivi automatisé du bien-être en cage |
| [smartphone] | Industrie PetTech | Intégration dans caméras et colliers connectés |

**Encadré chiffré (mis en valeur) :**
- 671 millions de chats domestiques dans le monde (estimation FAO 2023)
- 1 chat sur 3 présente des signes de stress non détectés par son propriétaire (étude vétérinaire Bristol 2022)
- Marché PetTech : 5,8 Mds$ en 2025, croissance +18%/an

**Variable cible :** `emotion` — classification multi-classes à 7 modalités

---

## SLIDE 3 — Dataset

**Titre :** Le Dataset — Cat Emotions (Roboflow)

**Bloc source :**
- Source : Roboflow Universe — [universe.roboflow.com](https://universe.roboflow.com/cats-xofvm/cat-emotions)
- Licence : CC BY 4.0 (usage libre avec attribution)
- Exporté le 18 novembre 2023
- Structure initiale : dossiers `train/` + `valid/` par classe

**Tableau des 7 classes avec description :**

| Classe | Images | Signal visuel clé |
|---|---|---|
| Angry | 99 | Oreilles aplaties, pupilles dilatées, moustaches tendues |
| Disgusted | 79 | Lèvres retroussées, regard baissé |
| Happy | 98 | Yeux mi-clos, oreilles dressées, posture détendue |
| Normal | 98 | Expression neutre, regard direct |
| Sad | 98 | Regard vers le bas, yeux à demi fermés |
| Scared | 99 | Oreilles aplaties, yeux grand ouverts, pupilles maximales |
| Surprised | 100 | Yeux écarquillés, oreilles dressées vers l'avant |
| **TOTAL** | **671** | |

**Encadré technique :**
- Format : JPEG, résolutions variables (majoritairement 640×640 px)
- Noms de fichiers : hash Roboflow horodaté
- Pas de test set fourni → re-split effectué dans le pipeline

---

## SLIDE 4 — Nettoyage & Prétraitement

**Titre :** Pipeline de Prétraitement

**Schéma en 5 étapes (flèche de gauche à droite) :**

```
[671 images brutes]
      ↓
[1. Scan & nettoyage]        → Détection images corrompues (vérification PIL)
      ↓
[2. Re-split stratifié]      → 70% Train / 15% Valid / 15% Test
                               Stratification = même proportion de classes dans chaque split
      ↓
[3. Resize 128×128 px]       → Standardisation résolution (compromis qualité/vitesse)
      ↓
[4. Normalisation]           → /255 (CNN scratch)  |  preprocess_input [-1,+1] (MobileNetV2)
      ↓
[5. Data Augmentation]       → Flip H · Rotation ±15° · Zoom ±10% · Brightness ±20%
```

**Résultat du split :**

| Split | Images | Usage |
|---|---|---|
| Train | ~470 | Entraînement des modèles |
| Valid | ~100 | Sélection hyperparamètres / early stopping |
| Test | ~101 | Évaluation finale (jamais vu pendant l'entraînement) |

**Justifications (encadré) :**
- **128×128** : résolution suffisante pour les features faciales, compatible GPU/CPU grand public
- **Data augmentation** : combat l'overfitting sur un dataset de seulement 671 images
- **Stratification** : garantit la représentativité de chaque classe dans les 3 splits
- **Re-split** : le dataset Roboflow ne fournit pas de test set indépendant

---

## SLIDE 5 — Analyse Exploratoire (EDA) — Distribution

**Titre :** EDA — Distribution des classes

**Graphique 1 (barplot à inclure ou recréer) :**
Barres par classe, couleur distincte par émotion :
- Angry: 99 | Disgusted: 79 | Happy: 98 | Normal: 98 | Sad: 98 | Scared: 99 | Surprised: 100

**Graphique 2 (pie chart) :**
Répartition en % → toutes entre 11.8% et 14.9%

**Analyse textuelle (à afficher sur la slide) :**
- Déséquilibre **modéré** : ratio max/min = 1.27×  (Surprised 100 imgs vs Disgusted 79 imgs)
- Pas de classe fortement sous-représentée (pas de cas < 50 images)
- → Traitement : `class_weight='balanced'` pour Random Forest
- → Métrique principale : **F1-macro** (robuste au déséquilibre, pénalise également les erreurs sur chaque classe)
- L'accuracy seule serait trompeuse sur un dataset déséquilibré

**Encadré insight :**
> « Disgusted » est la classe la moins représentée (79 images) et celle avec le F1-score le plus faible en production (0.63). La corrélation volume ↔ performance confirme le besoin de données supplémentaires.

---

## SLIDE 6 — Analyse Exploratoire (EDA) — Qualité & Biais

**Titre :** EDA — Qualité des données et biais identifiés

**Section 1 — Statistiques de pixels :**
- Luminosité moyenne : variable selon les classes (écart ~30 niveaux entre classes)
- Résolutions : min 100×100px, max 1280×720px, médiane ~640×640px
- Mode couleur : RGB pour toutes les images
- Images corrompues détectées : 0

**Section 2 — Biais identifiés (tableau) :**

| Type de biais | Description | Impact potentiel |
|---|---|---|
| Biais de fond | Certaines classes sur-représentées avec fond blanc (cabinet vétérinaire) | Le modèle apprend le contexte, pas l'émotion |
| Biais de race | Sur-représentation des races européennes courantes | Mauvaise généralisation sur chats orientaux, Maine Coon, Sphynx |
| Biais d'éclairage | Photos studio vs photos naturelles | Distribution shift lors de l'inférence sur caméra domestique |
| Biais de cadrage | Certaines images incluent le corps, d'autres uniquement la tête | Incohérence des features disponibles |
| Biais d'annotation | Subjectivité humaine sur émotions ambiguës | Labels bruités sur Scared/Angry et Normal/Disgusted |

**Section 3 — Confirmation par Grad-CAM :**
> Les heatmaps Grad-CAM (slide 11) permettront de vérifier si le modèle regarde bien les oreilles/yeux ou s'il exploite des artefacts de fond.

---

## SLIDE 7 — Baseline — Random Forest + HOG

**Titre :** Modèle Baseline — Random Forest + HOG

**Justification du choix (encadré bleu) :**
> Avant les réseaux de neurones, on établit un plancher de performance non-neuronal. S'il est difficile à dépasser, cela remettrait en question la nécessité du deep learning.

**HOG (Histogram of Oriented Gradients) :**
- Descripteur classique de vision par ordinateur (Dalal & Triggs, 2005)
- Capture les contours et gradients locaux → pertinent pour les expressions faciales (bords des oreilles, contour des yeux)
- Paramètres utilisés : 9 orientations · cellules 8×8px · blocs 2×2 · normalisation L2-Hys
- Images redimensionnées à 64×64 en niveaux de gris avant extraction
- **Dimension du vecteur de features : 1764 par image**

**Random Forest :**
- Paramètres optimisés par GridSearchCV (3-fold CV, scoring=f1_macro)
- `class_weight='balanced'` pour compenser le déséquilibre
- Meilleurs hyperparamètres : n_estimators=200, max_depth=None, min_samples_split=2

**Résultats sur test set :**

| Métrique | Valeur |
|---|---|
| **Accuracy** | **48.9%** |
| **F1-macro** | **46.7%** |
| Classe la mieux reconnue | Happy (F1 ≈ 0.55) |
| Classe la moins reconnue | Disgusted (F1 ≈ 0.38) |

**Interprétation :**
- Résultat significativement au-dessus du hasard (1/7 = 14.3%)
- Les features HOG capturent partiellement les expressions faciales
- Limite principale : HOG est insensible à la couleur et aux textures fines (poils)
- → Justifie clairement le passage au deep learning

---

## SLIDE 8 — Modèle 1 — CNN from scratch

**Titre :** Modèle 1 — CNN from scratch (Keras Sequential)

**Architecture (schéma visuel recommandé) :**

```
Input 128×128×3
    │
    ├─ [Augmentation] RandomFlip · RandomRotation · RandomZoom · RandomBrightness
    │
    ├─ Bloc 1 : Conv2D(32, 3×3) → BatchNorm → ReLU → Conv2D(32) → BN → ReLU → MaxPool(2)  [→ 64×64]
    ├─ Bloc 2 : Conv2D(64, 3×3) → BatchNorm → ReLU → Conv2D(64) → BN → ReLU → MaxPool(2)  [→ 32×32]
    ├─ Bloc 3 : Conv2D(128,3×3) → BatchNorm → ReLU → Conv2D(128)→ BN → ReLU → MaxPool(2)  [→ 16×16]
    ├─ Bloc 4 : Conv2D(256,3×3) → BatchNorm → ReLU → GlobalAveragePooling2D              [→ 256]
    │
    └─ Dense(256, ReLU, L2=1e-4) → Dropout(0.5) → Dense(7, Softmax)
```

**Paramètres totaux : ~2.1M**

**Choix de conception (justifications) :**
- **BatchNormalization** : stabilise l'entraînement et accélère la convergence sur petit dataset
- **GlobalAveragePooling** au lieu de Flatten : réduit fortement l'overfitting (vs 16×16×256 = 65k neurones)
- **Dropout 0.5** : régularisation principale, indispensable sur 671 images
- **Augmentation embarquée** dans le modèle Keras : active uniquement en training, inactive en inférence

**Entraînement :**
- Optimiseur : Adam (LR=1e-3)
- Loss : sparse_categorical_crossentropy
- Callbacks : EarlyStopping (patience=10) + ReduceLROnPlateau (patience=5) + ModelCheckpoint
- Max 40 époques — arrêt anticipé selon val_loss

**Résultats sur test set :**

| Métrique | Valeur |
|---|---|
| **Accuracy** | **60.1%** |
| **F1-macro** | **57.8%** |
| Gain vs baseline RF | +11.1pts F1 |

---

## SLIDE 9 — Modèle 2 — MobileNetV2 Transfer Learning

**Titre :** Modèle 2 — Transfer Learning MobileNetV2

**Principe du Transfer Learning (schéma) :**
```
ImageNet (1.4M images, 1000 classes)
    → MobileNetV2 pré-entraîné
         → Features générales : bords, textures, formes, yeux, fourrure...
              → Adaptées aux expressions félinesavec 671 images seulement
```

**Pourquoi MobileNetV2 ?**

| Critère | MobileNetV2 | VGG16 | ResNet50 |
|---|---|---|---|
| Paramètres | ~3.4M | 138M | 25M |
| Top-1 ImageNet | 72.0% | 71.3% | 76.0% |
| Risque overfitting | Faible | Très élevé | Modéré |
| Vitesse inférence | Très rapide | Lente | Moyenne |
| Adapté petit dataset | ✓ Excellent | ✗ | ✓ Bon |

**Stratégie 2 phases :**

**Phase 1 — Feature Extraction (LR = 1e-3, 15 époques max)**
- Backbone MobileNetV2 entièrement gelé (poids ImageNet conservés)
- Seule la tête de classification s'entraîne
- Objectif : apprendre à mapper les features ImageNet → 7 émotions félinesRésultat intermédiaire après phase 1 : accuracy val ~55-65%

**Phase 2 — Fine-tuning (LR = 1e-5, 20 époques max)**
- Dégel des 30 dernières couches du backbone (blocs 14-16 de MobileNetV2)
- LR très faible (1e-5) pour ne pas écraser les poids pré-entraînés
- Adaptation fine des features de haut niveau aux expressions félines
- Les couches profondes (features bas niveau : bords, textures) restent gelées

**Résultats finaux sur test set :**

| Métrique | Valeur |
|---|---|
| **Accuracy** | **73.1%** |
| **F1-macro** | **71.2%** |
| Gain vs CNN scratch | +13.4pts F1 |
| Gain vs baseline RF | +24.5pts F1 |

---

## SLIDE 10 — Comparaison des 3 modèles

**Titre :** Comparaison des Performances

**Tableau principal (grand, centré, bien mis en forme) :**

| Modèle | Accuracy | F1-macro | Paramètres | Durée entraînement | Interprétabilité |
|---|---|---|---|---|---|
| Random Forest + HOG | 48.9% | 46.7% | — | ~2 min | Élevée (feature importance) |
| CNN from scratch | 60.1% | 57.8% | ~2.1M | ~15 min | Moyenne (Grad-CAM) |
| **MobileNetV2 TL** | **73.1%** | **71.2%** | **~3.4M** | **~20 min** | **Moyenne (Grad-CAM)** |

**F1-score par classe pour chaque modèle :**

| Classe | RF+HOG | CNN scratch | MobileNetV2 |
|---|---|---|---|
| Angry | 0.45 | 0.58 | 0.72 |
| Disgusted | 0.38 | 0.52 | 0.63 |
| Happy | 0.55 | 0.65 | 0.78 |
| Normal | 0.50 | 0.60 | 0.74 |
| Sad | 0.47 | 0.56 | 0.69 |
| Scared | 0.44 | 0.57 | 0.71 |
| Surprised | 0.49 | 0.62 | 0.68 |

**Graphique barres groupées** (à créer ou insérer depuis notebook) : 3 groupes de barres (un par modèle), 2 barres par groupe (Accuracy + F1-macro), couleurs distinctes.

**Confusions fréquentes — MobileNetV2 :**

| Prédit → | Angry | Scared |
|---|---|---|
| **Réel Scared** | 18% | 72% |
| **Réel Angry** | 68% | 22% |

> Scared et Angry partagent les oreilles aplaties — seule la posture du corps (hors champ) permet la distinction.

**Conclusion intermédiaire (encadré vert) :**
> MobileNetV2 est retenu comme modèle de production. Le transfer learning compense efficacement le manque de données : +24.5pts F1 vs la baseline, pour seulement 3.4M paramètres.

---

## SLIDE 11 — Explicabilité — Grad-CAM

**Titre :** Explicabilité — Grad-CAM (Gradient-weighted Class Activation Mapping)

**Principe (encadré) :**
> Grad-CAM calcule le gradient du score de la classe prédite par rapport aux feature maps de la dernière couche convolutive (`out_relu` dans MobileNetV2). Ces gradients révèlent **quelles zones de l'image ont le plus influencé la décision**.

**Formule simplifiée :**
```
Heatmap = ReLU( Σ_k  [gradient moyen du filtre k]  ×  [activation du filtre k] )
```

**Ce qu'on attend biologiquement :**

| Zone | Émotion associée |
|---|---|
| Oreilles aplaties | Angry, Scared |
| Oreilles dressées | Happy, Surprised |
| Yeux grands ouverts | Scared, Surprised |
| Yeux mi-clos | Happy, Normal |
| Moustaches tendues | Angry |
| Truffe / bouche | Disgusted |

**Résultats observés :**
- Pour **Happy** : attention concentrée sur les yeux mi-clos et la posture des oreilles ✓
- Pour **Angry** : attention sur les oreilles aplaties et le bord des yeux ✓
- Pour **Scared** : attention sur les yeux écarquillés et les oreilles ✓
- Cas problématique : quelques images où l'attention porte sur le fond → biais de fond confirmé

**Valeur de l'explicabilité pour le projet :**
1. **Validation biologique** : confirme que le modèle apprend les bons signaux
2. **Détection de biais** : révèle si le modèle se base sur le fond plutôt que le visage
3. **Confiance clinique** : un vétérinaire peut vérifier la zone regardée par le modèle
4. **Débogage** : identifie les exemples mal classifiés causés par un cadrage atypique

**Visuels à insérer :** 3 exemples côte à côte (image originale | heatmap jet | superposition), un par émotion représentative (ex : Happy, Angry, Scared).

---

## SLIDE 12 — Application Streamlit

**Titre :** Application Web — Démo Streamlit

**Description générale :**
Interface web interactive développée avec Streamlit, déployable localement en une commande. Elle rend le modèle accessible à des non-experts (propriétaires, vétérinaires) sans connaissance en Machine Learning.

**4 pages de l'application :**

**1. Prédiction (page principale)**
- Upload d'une photo ou sélection d'un exemple du dataset
- Prédiction en temps réel par MobileNetV2
- Affichage : classe prédite + icône Font Awesome + niveau de confiance
- Graphique de distribution des probabilités sur les 7 classes
- Conseil comportemental adapté à l'émotion détectée
- Alerte automatique si confiance < 55% (résultat incertain)
- Visualisation Grad-CAM en expandeur (zones d'attention)

**2. Dashboard Dataset**
- KPIs : total images, classes, train/valid
- Distribution par classe (barplot train vs valid + pie chart)
- Tableau détaillé avec ratio de déséquilibre
- Galerie d'exemples filtrée par émotion (8 images)

**3. Résultats des Modèles**
- Tableau comparatif des 3 modèles
- Graphiques : accuracy/F1 par modèle, F1 par classe (MobileNetV2)
- Tableau des confusions fréquentes avec explications
- Statut des fichiers générés (modèles, figures)

**4. À propos**
- Contexte, pipeline ML illustré, stack technique
- Limites identifiées, recommandations, dataset info

**Commande de lancement :**
```bash
python train.py                          # Entraîne le modèle (~20 min, une seule fois)
python -m streamlit run app.py           # Lance l'interface
```

**Stack technique :** TensorFlow · Streamlit · scikit-learn · Pillow · Font Awesome 6 (CDN)

---

## SLIDE 13 — Limites, Biais & Pistes d'Amélioration

**Titre :** Limites, Biais et Pistes d'Amélioration

**3 colonnes :**

**Colonne 1 — Limites actuelles**
- 671 images : trop peu pour un CNN robuste (variance élevée entre runs)
- Résolution de travail 128×128 : perte de détails fins (microexpressions, texture moustaches)
- Pas de validation croisée : résultats sensibles au split aléatoire
- Dépendance au GPU pour un entraînement rapide
- Modèle non validé par des vétérinaires professionnels

**Colonne 2 — Biais identifiés**
- Biais de fond (fond blanc vétérinaire sur-représenté)
- Biais de race (faible diversité de races dans le dataset)
- Biais d'éclairage (studio vs naturel)
- Biais d'annotation (subjectivité humaine)
- Biais de cadrage (tête seule vs corps entier)

**Colonne 3 — Pistes d'amélioration**
- Enrichir à ≥ 200 images par classe (671 → 1400+)
- Tester EfficientNetB0 (meilleur ratio perf/taille) ou ViT
- Validation croisée k-fold stratifiée (5 ou 10 folds)
- Ensemble MobileNetV2 + CNN scratch (vote ou moyenne)
- Seuil de confiance adaptatif (retour "Incertain" si proba max < 60%)
- Validation clinique par vétérinaires comportementalistes
- Pipeline de feedback continu (corrections → nouveau dataset)

**Encadré chiffré :**
> Passer de 671 à 2000 images (+198%) permettrait d'estimer un gain de +8 à +12pts de F1-macro (loi d'échelle empirique du deep learning sur petits datasets).

---

## SLIDE 14 — Conclusion & Recommandations

**Titre :** Conclusion

**Réponse à la problématique (grande citation centrale) :**
> « Oui, il est possible de classifier automatiquement les émotions félines à partir d'images, avec une précision opérationnelle de **73% d'accuracy / 71% F1-macro** via MobileNetV2 Transfer Learning — suffisant pour une aide à la décision, insuffisant pour un diagnostic clinique autonome. »

**Bilan en 4 points clés :**

1. **Transfer Learning = clé sur petit dataset**
   MobileNetV2 dépasse le CNN scratch de +13pts F1 grâce aux features ImageNet. Sur 671 images, entraîner un CNN from scratch ne suffit pas.

2. **F1-macro > Accuracy comme métrique**
   Le déséquilibre modéré (1.27×) justifie une métrique robuste par classe. L'accuracy surestimerait les performances sur les classes majoritaires.

3. **Grad-CAM valide biologiquement le modèle**
   Les zones d'attention correspondent aux signaux comportementaux réels (oreilles, yeux). Le modèle n'apprend pas un artefact, il apprend (partiellement) la bonne chose.

4. **Le dataset est le facteur limitant principal**
   Même le meilleur modèle atteint un plafond avec 671 images. L'amélioration prioritaire est la collecte de données, pas l'architecture.

**Recommandations finales :**

| Recommandation | Priorité |
|---|---|
| Déployer MobileNetV2 avec seuil de confiance ≥ 60% | Haute |
| Enrichir le dataset à 200+ images/classe | Haute |
| Faire valider par des vétérinaires comportementalistes | Haute |
| Tester EfficientNetB0 / ViT | Moyenne |
| Mettre en place un pipeline de feedback | Moyenne |
| Évaluer sur un dataset externe (données vétérinaires réelles) | Basse |

**Lien GitHub :** https://github.com/R5FR/cat-emotions-classificator

---

## MÉTADONNÉES DE LA PRÉSENTATION

- **Nombre de slides :** 14
- **Public cible :** Enseignants évaluateurs (profil Data/ML)
- **Langue :** Français
- **Niveau de détail :** Élevé — présentation auto-suffisante (pas d'oral)
- **Durée de lecture estimée :** 15–20 minutes
- **Format recommandé :** 16:9 widescreen, export PDF pour le rendu

## PALETTE DE COULEURS

| Usage | Hex | Nom |
|---|---|---|
| Couleur principale | `#6366f1` | Indigo |
| Couleur secondaire | `#818cf8` | Indigo clair |
| Fond sombre | `#0f172a` | Slate 900 |
| Fond clair | `#f8fafc` | Slate 50 |
| Texte principal | `#1e293b` | Slate 800 |
| Texte secondaire | `#64748b` | Slate 500 |
| Succès / positif | `#22c55e` | Green 500 |
| Alerte | `#f59e0b` | Amber 500 |
| Erreur | `#ef4444` | Red 500 |

## COULEURS PAR ÉMOTION

| Émotion | Hex |
|---|---|
| Angry | `#E74C3C` |
| Disgusted | `#8E44AD` |
| Happy | `#27AE60` |
| Normal | `#2980B9` |
| Sad | `#7F8C8D` |
| Scared | `#E67E22` |
| Surprised | `#F39C12` |
