# Projet : Analyse EEG et PERCLOS pour Détection de Fatigue

Ce projet a pour objectif d’analyser les signaux EEG collectés dans deux contextes distincts (laboratoire et réel) et de les associer aux labels PERCLOS correspondants. L’objectif principal est de comparer les performances des modèles prédictifs entre ces deux contextes afin d’identifier et d’expliquer les différences potentielles.

## Structure du projet

### 1. Données
**Données laboratoire :**
- EEG : 20 fichiers .edf contenant les signaux EEG.
- PERCLOS : 20 fichiers .mat contenant les labels (un fichier par participant).

**Données réelles :**
- EEG : 14 fichiers .edf contenant les signaux EEG.
- PERCLOS : 14 fichiers .mat contenant les labels (un fichier par participant).

### 2. Scripts

**preprocessing.py :**
- Chargement et filtrage des EEG.
- Découpage en fenêtres avec chevauchement.
- Association avec les labels PERCLOS.

**feature_extraction.py :**
- Extraction des caractéristiques temporelles et fréquentielles.

**model_training.py :**
- Entraînement et évaluation des modèles sur les données EEG et PERCLOS.

**analysis.py :**
- Comparaison des caractéristiques entre labo et réel.
- Visualisation des distributions.

## Installation

### 1. Prérequis

Assurez-vous d’avoir Python et les bibliothèques suivantes :
```bash 
mne, numpy, scipy, matplotlib, seaborn, scikit-learn.
```

Installez-les avec :

```bash 
pip3 install mne numpy scipy matplotlib seaborn scikit-learn
```

## Utilisation

### Étape 1 : Prétraitement des données

Exécutez le script **preprocessing.py** pour :
1. Charger les fichiers .edf (EEG) et .mat (PERCLOS).
2. Appliquer les filtres (notch et passe-bande).
3. Découper les EEG en fenêtres de 3 secondes avec recouvrement.
4. Associer les fenêtres EEG aux labels PERCLOS.

```bash
python3 preprocessing.py \
  --lab_data "../VLA_VRW/lab/EEG/" \
  --lab_labels "../VLA_VRW/lab/perclos/" \
  --real_data "../VLA_VRW/real/EEG/" \
  --real_labels "../VLA_VRW/real/perclos/" \
  --output "processed/"
```

### Étape 2 : Extraction des caractéristiques

Exécutez feature_extraction.py pour extraire des caractéristiques temporelles et fréquentielles :

```bash
python feature_extraction.py --input "processed/" --output "features/"
```

### Étape 3 : Entraînement et évaluation des modèles

Utilisez model_training.py pour entraîner les modèles sur les données labo et les tester sur les données réelles :

```bash
python model_training.py --train "features/labo_features.pkl" --test "features/reel_features.pkl" \
                         --output "results/model_performance.json"
```

### Étape 4 : Analyse comparative

Exécutez analysis.py pour visualiser les différences entre les contextes labo et réel :

```bash
python analysis.py --labo "features/labo_features.pkl" --reel "features/reel_features.pkl"
```

## Organisation des fichiers
```bash
📂 Projet_EEG
├── VLA_VRW/
│   ├── lab/               
│   │   ├── EEG/            # 20 fichiers .edf représentant les données EEG en laboratoire.
│   │   ├── perclos/        # 20 fichiers .mat représentant les labels PERCLOS.
│   ├── real/               
│   │   ├── EEG/            # 14 fichiers .edf représentant les données EEG en situation réelle.
│   │   ├── perclos/        # 14 fichiers .mat représentant les labels PERCLOS.
├── Github/
│   ├── processed/    
│   │   ├── labo_segments.pkl   # Données EEG segmentées (laboratoire).
│   │   ├── reel_segments.pkl   # Données EEG segmentées (réel).
│   ├── features/
│   │   ├── labo_features.pkl   # Caractéristiques extraites des données laboratoire.
│   │   ├── reel_features.pkl   # Caractéristiques extraites des données réelles.
│   ├── results/
│   │   ├── model_performance.json  # Résultats des performances des modèles.
│   ├── preprocessing.py       # Script de prétraitement des données brutes.
│   ├── feature_extraction.py  # Script pour extraire les caractéristiques des EEG.
│   ├── model_training.py      # Script d'entraînement et de validation des modèles.
│   ├── analysis.py            # Script pour l'analyse comparative entre labo et réel.
│   ├── README.md              # Documentation détaillée du projet.
```

## Résultats attendus

### 1. Comparaison labo vs réel
- Identifier les différences entre les caractéristiques extraites (variance, puissance dans les bandes alpha/beta).
- Visualiser les distributions des caractéristiques via des boxplots ou histogrammes.

### 2. Performance des modèles
- Évaluer les performances des modèles sur les données laboratoire et réelles.
- Métriques utilisées : Précision, RMSE, etc.
- Vérifier si les performances sont comparables entre les deux contextes.

## Contact

Pour toute question, contactez :
- **ALEMANY Clarisse**
- **ASSOUANE Inès**
- **BOUKHEDRA Khitam**