# Projet : Analyse EEG et PERCLOS pour Détection de Fatigue

Ce projet analyse les signaux EEG collectés dans deux contextes (laboratoire et réel) et les associe aux labels PERCLOS correspondants. L’objectif est de comparer les performances des modèles prédictifs entre ces deux contextes et d’identifier les différences éventuelles.

## Structure du projet

### 1. Données
**Données laboratoire :**
- 20 fichiers .edf contenant les signaux EEG.
- 20 fichiers .mat contenant les labels PERCLOS (un fichier .mat par participant).

**Données réelles :**
- 14 fichiers .edf contenant les signaux EEG.
- 14 fichiers .mat contenant les labels PERCLOS (un fichier .mat par participant).

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
2.	Appliquer les filtres (notch et passe-bande).
3.	Découper les EEG en fenêtres de 3 secondes avec recouvrement.
4.	Associer les fenêtres EEG aux labels PERCLOS.

```bash
python preprocessing.py --lab_data "data/labo/" --lab_labels "data/labo_perclos.mat" \
                        --real_data "data/reel/" --real_labels "data/reel_perclos.mat" \
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
│   ├── labo/               
│   │   ├── EEG/            # 20 fichiers .edf (EEG labo)
│   │   ├── perclos/        # 20 fichiers .mat (PERCLOS labo)
│   │   │   ├── 1.mat
│   │   │   ├── 2.mat
│   │   │   ├── ...
│   ├── real/               
│   │   ├── EEG/            # 14 fichiers .edf (EEG réel)
│   │   ├── perclos/        # 14 fichiers .mat (PERCLOS réel)
│   │   │   ├── 1.mat
│   │   │   ├── 2.mat
│   │   │   ├── ...
├── processed/
│   ├── labo_segments.pkl   # Données segmentées (labo)
│   ├── reel_segments.pkl   # Données segmentées (réel)
├── features/
│   ├── labo_features.pkl   # Caractéristiques extraites (labo)
│   ├── reel_features.pkl   # Caractéristiques extraites (réel)
├── results/
│   ├── model_performance.json  # Résultats des modèles
├── preprocessing.py       # Script de prétraitement
├── feature_extraction.py  # Script d'extraction des caractéristiques
├── model_training.py      # Script d'entraînement des modèles
├── analysis.py            # Script d'analyse comparative
├── README.md              # Documentation du projet
```

## Résultats attendus

### 1. Comparaison labo vs réel
Différences entre les caractéristiques extraites (variance, puissance dans les bandes alpha/beta).
Visualisation des distributions via des boxplots ou histogrammes.

### 2. Performance des modèles
Évaluer les performances des modèles sur données labo et réelles :
- Métriques : Précision, RMSE, etc.
- Les performances sont-elles comparables entre les deux contextes ?

## Contact

Pour toute question, contactez :
- **ALEMANY Clarisse**
- **ASSOUANE Inès**
- **BOUKHEDRA Khitam**