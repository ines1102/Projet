# Projet : Analyse EEG et PERCLOS pour Détection de Fatigue

Ce projet vise à effectuer le prétraitement des données EEG (électroencéphalogramme) et des labels PERCLOS (pourcentage du temps où les yeux sont fermés, utilisé pour évaluer l'état de vigilance) dans le but de préparer les données pour un modèle de classification. Le processus inclut le filtrage des données EEG, leur segmentation en fenêtres temporelles, l'extraction de caractéristiques et la création de modèles de prédiction.

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
Le script preprocessing.py permet de :
- Charger et filtrer les données EEG : Applique un filtre notch (50 Hz) pour supprimer les interférences électriques et un filtre passe-bande entre 1 et 50 Hz.
- Segmenter les données EEG : Divise les données EEG en segments de taille fixe (par défaut 3 secondes) avec un chevauchement de 50%.
- Associer les segments aux labels PERCLOS : Les segments EEG sont associés aux labels PERCLOS extraits des fichiers .mat.
- Sauvegarder les segments et labels : Les segments EEG et les labels PERCLOS sont sauvegardés sous forme de fichiers .pkl.

**feature_extraction.py :**
Le script feature_extraction.py est utilisé pour extraire des caractéristiques des segments EEG. Ces caractéristiques sont ensuite utilisées pour entraîner des modèles de machine learning.

**model_training.py :**
Le script model_training.py permet d’entraîner et de valider des modèles sur les caractéristiques extraites des données EEG. Il enregistre le modèle entraîné dans le fichier model.pkl.

**matrice_correlation.py**
Le projet génère diverses visualisations pour aider à l’analyse des données et des modèles, comme la matrice de corrélation des caractéristiques (matrice_correlation.py) et des graphiques des performances des modèles.

**analysis.py :**
Le script analysis.py permet d’analyser les performances des modèles entraînés. Il génère des graphiques et des rapports, tels que la matrice de confusion (matrice_confusion.py) et des visualisations des erreurs résiduelles.

## Installation

### 1. Prérequis

## Prérequis

Pour exécuter ce projet, vous aurez besoin des bibliothèques suivantes :

- `mne` : Pour charger et traiter les données EEG.
- `scipy` : Pour manipuler les fichiers PERCLOS et effectuer des calculs scientifiques.
- `numpy` : Pour manipuler les tableaux de données.
- `pickle` : Pour sérialiser les segments EEG et les labels.
- `matplotlib` et `seaborn` : Pour les visualisations.

Installez-les avec :

```bash 
pip3 install mne scipy numpy pickle matplotlib seaborn
```

## Utilisation

### Étape 1 : Prétraitement des données

Exécutez le script **preprocessing.py** pour :
1. Charger les fichiers .edf (EEG) et .mat (PERCLOS).
2. Appliquer les filtres (notch et passe-bande).
3. Découper les EEG en fenêtres de 3 secondes avec recouvrement.
4. Associer les fenêtres EEG aux labels PERCLOS.

#### Commande d’exécution
```bash
python3 preprocessing.py --lab_data "../../VLA_VRW/lab/EEG" --lab_labels "../../VLA_VRW/lab/perclos" \
                       --real_data "../../VLA_VRW/real/EEG" --real_labels "../../VLA_VRW/real/perclos" \
                       --output "processed/"
```

#### Arguments

- --lab_data : Dossier contenant les fichiers EEG en laboratoire (format .edf).
- --lab_labels : Dossier contenant les fichiers PERCLOS en laboratoire (format .mat).
- --real_data : Dossier contenant les fichiers EEG en situation réelle (format .edf).
- --real_labels : Dossier contenant les fichiers PERCLOS en situation réelle (format .mat).
- --output : Dossier où les segments et labels traités seront sauvegardés sous format .pkl.

Cette commande prétraitera les fichiers EEG et PERCLOS, segmentera les données et sauvegardera les segments et labels traités dans le dossier Github/Projet/processed/.

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

Voici l'organisation des fichiers et dossiers du projet :
```bash
📂 BCI
├── VLA_VRW/
│   ├── lab/
│   │   ├── EEG/            # 20 fichiers .edf représentant les données EEG en laboratoire.
│   │   ├── perclos/        # 20 fichiers .mat représentant les labels PERCLOS.
│   ├── real/
│   │   ├── EEG/            # 14 fichiers .edf représentant les données EEG en situation réelle.
│   │   ├── perclos/        # 14 fichiers .mat représentant les labels PERCLOS.
├── Github/
│   ├── Projet/
│   │   ├── processed/
│   │   │   ├── lab_segments.pkl          # Données EEG segmentées (laboratoire).
│   │   │   ├── reel_segments.pkl         # Données EEG segmentées (réel).
│   │   ├── features/
│   │   │   ├── lab_features.pkl          # Caractéristiques extraites des données laboratoire.
│   │   │   ├── reel_features.pkl         # Caractéristiques extraites des données réelles.
│   │   ├── results/
│   │   │   ├── model.pkl                # Modèle entraîné.
│   │   │   ├── results.json             # Résultats des performances des modèles.
│   │   ├── preprocessing.py             # Script de prétraitement des données brutes.
│   │   ├── feature_extraction.py        # Script pour extraire les caractéristiques des EEG.
│   │   ├── matrice_correlation.py       # Script pour calculer la matrice de corrélation des caractéristiques.
│   │   ├── model_training.py            # Script d’entraînement et de validation des modèles.
│   │   ├── analysis.py                  # Script pour l’analyse des résultats et des performances.
│   │   ├── README.md                    # Documentation détaillée du projet.
│   ├── correlation_0.4.png              # Visualisation de la corrélation des caractéristiques avec un seuil de 0.4.
│   ├── erreurs_residuelles.png          # Visualisation des erreurs résiduelles du modèle.
│   ├── features_importants.png          # Visualisation des caractéristiques les plus importantes pour le modèle.
│   ├── predictions_valeurs-reelles.png  # Visualisation des prédictions par rapport aux valeurs réelles.
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






Visualisations

Le projet génère plusieurs visualisations utiles pour évaluer les performances du modèle :
- Matrice de corrélation des caractéristiques (correlation_0.4.png).
- Erreurs résiduelles (erreurs_residuelles.png).
- Caractéristiques les plus importantes (features_importants.png).
- Comparaison des prédictions et des valeurs réelles (predictions_valeurs-reelles.png).

Ces graphiques peuvent être utilisés pour interpréter les résultats et ajuster les paramètres du modèle.

Conclusion

Ce projet permet de traiter et d’analyser les données EEG et les labels PERCLOS pour développer un modèle de classification basé sur les caractéristiques extraites des signaux EEG. Il est conçu pour être facilement extensible et adaptable à de nouvelles données et objectifs de recherche.

N'hésitez pas à personnaliser ou étendre cette documentation en fonction des détails spécifiques de votre projet.