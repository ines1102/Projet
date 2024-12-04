import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
features_path = "features/reel_features.pkl"

with open(features_path, 'rb') as file:
    features, labels = pickle.load(file)

# Convertir en DataFrame pour une manipulation facile
features_df = pd.DataFrame(features)
labels_series = pd.Series(labels, name="Subject")

# Ajouter les labels pour analyse inter-sujets
features_df['Subject'] = labels_series

# Fonction pour calculer la corrélation inter-sujets
def inter_subject_correlation(features_df):
    """
    Calcule la corrélation inter-sujets pour chaque feature.
    """
    unique_subjects = features_df['Subject'].unique()
    inter_corr = []

    print("\nCorrélation inter-sujets pour chaque feature :")
    for feature in features_df.columns[:-1]:  # Exclure la colonne 'Subject'
        # Récupérer les données par sujet
        feature_values = []
        for subject in unique_subjects:
            subject_data = features_df[features_df['Subject'] == subject][feature].values
            if len(subject_data) > 0:
                feature_values.append(subject_data[:min(len(subject_data), 100)])  # Uniformiser à 100 valeurs max

        # Convertir en tableau numpy avec dimensions égales
        feature_values = np.array([np.pad(fv, (0, max(0, 100 - len(fv))), 'constant', constant_values=np.nan)
                                    for fv in feature_values])

        # Calculer la matrice de corrélation
        feature_corr = np.ma.corrcoef(np.ma.masked_invalid(feature_values), rowvar=False)
        # Filtrer les valeurs masquées (MaskedConstant) avant de calculer la moyenne
        mean_corr = np.ma.mean(feature_corr)

        # Vérifiez si `mean_corr` est un nombre valide, sinon attribuez une valeur par défaut
        if np.ma.is_masked(mean_corr) or np.isnan(mean_corr):
            mean_corr = 0.0

        inter_corr.append((feature, mean_corr))
        print(f"Feature: {feature} - Corrélation inter-sujets moyenne : {mean_corr:.2f}")

    return inter_corr

# Calcul des corrélations inter-sujets
inter_subject_results = inter_subject_correlation(features_df)

# Visualisation des corrélations inter-sujets
def plot_inter_subject_correlation(inter_corr):
    """
    Affiche un graphique des corrélations inter-sujets par feature.
    """
    features = [item[0] for item in inter_corr]
    correlations = [item[1] for item in inter_corr]

    # Filtrer les valeurs nulles ou `nan`
    valid_features = [features[i] for i in range(len(correlations)) if not np.isnan(correlations[i])]
    valid_correlations = [correlations[i] for i in range(len(correlations)) if not np.isnan(correlations[i])]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=valid_features, y=valid_correlations, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title("Corrélation inter-sujets par feature")
    plt.ylabel("Corrélation moyenne")
    plt.xlabel("Features")
    plt.tight_layout()
    plt.show()

# Afficher les résultats
plot_inter_subject_correlation(inter_subject_results)
