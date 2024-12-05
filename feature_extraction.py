import numpy as np
import mne
from scipy.stats import pearsonr
import pickle
import os

def extract_features(segments):
    # Exemple d'extraction de caractéristiques simples : moyenne et variance pour chaque segment
    features = []
    for segment in segments:
        # Moyenne et variance de chaque segment EEG pour chaque canal
        mean_features = np.mean(segment, axis=1)
        var_features = np.var(segment, axis=1)
        features.append(np.concatenate([mean_features, var_features]))
    
    return np.array(features)

def load_and_process_data(input_folder):
    all_segments = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.pkl'):
            with open(os.path.join(input_folder, filename), 'rb') as f:
                segments, labels = pickle.load(f)
                all_segments.append(segments)
    return np.concatenate(all_segments, axis=0)

def compute_correlation_matrix(features):
    # Calcul de la matrice de corrélation entre toutes les caractéristiques
    corr_matrix = np.corrcoef(features, rowvar=False)
    return corr_matrix

def filter_features_based_on_correlation(features, correlation_matrix, threshold=0.4):
    # Garder les caractéristiques dont la corrélation est >= seuil
    selected_features = []
    num_features = features.shape[1]
    
    for i in range(num_features):
        for j in range(i + 1, num_features):
            if abs(correlation_matrix[i, j]) >= threshold:
                if i not in selected_features:
                    selected_features.append(i)
                if j not in selected_features:
                    selected_features.append(j)
    
    return features[:, selected_features]

def main(input_folder, output_folder):
    # Charger les segments EEG
    all_segments = load_and_process_data(input_folder)

    # Extraire les caractéristiques
    features = extract_features(all_segments)

    # Calculer la matrice de corrélation
    corr_matrix = compute_correlation_matrix(features)

    # Filtrer les caractéristiques en fonction de la corrélation
    filtered_features = filter_features_based_on_correlation(features, corr_matrix, threshold=0.4)

    # Sauvegarder les caractéristiques filtrées
    output_file = os.path.join(output_folder, "filtered_features.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(filtered_features, f)
    print(f"Caractéristiques filtrées sauvegardées dans {output_file}")

if __name__ == "__main__":
    input_folder = "processed/"  # Dossier contenant les fichiers de segments traités
    output_folder = "features/"  # Dossier où les caractéristiques filtrées seront sauvegardées
    main(input_folder, output_folder)