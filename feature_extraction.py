import numpy as np
import pickle
from scipy.stats import skew, kurtosis
import argparse
import os

def extract_features(segments):
    """
    Extrait les caractéristiques temporelles et fréquentielles des segments EEG.
    """
    features = []
    for segment in segments:
        # Caractéristiques temporelles : moyenne, variance, asymétrie, aplatissement
        mean = np.mean(segment, axis=1)
        variance = np.var(segment, axis=1)
        skewness = skew(segment, axis=1)
        kurt = kurtosis(segment, axis=1)

        # Caractéristiques fréquentielles : puissance dans les bandes alpha et beta
        fft_vals = np.fft.rfft(segment, axis=1)
        fft_freqs = np.fft.rfftfreq(segment.shape[1], 1/300)  # Fréquence d'échantillonnage de 300 Hz

        alpha_band = (8 <= fft_freqs) & (fft_freqs <= 13)
        beta_band = (14 <= fft_freqs) & (fft_freqs <= 30)

        alpha_power = np.sum(np.abs(fft_vals[:, alpha_band])**2, axis=1)
        beta_power = np.sum(np.abs(fft_vals[:, beta_band])**2, axis=1)

        # Concaténation des caractéristiques
        feature_vector = np.concatenate([mean, variance, skewness, kurt, alpha_power, beta_power])
        features.append(feature_vector)

    return np.array(features)

def process_features(input_path, output_path):
    """
    Charge les segments EEG et labels, extrait les caractéristiques, et sauvegarde les résultats.
    """
    # Création du dossier de sortie si nécessaire
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # Chargement des segments EEG et des labels
    with open(input_path, 'rb') as f:
        segments, labels = pickle.load(f)

    print(f"Extraction des caractéristiques pour {len(segments)} segments...")
    features = extract_features(segments)

    # Sauvegarde des caractéristiques et des labels
    with open(output_path, 'wb') as f:
        pickle.dump((features, labels), f)

    print(f"Caractéristiques sauvegardées dans {output_path}")

if __name__ == "__main__":
    # Gestion des arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Extraction des caractéristiques EEG")
    parser.add_argument('--input', type=str, required=True, help="Chemin du fichier d'entrée (segments EEG)")
    parser.add_argument('--output', type=str, required=True, help="Chemin du fichier de sortie (caractéristiques)")
    args = parser.parse_args()

    # Exécution de l'extraction des caractéristiques
    process_features(args.input, args.output)