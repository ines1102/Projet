import numpy as np
import pickle
import argparse
from scipy.stats import skew, kurtosis

def extract_features(segments):
    features = []
    for segment in segments:
        # Caractéristiques temporelles : moyenne, variance, skewness, kurtosis
        mean = np.mean(segment, axis=1)
        variance = np.var(segment, axis=1)
        skewness = skew(segment, axis=1)
        kurt = kurtosis(segment, axis=1)

        # Caractéristiques fréquentielles : puissance dans les bandes alpha et beta
        fft_vals = np.fft.rfft(segment, axis=1)
        fft_freqs = np.fft.rfftfreq(segment.shape[1], 1/100)  # Exemple de fréquence d'échantillonnage

        alpha_band = (8 <= fft_freqs) & (fft_freqs <= 13)
        beta_band = (14 <= fft_freqs) & (fft_freqs <= 30)

        alpha_power = np.sum(np.abs(fft_vals[:, alpha_band])**2, axis=1)
        beta_power = np.sum(np.abs(fft_vals[:, beta_band])**2, axis=1)

        # Concaténation des caractéristiques
        feature_vector = np.concatenate([mean, variance, skewness, kurt, alpha_power, beta_power])
        features.append(feature_vector)
    
    return np.array(features)

def process_features(input_path, output_path):
    with open(input_path, 'rb') as f:
        segments, labels = pickle.load(f)
    
    features = extract_features(segments)
    
    with open(output_path, 'wb') as f:
        pickle.dump((features, labels), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Fichier d'entrée des segments")
    parser.add_argument('--output', type=str, required=True, help="Fichier de sortie des caractéristiques")
    args = parser.parse_args()
    
    process_features(args.input, args.output)
