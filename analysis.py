import pickle
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_features(labo_path, real_path):
    with open(labo_path, 'rb') as f:
        labo_features, _ = pickle.load(f)
    
    with open(real_path, 'rb') as f:
        real_features, _ = pickle.load(f)
    
    # Comparaison des distributions
    for i in range(labo_features.shape[1]):
        plt.figure()
        sns.kdeplot(labo_features[:, i], label='Labo', fill=True)
        sns.kdeplot(real_features[:, i], label='Réel', fill=True)

        plt.title(f'Caractéristique {i+1}')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labo', type=str, required=True, help="Caractéristiques labo")
    parser.add_argument('--reel', type=str, required=True, help="Caractéristiques réel")
    args = parser.parse_args()
    
    analyze_features(args.labo, args.reel)
