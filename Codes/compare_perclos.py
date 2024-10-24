import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Chemin des dossiers contenant les fichiers CSV
lab_folder = "/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/Projet/VLA_VRW/lab/perclos/csv"  # Remplace par le chemin de ton dossier pour les données de laboratoire
real_folder = "/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/Projet/VLA_VRW/real/perclos/csv"  # Remplace par le chemin de ton dossier pour les données réelles

# Liste des fichiers dans chaque dossier
lab_files = [f for f in os.listdir(lab_folder) if f.endswith('.csv')]
real_files = [f for f in os.listdir(real_folder) if f.endswith('.csv')]

# Créer des DataFrames pour stocker toutes les données
lab_data_all = pd.DataFrame()
real_data_all = pd.DataFrame()

# Charger et concaténer les fichiers de laboratoire
for file in lab_files:
    file_path = os.path.join(lab_folder, file)
    df_lab = pd.read_csv(file_path, header=None)
    df_lab.columns = ['perclos']  # Ajuste selon le nom des colonnes de tes fichiers
    lab_data_all = pd.concat([lab_data_all, df_lab], ignore_index=True)

# Charger et concaténer les fichiers réels
for file in real_files:
    file_path = os.path.join(real_folder, file)
    df_real = pd.read_csv(file_path, header=None)
    df_real.columns = ['perclos']  # Ajuste selon le nom des colonnes de tes fichiers
    real_data_all = pd.concat([real_data_all, df_real], ignore_index=True)

# 1. Statistiques descriptives
print("Statistiques descriptives pour les données de laboratoire :")
print(lab_data_all.describe())
print("\nStatistiques descriptives pour les données réelles :")
print(real_data_all.describe())

# 2. Visualisation des distributions (Histogrammes)
plt.figure(figsize=(10,6))
plt.hist(lab_data_all['perclos'], bins=30, alpha=0.5, label='Laboratoire', color='blue', density=True)
plt.hist(real_data_all['perclos'], bins=30, alpha=0.5, label='Réel', color='orange', density=True)
plt.title('Comparaison des distributions PERCLOS: Lab vs Real')
plt.xlabel('Valeur PERCLOS')
plt.ylabel('Densité')
plt.legend()
plt.grid(True)
plt.show()

# 3. Test statistique (Mann-Whitney U)
stat, p_value = stats.mannwhitneyu(lab_data_all['perclos'], real_data_all['perclos'])
print(f"\nRésultat du test de Mann-Whitney U :")
print(f"Statistique U = {stat}, p-value = {p_value}")

if p_value < 0.05:
    print("Les distributions des valeurs PERCLOS en laboratoire et en réel sont significativement différentes.")
else:
    print("Il n'y a pas de différence significative entre les distributions des valeurs PERCLOS en laboratoire et en réel.")