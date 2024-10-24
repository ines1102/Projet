import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dossier contenant les fichiers .csv
input_folder = "/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/Projet/VLA_VRW/lab/EEG/csv"
output_folder = "/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/Projet/VLA_VRW/lab/EEG/csv_filtre/"

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_folder, exist_ok=True)

# Liste des fichiers dans le dossier .csv
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# Définir les colonnes EEG (sans Trigger, time et diff_EEG)
eeg_columns = ['EEG P3 - Pz', 'EEG C3 - Pz', 'EEG F3 - Pz', 'EEG Fz - Pz', 'EEG F4 - Pz', 
               'EEG C4 - Pz', 'EEG P4 - Pz', 'EEG Cz - Pz', 'EEG CM - Pz', 'EEG A1 - Pz', 
               'EEG Fp1 - Pz', 'EEG Fp2 - Pz', 'EEG T3 - Pz', 'EEG T5 - Pz', 'EEG O1 - Pz', 
               'EEG O2 - Pz']

# Seuil de 50 microvolts
threshold = 50  # µV

# Boucle sur chaque fichier .csv dans le dossier
for file in csv_files:
    # Charger le fichier CSV
    df = pd.read_csv(os.path.join(input_folder, file))
    
    # 1. Détection et suppression des valeurs aberrantes (> 3*écart-type) et limitation à 50 µV
    for col in eeg_columns:
        if col in df.columns:  # Vérifier si la colonne est présente
            mean = df[col].mean()
            std = df[col].std()
            
            # Remplacer les valeurs aberrantes (> 3*écart-type) et celles > 50 µV par NaN
            df[col] = np.where((df[col] > mean + 3*std) | (df[col] < mean - 3*std) | (np.abs(df[col]) > threshold), np.nan, df[col])
    
    # 2. Interpolation des valeurs manquantes après suppression des pics et filtrage
    df[eeg_columns] = df[eeg_columns].interpolate(method='linear')
    
    # 3. Supprimer la colonne 'EEG CM - Pz' si elle est présente
    if 'EEG CM - Pz' in df.columns:
        df = df.drop(columns=['EEG CM - Pz'])

    # 4. Sauvegarder les données nettoyées avec un nouveau nom de fichier
    output_file = os.path.join(output_folder, file.replace(".csv", "_cleaned.csv"))
    df.to_csv(output_file, index=False)

    # Optionnel : afficher un message pour chaque fichier traité
    print(f"Fichier traité et sauvegardé : {output_file}")

# Optionnel : message final
print("Traitement terminé pour tous les fichiers .csv du dossier.")