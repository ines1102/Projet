import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Chemin du dossier contenant les fichiers CSV
input_folder = "/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/Projet/VLA_VRW/real/perclos/csv"  # Remplacez par le chemin de votre dossier

# Lister tous les fichiers CSV dans le dossier
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# Boucle sur chaque fichier CSV
for file in csv_files:
    # Chemin complet du fichier CSV
    file_path = os.path.join(input_folder, file)
    
    # Charger le fichier CSV sans en-tête
    df_perclos = pd.read_csv(file_path, header=None)
    
    # Donner un nom à la colonne
    df_perclos.columns = ['perclos']
    
    # Vérifier s'il existe des valeurs aberrantes (hors de la plage 0-1)
    aberrant_values = df_perclos[(df_perclos['perclos'] < 0) | (df_perclos['perclos'] > 1)]
    
    # Plot des valeurs PERCLOS pour chaque fichier
    plt.figure(figsize=(10,6))
    plt.plot(df_perclos['perclos'], label='PERCLOS', color='blue')
    
    # Mettre en évidence les valeurs aberrantes
    if not aberrant_values.empty:
        plt.scatter(aberrant_values.index, aberrant_values['perclos'], color='red', label='Valeurs aberrantes')
    
    # Ajouter des détails au graphique
    plt.axhline(y=1, color='green', linestyle='--', label='Seuil supérieur (1)')
    plt.axhline(y=0, color='green', linestyle='--', label='Seuil inférieur (0)')
    plt.title(f'Visualisation des valeurs PERCLOS - {file}')
    plt.xlabel('Temps (échantillons)')
    plt.ylabel('Valeurs PERCLOS')
    plt.legend()
    plt.grid(True)
    
    # Afficher le graphique
    plt.show()