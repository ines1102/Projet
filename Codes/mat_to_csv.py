import os
from scipy.io import loadmat
import pandas as pd

# Dossier contenant les fichiers .mat
input_folder = r"/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/Projet/VLA_VRW/real/perclos"

# Dossier de sortie pour les fichiers .csv
output_folder = r"/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/Projet/VLA_VRW/real/perclos/csv"

# Liste des fichiers dans le dossier .mat
mat_files = [f for f in os.listdir(input_folder) if f.endswith('.mat')]

# Conversion de chaque fichier .mat en .csv
for mat_file in mat_files:
    # Chargement du fichier .mat
    data = loadmat(os.path.join(input_folder, mat_file))
    
    # Extraction des données PERCLOS (ou autre clé si nécessaire)
    x = data["perclos"]  # Adaptez la clé si nécessaire
    
    # Conversion en DataFrame
    df = pd.DataFrame(x)
    
    # Sauvegarde en .csv dans le dossier de sortie
    csv_filename = mat_file.replace('.mat', '.csv')
    df.to_csv(os.path.join(output_folder, csv_filename), index=False)

print("Conversion terminée pour tous les fichiers .mat.")