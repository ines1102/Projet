import os
import mne
import pandas as pd

# Dossier contenant les fichiers .edf
input_folder = r"/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/Projet/VLA_VRW/real/EEG"

# Dossier de sortie pour les fichiers .csv
output_folder = r"/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/Projet/VLA_VRW/real/EEG/csv"

# Liste des fichiers dans le dossier .edf
edf_files = [f for f in os.listdir(input_folder) if f.endswith('.edf')]

# Conversion de chaque fichier .edf en .csv
for edf_file in edf_files:
    # Lecture du fichier .edf avec MNE
    edf_path = os.path.join(input_folder, edf_file)
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    
    # Extraction des données en tant que DataFrame
    data, times = raw.get_data(return_times=True)
    df = pd.DataFrame(data.T, columns=raw.ch_names)  # Transposition pour avoir un format propre
    
    # Ajout de la colonne du temps (timestamps)
    df['time'] = times
    
    # Sauvegarde en .csv dans le dossier de sortie
    csv_filename = edf_file.replace('.edf', '.csv')
    df.to_csv(os.path.join(output_folder, csv_filename), index=False)

print("Conversion terminée pour tous les fichiers .edf.")