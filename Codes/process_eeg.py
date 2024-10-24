import os
import mne
import numpy as np

# Dossier contenant les fichiers EDF à traiter
edf_folder = "/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/Projet/VLA_VRW/real/EEG/edf"
output_folder = "/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/Projet/VLA_VRW/real/EEG/fif_process"

# Liste des fichiers .edf dans le dossier
edf_files = [f for f in os.listdir(edf_folder) if f.endswith('.edf')]

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_folder, exist_ok=True)

# Fonction pour supprimer les valeurs aberrantes
def remove_outliers(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    threshold = 3 * std_dev
    # Remplacer les valeurs aberrantes par NaN
    data_cleaned = np.where(np.abs(data - mean) > threshold, np.nan, data)
    return data_cleaned

# Fonction pour filtrer les données EEG à un seuil de 50 microvolts
def filter_to_50_microvolts(data):
    # Remplacer les valeurs supérieures à 50 microvolts par NaN
    data_filtered = np.where(np.abs(data) > 50, np.nan, data)
    return data_filtered

# Boucle pour traiter chaque fichier EDF
for file in edf_files:
    file_path = os.path.join(edf_folder, file)
    print(f"Traitement du fichier : {file}")

    # Charger le fichier EDF
    raw_eeg = mne.io.read_raw_edf(file_path, preload=True)

    # Supprimer la colonne "EEG CM - Pz" si elle est présente
    if "EEG CM - Pz" in raw_eeg.ch_names:
        raw_eeg.drop_channels(["EEG CM - Pz"])

    # Extraire les données EEG (tous les canaux)
    eeg_data = raw_eeg.get_data()

    # 1. Appliquer la suppression des valeurs aberrantes pour chaque canal
    eeg_data_cleaned = np.apply_along_axis(remove_outliers, 1, eeg_data)

    # 2. Filtrer les données à 50 microvolts
    eeg_data_filtered = np.apply_along_axis(filter_to_50_microvolts, 1, eeg_data_cleaned)

    # Remplacer les données originales par les données nettoyées et filtrées
    raw_eeg._data = eeg_data_filtered

    # Sauvegarder le fichier EDF nettoyé et filtré au format .fif
    output_file = os.path.join(output_folder, file.replace(".edf", "_cleaned_filtered.fif"))
    raw_eeg.save(output_file, overwrite=True)

    # Optionnel : afficher un message pour chaque fichier traité
    print(f"Fichier nettoyé et filtré sauvegardé : {output_file}")

# Message final
print("Traitement terminé pour tous les fichiers .edf du dossier.")