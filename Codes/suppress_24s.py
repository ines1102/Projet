import os
import mne
import pyedflib
import numpy as np

# Dossiers d'entrée et de sortie
edf_folder = "/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/Projet/VLA_VRW/real/EEG/edf"
output_folder = "/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/Projet/VLA_VRW/real/EEG/edf_cleaned"
os.makedirs(output_folder, exist_ok=True)

# Durée de la période initiale à supprimer (en secondes)
initial_duration = 24

# Liste des fichiers EDF
edf_files = [f for f in os.listdir(edf_folder) if f.endswith('.edf')]

for file in edf_files:
    file_path = os.path.join(edf_folder, file)
    print(f"Traitement du fichier : {file}")

    # Charger le fichier EDF avec MNE
    raw_eeg = mne.io.read_raw_edf(file_path, preload=True)
    
    # Supprimer les 24 premières secondes
    raw_eeg.crop(tmin=initial_duration)

    # Extraire les données et informations de canaux
    data = raw_eeg.get_data() * 1e6  # Convertir en microvolts
    sample_rate = int(raw_eeg.info['sfreq'])  # Fréquence d'échantillonnage

    # Préparer les métadonnées des canaux pour le fichier EDF
    channel_info = []
    for i, ch_info in enumerate(raw_eeg.info['chs']):
        # Paramètres de chaque canal
        physical_min = data[i, :].min()
        physical_max = data[i, :].max()

        # Spécifier les valeurs pour le canal 'Trigger' pour éviter les erreurs
        if ch_info['ch_name'] == "Trigger" and physical_min == 0 and physical_max == 0:
            physical_min = -1.0
            physical_max = 1.0

        channel = {
            'label': ch_info['ch_name'],  # Nom du canal
            'dimension': 'uV',            # Unité physique
            'sample_rate': sample_rate,   # Fréquence d'échantillonnage
            'physical_min': physical_min,  # Valeur physique minimale
            'physical_max': physical_max,  # Valeur physique maximale
            'digital_min': -32768,        # Minimum numérique
            'digital_max': 32767,         # Maximum numérique
            'transducer': '',             # Type de transducteur
            'prefilter': ''               # Préfiltrage
        }
        channel_info.append(channel)

    # Chemin de sauvegarde du fichier EDF nettoyé
    output_file = os.path.join(output_folder, file.replace(".edf", "_cleaned.edf"))

    # Écrire le fichier EDF avec pyedflib
    with pyedflib.EdfWriter(output_file, len(channel_info), file_type=pyedflib.FILETYPE_EDFPLUS) as edf_writer:
        # Définir les informations de chaque canal dans le fichier EDF
        edf_writer.setSignalHeaders(channel_info)
        
        # Appliquer la fréquence à chaque canal individuellement
        for idx, channel in enumerate(channel_info):
            edf_writer.setSamplefrequency(idx, sample_rate)  # Assigner la fréquence pour chaque canal

        # Écrire les données des canaux
        edf_writer.writeSamples(data)

    print(f"Fichier nettoyé sauvegardé : {output_file}")

print("Traitement terminé pour tous les fichiers .edf du dossier.")