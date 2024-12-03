import os
import mne
import numpy as np
import scipy.io
import argparse
import pickle

def load_and_filter_eeg(file_path):
    print(f"Chargement du fichier EEG: {file_path}")
    # Chargement des données EEG
    raw = mne.io.read_raw_edf(file_path, preload=True)
    print(f"Nombre d'échantillons dans le fichier {file_path}: {raw.n_times}")
    # Filtrage (notch et passe-bande 1-50 Hz)
    raw.notch_filter(freqs=50)
    raw.filter(1, 50)
    return raw

def segment_data(raw, window_size=3, overlap=0.5):
    fs = raw.info['sfreq']
    window_samples = int(window_size * fs)
    step_samples = int(window_samples * (1 - overlap))
    data, times = raw[:, :]
    
    print(f"Fréquence d'échantillonnage: {fs}")
    print(f"Taille de la fenêtre: {window_samples} échantillons")
    print(f"Taille de l'étape: {step_samples} échantillons")
    print(f"Nombre total d'échantillons: {data.shape[1]}")
    
    segments = []
    for start in range(0, data.shape[1] - window_samples + 1, step_samples):
        segment = data[:, start:start + window_samples]
        segments.append(segment)
    
    print(f"Nombre de segments extraits: {len(segments)}")
    return np.array(segments)

def associate_with_perclos(segments, perclos_file):
    print(f"Chargement du fichier PERCLOS: {perclos_file}")
    perclos_data = scipy.io.loadmat(perclos_file)['perclos']
    perclos_data = perclos_data.flatten()

    print(f"Nombre de labels PERCLOS: {len(perclos_data)}")

    # Associer chaque segment avec le label PERCLOS correspondant
    labels = []
    for i in range(len(segments)):
        labels.append(perclos_data[i % len(perclos_data)])

    print(f"Nombre de labels associés: {len(labels)}")
    return np.array(labels)

def process_data(lab_eeg_path, lab_perclos_path, real_eeg_path, real_perclos_path, output_folder):
    # Création du dossier de sortie si nécessaire
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for data_type, eeg_folder, perclos_folder in [('lab', lab_eeg_path, lab_perclos_path), 
                                                  ('reel', real_eeg_path, real_perclos_path)]:
        print(f"\nTraitement des données {data_type}...")
        segments_all = []
        labels_all = []
        
        for filename in os.listdir(eeg_folder):
            if filename.endswith('.edf'):
                eeg_file_path = os.path.join(eeg_folder, filename)
                perclos_file_path = os.path.join(perclos_folder, filename.replace('.edf', '.mat'))
                
                # Vérification de l'existence du fichier PERCLOS correspondant
                if not os.path.exists(perclos_file_path):
                    print(f"Warning: Le fichier PERCLOS correspondant pour {filename} est introuvable.")
                    continue
                
                raw = load_and_filter_eeg(eeg_file_path)
                segments = segment_data(raw)
                
                if len(segments) == 0:
                    print(f"Warning: Aucun segment n'a été extrait pour le fichier {filename}.")
                    continue
                
                labels = associate_with_perclos(segments, perclos_file_path)
                
                segments_all.extend(segments)
                labels_all.extend(labels)
        
        # Vérification du nombre de segments avant la sauvegarde
        print(f"Nombre total de segments pour {data_type}: {len(segments_all)}")
        print(f"Nombre total de labels pour {data_type}: {len(labels_all)}")

        # Sauvegarde des segments et des labels dans un fichier .pkl
        output_path = os.path.join(output_folder, f"{data_type}_segments.pkl")

        print(f"Vérification avant sauvegarde - Nombre total de segments: {len(segments_all)}, Nombre total de labels: {len(labels_all)}")

        with open(output_path, 'wb') as f:
            pickle.dump((segments_all, labels_all), f)

        print(f"Données sauvegardées dans {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prétraitement des données EEG et PERCLOS")
    parser.add_argument('--lab_data', type=str, required=True, help="Dossier contenant les fichiers EEG de laboratoire")
    parser.add_argument('--lab_labels', type=str, required=True, help="Dossier contenant les fichiers PERCLOS de laboratoire")
    parser.add_argument('--real_data', type=str, required=True, help="Dossier contenant les fichiers EEG en conditions réelles")
    parser.add_argument('--real_labels', type=str, required=True, help="Dossier contenant les fichiers PERCLOS en conditions réelles")
    parser.add_argument('--output', type=str, required=True, help="Dossier de sortie pour les données traitées")
    args = parser.parse_args()
    
    process_data(args.lab_data, args.lab_labels, args.real_data, args.real_labels, args.output)
