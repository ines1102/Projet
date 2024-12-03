import os
import mne
import numpy as np
import scipy.io
import argparse
import pickle

def load_and_filter_eeg(file_path):
    # Chargement des données EEG
    raw = mne.io.read_raw_edf(file_path, preload=True)
    # Filtrage (notch et passe-bande 1-50 Hz)
    raw.notch_filter(freqs=50)
    raw.filter(1, 50)
    return raw

def segment_data(raw, window_size=3, overlap=1.5):
    fs = raw.info['sfreq']
    window_samples = int(window_size * fs)
    step_samples = int((window_size - overlap) * fs)
    data, times = raw[:, :]
    
    segments = []
    for start in range(0, data.shape[1] - window_samples + 1, step_samples):
        segment = data[:, start:start + window_samples]
        segments.append(segment)
    
    return np.array(segments)

def associate_with_perclos(segments, perclos_file):
    perclos_data = scipy.io.loadmat(perclos_file)['perclos']
    labels = []
    num_segments = len(segments)
    num_perclos = len(perclos_data)
    
    assert num_segments == num_perclos, "Le nombre de segments et de labels PERCLOS doit être identique"
    
    for i in range(num_segments):
        labels.append(perclos_data[i])
    
    return labels

def process_data(input_folder, output_folder):
    for data_type in ['labo', 'reel']:
        eeg_folder = os.path.join(input_folder, data_type, 'EEG')
        perclos_folder = os.path.join(input_folder, data_type, 'perclos')
        
        segments_all = []
        labels_all = []
        
        for filename in os.listdir(eeg_folder):
            if filename.endswith('.edf'):
                file_path = os.path.join(eeg_folder, filename)
                perclos_path = os.path.join(perclos_folder, filename.replace('.edf', '.mat'))
                
                raw = load_and_filter_eeg(file_path)
                segments = segment_data(raw)
                labels = associate_with_perclos(segments, perclos_path)
                
                segments_all.extend(segments)
                labels_all.extend(labels)
        
        output_path = os.path.join(output_folder, f"{data_type}_segments.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump((segments_all, labels_all), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Dossier contenant les données brutes")
    parser.add_argument('--output', type=str, required=True, help="Dossier de sortie pour les données traitées")
    args = parser.parse_args()
    
    process_data(args.input, args.output)
