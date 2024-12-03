import os
import mne
import numpy as np
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.io as pio

# Chemin vers ton fichier EEG (FIF ou EDF)
file_path = "/Users/mac/Documents/ITS/S5/Dispositif médical (Labiod)/Projet/VLA_VRW/lab/EEG/edf_cleaned/1_cleaned.edf"  # Remplace par ton fichier

# Charger le fichier EEG
if file_path.endswith('.fif'):
    raw_eeg = mne.io.read_raw_fif(file_path, preload=True)
elif file_path.endswith('.edf'):
    raw_eeg = mne.io.read_raw_edf(file_path, preload=True)
else:
    raise ValueError("Format de fichier non pris en charge : uniquement .fif et .edf")

# Supprimer la colonne "EEG CM - Pz" et "Trigger" si elles sont présentes
channels_to_drop = [ch for ch in ["EEG CM - Pz", "Trigger"] if ch in raw_eeg.ch_names]
if channels_to_drop:
    raw_eeg.drop_channels(channels_to_drop)

# Extraire les données EEG (tous les canaux restants) et les informations de canaux
eeg_data = raw_eeg.get_data()
channel_names = raw_eeg.ch_names  # Noms des capteurs/canaux
sfreq = int(raw_eeg.info['sfreq'])  # Fréquence d'échantillonnage

# Durée d'affichage en secondes
duration = 10  # Affiche 10 secondes de données

# Sélection des échantillons à afficher (selon la durée spécifiée)
samples_to_plot = int(sfreq * duration)
time = np.linspace(0, duration, samples_to_plot)

# Création de sous-graphiques pour chaque capteur EEG
fig = sp.make_subplots(rows=len(channel_names), cols=1, shared_xaxes=True, vertical_spacing=0.02)

# Boucle pour tracer chaque capteur EEG dans un sous-graphique distinct
for i, channel_name in enumerate(channel_names):
    fig.add_trace(
        go.Scatter(x=time, y=eeg_data[i, :samples_to_plot], mode='lines', name=channel_name),
        row=i+1, col=1
    )
    # Ajouter le nom du capteur comme titre pour chaque sous-graphique
    fig.update_yaxes(title_text=channel_name, row=i+1, col=1)

# Mettre à jour la mise en forme du graphique
fig.update_layout(
    height=300 * len(channel_names),  # Ajuster la hauteur pour chaque capteur
    title="Affichage individuel des capteurs EEG",
    xaxis_title="Temps (secondes)",
    yaxis_title="Amplitude (µV)",
    showlegend=False,
    template="plotly_white"
)

# Afficher la figure dans ton navigateur (ou dans Jupyter si tu l'utilises)
pio.show(fig)