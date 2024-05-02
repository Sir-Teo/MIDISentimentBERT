import json
import os
import numpy as np
from feature_extraction import extract_midi_features

def preprocess_data(data_file, output_dir):
    with open(data_file, 'r') as file:
        data = json.load(file)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Preprocess each MIDI file
    for item in data:
        midi_path = item['midi_path']
        valence = item['valence']
        arousal = item['arousal']

        # Extract features from the MIDI file
        midi_features = extract_midi_features(midi_path)

        # Save the extracted features and labels
        feature_path = os.path.join(output_dir, f"{os.path.basename(midi_path)}.npy")
        np.save(feature_path, midi_features)

        # Save the labels in a separate file or append to an existing file
        label_path = os.path.join(output_dir, 'labels.csv')
        with open(label_path, 'a') as label_file:
            label_file.write(f"{os.path.basename(midi_path)},{valence},{arousal}\n")

    print("Data preprocessing completed.")

if __name__ == '__main__':
    data_file = 'path/to/your/data.json'
    output_dir = 'path/to/processed/data'
    preprocess_data(data_file, output_dir)