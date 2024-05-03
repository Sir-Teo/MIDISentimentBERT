import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MidiDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'r') as file:
            self.data = json.load(file)
        
        self.abc_contents = []
        self.labels = []

        for item in self.data:
            abc_content = item['abc_content']

            self.abc_contents.append(abc_content)

            valence_score = 0.8 if item['Valence'] == 'High' else 0.3
            arousal_score = 0.8 if item['Arousal'] == 'High' else 0.3
            self.labels.append([valence_score, arousal_score])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        midi_features = self.abc_contents[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return midi_features, label

if __name__ == '__main__':
    # Example usage
    train_data = MidiDataset('data/prompts.json')
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)