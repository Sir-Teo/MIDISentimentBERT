import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import MidiSentimentModel
from dataset import MidiDataset

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    valence_predictions = []
    arousal_predictions = []
    valence_labels = []
    arousal_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            valence_labels_batch = batch['valence'].to(device)
            arousal_labels_batch = batch['arousal'].to(device)

            valence_outputs, arousal_outputs = model(input_ids, attention_mask)
            valence_loss = criterion(valence_outputs.squeeze(), valence_labels_batch)
            arousal_loss = criterion(arousal_outputs.squeeze(), arousal_labels_batch)
            loss = valence_loss + arousal_loss
            test_loss += loss.item()

            valence_predictions.extend(valence_outputs.squeeze().tolist())
            arousal_predictions.extend(arousal_outputs.squeeze().tolist())
            valence_labels.extend(valence_labels_batch.tolist())
            arousal_labels.extend(arousal_labels_batch.tolist())

    test_loss /= len(test_loader)
    valence_mae = mean_absolute_error(valence_labels, valence_predictions)
    arousal_mae = mean_absolute_error(arousal_labels, arousal_predictions)

    return test_loss, valence_mae, arousal_mae

def mean_absolute_error(labels, predictions):
    return sum(abs(l - p) for l, p in zip(labels, predictions)) / len(labels)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess data
    test_data = MidiDataset('path/to/test/data.json')
    test_loader = DataLoader(test_data, batch_size=16)

    # Load BERT tokenizer and model
    bert_config = 'path/to/bert/config.json'
    tokenizer = BertTokenizer.from_pretrained(bert_config)
    model = MidiSentimentModel(bert_config).to(device)
    model.load_state_dict(torch.load('path/to/trained/model.pt'))

    # Set loss function
    criterion = nn.MSELoss()

    # Evaluate the model
    test_loss, valence_mae, arousal_mae = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Valence MAE: {valence_mae:.4f}')
    print(f'Arousal MAE: {arousal_mae:.4f}')

if __name__ == '__main__':
    main()