import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import MidiSentimentModel
from dataset import MidiDataset

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        valence_labels = batch['valence'].to(device)
        arousal_labels = batch['arousal'].to(device)

        optimizer.zero_grad()
        valence_outputs, arousal_outputs = model(input_ids, attention_mask)
        valence_loss = criterion(valence_outputs.squeeze(), valence_labels)
        arousal_loss = criterion(arousal_outputs.squeeze(), arousal_labels)
        loss = valence_loss + arousal_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess data
    train_data = MidiDataset('data/prompts.json')
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    # Load BERT tokenizer and model
    bert_config = 'config.json'
    tokenizer = BertTokenizer.from_pretrained(bert_config)
    model = MidiSentimentModel(bert_config).to(device)

    # Set optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'path/to/save/trained_model.pt')

if __name__ == '__main__':
    main()