import torch
import torch.nn as nn
from transformers import BertModel
from tokenizer import MidiTokenizer

class MidiSentimentModel(nn.Module):
    def __init__(self):
        super(MidiSentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained('roberta-base')
        self.tokenizer = MidiTokenizer()
        self.dropout = nn.Dropout(0.1)
        self.fc_valence = nn.Linear(self.bert.config.hidden_size, 1)
        self.fc_arousal = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        # Ensure input tensors are on the correct device
        input_ids = input_ids.to(self.bert.device)
        attention_mask = attention_mask.to(self.bert.device)

        # Get the outputs from the BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Get valence and arousal outputs
        valence_output = self.fc_valence(pooled_output)
        arousal_output = self.fc_arousal(pooled_output)

        # Apply sigmoid to get values between 0 and 1
        valence_output = self.sigmoid(valence_output)
        arousal_output = self.sigmoid(arousal_output)

        # Return the valence and arousal scores
        return valence_output, arousal_output

    def predict_from_text(self, text):
        if isinstance(text, torch.Tensor):
            text = ''.join([chr(t) for t in text.squeeze().tolist()])

        # Tokenize the input text using MidiTokenizer
        inputs = self.tokenizer.tokenize(text)
        input_ids = torch.tensor([inputs]).to(self.bert.device)  # Add batch dimension and move to correct device
        attention_mask = torch.ones(input_ids.shape).to(self.bert.device)  # Generate attention mask

        # Return the valence and arousal scores
        return self.forward(input_ids, attention_mask)


if __name__ == '__main__':
    model = MidiSentimentModel()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move the model to the appropriate device
    valence, arousal = model.predict_from_text("Your input string here")
    print("Valence:", valence, "Arousal:", arousal)
