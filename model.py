import torch
import torch.nn as nn
from transformers import BertModel

class MidiSentimentModel(nn.Module):
    def __init__(self, bert_config):
        super(MidiSentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_config)
        self.dropout = nn.Dropout(0.1)
        self.fc_valence = nn.Linear(self.bert.config.hidden_size, 1)
        self.fc_arousal = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        valence_output = self.fc_valence(pooled_output)
        arousal_output = self.fc_arousal(pooled_output)

        valence_output = self.sigmoid(valence_output)
        arousal_output = self.sigmoid(arousal_output)

        return valence_output, arousal_output