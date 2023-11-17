import torch.nn as nn
import torch
from transformers import AutoModel

class PhoBertExtractor(nn.Module):
    def __init__(self):
        super(PhoBertExtractor, self).__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            last_hidden_states = self.phobert(input_ids, attention_mask)
        features = last_hidden_states[0][:, 0, :]
        return features
    
class BartPhoExtractor(nn.Module):
    def __init__(self):
        super(BartPhoExtractor, self).__init__()
        self.bartpho_word = AutoModel.from_pretrained("vinai/bartpho-word")
        
    def forward(self, input_ids, attention_mask):
        last_hidden_states = self.bartpho_word(input_ids, attention_mask)
        features = last_hidden_states[0]
        return features