from transformers import AutoModel
import torch.nn as nn

class BartPhoExtractor(nn.Module):
    def __init__(self):
        super(BartPhoExtractor, self).__init__()
        self.bartpho_syllable = AutoModel.from_pretrained("vinai/bartpho-syllable")
        
    def forward(self, input_ids, attention_mask):
        last_hidden_states = self.bartpho_syllable(input_ids, attention_mask)
        features = last_hidden_states[0]
        return features