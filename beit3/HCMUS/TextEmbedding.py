from transformers import AutoModel
import torch.nn as nn

class BartPhoExtractor(nn.Module):
    def __init__(self):
        super(BartPhoExtractor, self).__init__()
        self.bartpho_word = AutoModel.from_pretrained("vinai/bartpho-word")
        
    def forward(self, input_ids, attention_mask):
        last_hidden_states = self.bartpho_word(input_ids, attention_mask)
        features = last_hidden_states[0]
        return features