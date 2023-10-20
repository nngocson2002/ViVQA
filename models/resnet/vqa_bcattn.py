import torch.nn as nn
from utils import config
from modules.TextEncoder import PhoBertExtractor
import numpy as np

class BiDirectionalCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, mid_features, dropout=0.0):
        super(BiDirectionalCrossAttention, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(config.question_features*config.output_size*config.output_size, config.question_features)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, mid_features),
            nn.ReLU(),
            nn.Linear(mid_features, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, v_features, q_features):

        v_attn_output, _ = self.multihead_attn(query=v_features, key=v_features, value=v_features)
        q_attn_output, _ = self.multihead_attn(query=q_features, key=q_features, value=q_features)

        v = self.layer_norm(v_features + q_features + v_attn_output)
        q = self.layer_norm(v_features + q_features + q_attn_output)

        v_fc = self.fc(v)
        q_fc = self.fc(q)

        v = self.layer_norm(v + self.dropout(v_fc))
        q = self.layer_norm(q + self.dropout(q_fc))

        return v, q
    
class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, dropout=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(dropout))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(dropout))
        self.add_module('lin2', nn.Linear(mid_features, out_features))

class ViVQAModel(nn.Module):
    def __init__(self, q_features, num_heads, mid_features, num_classes, num_cross_attn_layers=1, dropout=0.0):
        super(ViVQAModel, self).__init__()

        self.text = PhoBertExtractor()
        self.linear = nn.Linear(np.prod(config.VISUAL_MODEL['Resnet152']['feature_shape']), q_features)
        self.flatten = nn.Flatten()
        self.num_cross_attention_layers = num_cross_attn_layers

        self.cross_attn_layers = nn.ModuleList([
            BiDirectionalCrossAttention(
                embed_dim=q_features,
                num_heads=num_heads,
                mid_features=mid_features,
                dropout=dropout
            ) for _ in range(self.num_cross_attention_layers)
        ])

        self.classifier = Classifier(
            in_features=q_features,
            mid_features=512,
            out_features=num_classes,
            dropout=dropout
        )

    def forward(self, v, q):
        v = v/(v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8) # Normalize
        v = self.linear(self.flatten(v))
        q = self.text(q['input_ids'].squeeze(dim=1), q['attention_mask'].squeeze(dim=1))
        
        for cross_attn in self.cross_attn_layers:
            v, q = cross_attn(v, q)

        x = v * q
        answer = self.classifier(x)
        return answer
    
if __name__ == '__main__':
    # ???
    model = ViVQAModel(
        q_features=config.TEXT_MODEL['PhoBert']['text_features'], 
        num_heads=12, 
        mid_features=config.TEXT_MODEL['PhoBert']['text_features']*2,
        num_classes=config.max_answers, 
        num_cross_attn_layers=1,
        dropout=0.3
    )