import sys
sys.path.append('./')
import torch.nn as nn
import torch
from utils import config
import torch.nn.functional as F
from modules.TextEncoder import PhoBertExtractor

class ViVQAModel(nn.Module):
    def __init__(self, v_features, q_features, num_attn_maps, mid_features, num_classes, dropout=0.0):
        super(ViVQAModel, self).__init__()

        self.text = PhoBertExtractor()

        self.attention = Attention(
            v_features=v_features,
            q_features=q_features,
            mid_features=mid_features, # 512
            num_attn_maps=num_attn_maps,
            dropout=dropout
        )

        self.classifier = Classifier(
            in_features=num_attn_maps*mid_features+q_features,
            mid_features=512,
            out_features=num_classes,
            dropout=dropout
        )

    def forward(self, v, q):
        q = self.text(q['input_ids'].squeeze(dim=1), q['attention_mask'].squeeze(dim=1))
        v = v/(v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8) # Normalize
        v = self.attention(v, q) # (b, out2 * c)
        concat = torch.cat((v, q), dim=1)
        answer = self.classifier(concat)
        return answer

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, dropout=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(dropout))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(dropout))
        self.add_module('lin2', nn.Linear(mid_features, out_features))

class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, num_attn_maps, dropout=0.0):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(v_features, mid_features, 1)
        self.lin1 = nn.Linear(q_features, mid_features)
        self.conv2 = nn.Conv2d(mid_features, num_attn_maps, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, v, q):
        v = self.conv1(self.dropout(v)) # (b, mid_features, 14, 14)
        q = self.lin1(q)[:, :, None, None].expand_as(v) # (b, mid_features, 14, 14)

        fuse = self.relu(v * q)
        x = self.conv2(self.dropout(fuse)) # (b, num_attn_maps, 14, 14)

        attn_weighted = compute_attention_weights(x)
        weighted_average = compute_weighted_average(v, attn_weighted)

        return weighted_average
        
def compute_attention_weights(conv_result):
    b, c = conv_result.size()[:2] # c = num_attn_maps
    flattened = conv_result.view(b, c, -1).unsqueeze(2) # (b, num_attn_maps, 1, h*w)
    attn_weighted = F.softmax(flattened, dim=-1)
    return attn_weighted # (b, num_attn_maps, 1, h*w)
        
def compute_weighted_average(v, attn_weighted):
    b, c = v.size()[:2]
    flattened = v.view(b, 1, c, -1) # (b, 1, c, h*w)
    features_glimpse = attn_weighted * flattened # (b, num_attn_maps, c, h*w)
    weighted_average = features_glimpse.sum(dim=-1) # (b, num_attn_maps, c, 1)
    return weighted_average.view(b, -1) # concat out2 glimpse => (b, num_attn_maps * c) 2*512

if __name__ == '__main__':
    # ????    
    model = ViVQAModel(
        v_features=config.VISUAL_MODEL['Resnet152']['visual_features'],
        q_features=config.TEXT_MODEL['PhoBert']['text_features'], 
        num_attn_maps=2, 
        mid_features=config.TEXT_MODEL['PhoBert']['text_features'], 
        num_classes=config.max_answers, 
        dropout=0.3
    )