from utils import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.textEncoder import PhoBertExtractor

class ViVQAModel(nn.Module):
    def __init__(self):
        super(ViVQAModel, self).__init__()

        self.text = PhoBertExtractor()

        self.attention = Attention(
            v_features=config.visual_features,
            q_features=config.question_features,
            mid_features=512,
            num_attn_maps=config.num_attention_maps
        )

        self.classifier = Classifier(
            in_features= config.num_attention_maps*config.visual_features+config.question_features,
            mid_features=512,
            out_features=config.max_answers,
        )

    def forward(self, v, q):
        q = self.text(q['input_ids'].squeeze(dim=1), q['attention_mask'].squeeze(dim=1))
        v = v/(v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8) # Normalize
        v = self.attention(v, q) # (b, out2 * c)
        concat = torch.cat((v, q), dim=1)
        answer = self.classifier(concat)
        return answer

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features):
        super(Classifier, self).__init__()
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('lin2', nn.Linear(mid_features, out_features))

class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, num_attn_maps):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(v_features, mid_features, 1)
        self.lin1 = nn.Linear(q_features, mid_features)
        self.conv2 = nn.Conv2d(mid_features, num_attn_maps, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        v = self.conv1(v) # (b, mid_features, 14, 14)
        q = self.lin1(q)[:, :, None, None].expand_as(v) # (b, mid_features, 14, 14)

        fuse = self.relu(v * q)
        x = self.conv2(fuse)

        attn_weighted = compute_attention_weights(x)
        weighted_average = compute_weighted_average(v, attn_weighted)

        return weighted_average
        
def compute_attention_weights(conv_result):
    b, c = conv_result.size()[:2]
    flattened = conv_result.view(b, c, -1).unsqueeze(2) # (b, out2, 1, h*w)
    attn_weighted = F.softmax(flattened, dim=-1)
    return attn_weighted # (b, out2, 1, h*w)
        
def compute_weighted_average(v, attn_weighted):
    b, c = v.size()[:2]
    flattened = v.view(b, 1, c, -1) # (b, 1, c, h*w)
    features_glimpse = attn_weighted * flattened # (b, out2, c, h*w)
    weighted_average = features_glimpse.sum(dim=-1) # (b, out2, c, 1)
    return weighted_average.view(b, -1) # concat out2 glimpse => (b, out2 * c)

# (batch_size, hidden_size)
# (batch_size, num_features, output_size, output_size)

def tile(q_features, v_features):
    batch_size, num_features = q_features.size()
    _, _, height, width = v_features.size()
    spatial_size = v_features.dim() - 2
    tiled = q_features.view(batch_size, num_features, *([1]*spatial_size)).expand(batch_size, num_features, height, width)
    return tiled