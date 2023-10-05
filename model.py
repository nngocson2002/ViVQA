from utils import config
import torch
import torch.nn as nn
import torch.nn.functional as F

class ViVQAModel(nn.Module):
    def __init__(self):
        super(ViVQAModel, self).__init__()

        self.text = TextEncoder(
            vocab_q_size=config.max_vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_size=config.question_features,
            num_layers=1
        )

        self.attention = Attention(
            in_channels=config.visual_features+config.question_features,
            out_chanels1=512,
            out_chanels2=2
        )

        self.classifier = Classifier(
            in_features= config.num_attention_maps*config.visual_features+config.question_features,
            mid_features=1024,
            out_features=config.max_answers,
        )

    def forward(self, v, q):
        q = self.text(q) # (b, hidden_size)
        v = v/(v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8) # Normalize
        v = self.attention(v, q) # (b, out2 * c)
        concat = torch.cat([v, q], dim=1)
        answer = self.classifier(concat)
        return answer

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features):
        super(Classifier, self).__init__()
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('lin2', nn.Linear(mid_features, out_features))

class Attention(nn.Module):
    def __init__(self, in_channels, out_chanels1, out_chanels2):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_chanels1, 1)
        self.conv2 = nn.Conv2d(out_chanels1, out_chanels2, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        q_tiled = tile(q, v)
        concat = torch.cat([v, q_tiled], dim=1)

        conv1 = self.conv1(concat)
        relu = self.relu(conv1)
        conv2 = self.conv2(relu) # (b, out2, h, w)

        attn_weighted = compute_attention_weights(conv2)
        weighted_average = compute_weighted_average(v, attn_weighted)

        return weighted_average
    
class TextEncoder(nn.Module):
    def __init__(self, vocab_q_size, embedding_dim, hidden_size, num_layers):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_q_size+1, embedding_dim, padding_idx=0)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)

    def forward(self, question):
        embedded = self.embedding(question) # (b, embedding_dim)
        tanhed = self.tanh(embedded) 
        _, (_, cell) = self.lstm(tanhed) # (b, hidden_size)
        return cell.squeeze(0)
        
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

