from models.resnet.vqa_attn import ViVQAModel
import torch


def load_model(checkpoint_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ViVQAModel()
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    model.to(device)
    return model

checkpoint_path = ''
model = load_model(checkpoint_path)