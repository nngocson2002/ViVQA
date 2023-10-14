from torch.utils.data import Dataset
from utils import config
from utils.vocab import preprocess_questions, preprocess_answers
import json
import torch
import h5py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)

class ViVQADataset(Dataset):
    def __init__(self, df, image_features_path):
        with open(config.__VOCAB__, 'r') as f:
            vocab = json.loads(f.read())

        self.vocab_a = vocab['answer']
        self.dataset = df

        # q and a
        self.questions = [self.question2ids(question) for question in preprocess_questions(df)]
        self.answers = self.answers2idx(preprocess_answers(df), self.vocab_a)
        
        # v 
        self.image_features_path = image_features_path
        self.visuals_id_to_index = self.create_visuals_id_to_index()
        self.visuals_ids = self.dataset['img_id']

    def question2ids(self, question, max_len=100):
        tkz = tokenizer.encode_plus(
            text=question,
            padding='max_length',
            max_length=max_len,
            truncation=True, 
            return_tensors='pt', 
            return_attention_mask=True,
            return_token_type_ids=False
        ) 
        # {'input_ids': tensor, 'attention_mask': tensor}
        return tkz

    def answers2idx(self, answers, vocab_a):
        return [vocab_a[answer] for answer in answers]

    def create_visuals_id_to_index(self):
        if not hasattr(self, 'features_file'):
            self.features_file = h5py.File(self.image_features_path, 'r')
        visuals_ids = self.features_file ['ids'][()]
        visuals_id_to_index = {id: i for i, id in enumerate(visuals_ids)}
        return visuals_id_to_index

    def load_image(self, image_id):
        index = self.visuals_id_to_index[image_id]
        dataset = self.features_file['features']
        img = dataset[index].astype('float32')
        return torch.from_numpy(img)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        image_id = self.visuals_ids[idx]
        v = self.load_image(image_id)
        q = self.questions[idx]
        a = self.answers[idx]
        return v, q, a