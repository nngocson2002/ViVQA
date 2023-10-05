from torch.utils.data import Dataset
from utils import config
from utils.vocab import preprocess_questions, preprocess_answers
import json
from underthesea import text_normalize, word_tokenize
import torch
import torch.nn as nn
import numpy as np

class ViVQADataset(Dataset):
    def __init__(self, df, feature_path):
        with open(config.__VOCAB__, 'r') as f:
            vocab = json.loads(f.read())

        self.vocab_q = vocab['question']
        self.vocab_a = vocab['answer']
        self.dataset = df

        # q and a
        self.questions = self.encode_questions(preprocess_questions(df), self.vocab_q)
        self.answers = self.encode_answers(preprocess_answers(df), self.vocab_a)
        
        # v 
        self.image_features_path = feature_path
        self.visuals_id_to_index = self.create_visuals_id_to_index()
        self.visuals_ids = self.dataset['img_id']
        
    def max_question_len(self, questions):
        return max([len(word_tokenize(question)) for question in questions])

    def encode_questions(self, questions, vocab_q):
        max_len = self.max_question_len(questions)
        vecs = []
        for question in questions:
            words = word_tokenize(question)
            vec = torch.zeros(max_len, dtype=torch.long)
            for i, word in enumerate(words):
                idx = vocab_q.get(word, 0)
                vec[i] = idx
            vecs.append(vec)
        return vecs
    
    def encode_answers(self, answers, vocab_a):
        max_len = len(vocab_a)
        vecs = []
        for answer in answers:
            vec = torch.zeros(max_len)
            idx = vocab_a.get(answer)
            if idx is not None:
                vec[idx] = 1
                vecs.append(vec)
                continue
        return vecs

    def create_visuals_id_to_index(self):
        if not hasattr(self, 'features_file'):
            self.features_file = np.load(self.image_features_path)
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