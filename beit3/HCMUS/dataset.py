from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer
from underthesea import word_tokenize
import re
import json
import os
import h5py
import torch
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word", use_fast=False)

class ViVQADataset(Dataset):
    def __init__(self, df, answer_path, *image_features_paths):
        with open(answer_path, 'r') as f:
            vocab = json.loads(f.read())

        self.vocab_a = vocab['answer']
        self.dataset = df

        # q and a
        self.questions = [self.question2ids(question) for question in preprocess_questions(df)]
        self.answers = self.answers2idx(preprocess_answers(df), self.vocab_a)
        
        # v 
        self.image_features_paths = image_features_paths
        self.visuals_id_to_index = self.create_visuals_id_to_index()
        self.visuals_ids = self.dataset['img_id']

        self.feature_names = [os.path.basename(path).split('.')[0] for path in self.image_features_paths]

    def question2ids(self, question, max_len=40):
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
        if not hasattr(self, 'features_files'):
            self.features_files = [h5py.File(image_features_path, 'r') for image_features_path in self.image_features_paths]
        visuals_ids = self.features_files[-1]['ids'][()]
        visuals_id_to_index = {id: i for i, id in enumerate(visuals_ids)}
        return visuals_id_to_index

    def load_image(self, image_id):
        index = self.visuals_id_to_index[image_id]
        data_img = [features_file['features'] for features_file in self.features_files]
        d = {}
        for name, data in zip(self.feature_names, data_img):
            d[name] = torch.from_numpy(data[index].astype('float32'))
        return d

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        image_id = self.visuals_ids[idx]
        v = self.load_image(image_id)
        v = torch.cat([v[self.feature_names[0]], v[self.feature_names[1]]], dim=0)
        
        q_inputs, q_attn_mask = self.questions[idx].values()
        a = self.answers[idx]
        
        item = {
            'image': v, 
            'question': q_inputs, 
            'padding_mask': q_attn_mask,
            'labels': a
        }
        
        return item
    
def get_dataset(opt, feature_paths):
    df_train = pd.read_csv(opt.train_path, index_col=0)
    df_test = pd.read_csv(opt.test_path, index_col=0)

    train_dataset = ViVQADataset(df_train, opt.ans_path, *feature_paths)
    test_dataset = ViVQADataset(df_test, opt.ans_path, *feature_paths)

    return train_dataset, test_dataset
    
period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
comma_strip = re.compile(r'(\d)(,)(\d)')
punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!.')
punctuation = re.compile(r'([{}])'.format(re.escape(punctuation_chars)))
punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(punctuation_chars))

def process_punctuation(s):
    if punctuation.search(s) is None:
        return s
    s = punctuation_with_a_space.sub('', s)
    if re.search(comma_strip, s) is not None:
        s = s.replace(',', '')
    s = punctuation.sub(' ', s)
    s = period_strip.sub('', s)
    return s.strip()

def preprocess_questions(df):
    questions = [word_tokenize(question.lower(), format='text') for question in list(df['question'])]
    return questions

def preprocess_answers(df):
    answers = [process_punctuation(answer.lower()) for answer in list(df['answer'])]
    return answers