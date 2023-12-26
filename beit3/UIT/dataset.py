from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer
import re
import json
import os
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable", use_fast=False)

class ViVQADataset(Dataset):
    def __init__(self, df, image_path, answer_path):
        with open(answer_path, 'r') as f:
            vocab = json.loads(f.read())

        self.vocab_a = vocab['answer']
        self.dataset = df

        # q and a
        self.questions = [self.question2ids(question) for question in preprocess_questions(df)]
        self.answers = self.answers2idx(preprocess_answers(df), self.vocab_a)
        
        # v
        self.image_path = image_path
        self.visual_ids = self.dataset['img_id']
        self.visual_ids2idx = {self.visual_ids[i]:i for i in range(len(self.visual_ids))}


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

    def load_image(self, image_id):
        path = os.path.join(self.image_path, f'{image_id}.jpg')
        def get_transforms(target_size, central_fraction=1.0):
            return transforms.Compose([
                transforms.Resize(int(target_size / central_fraction)),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
        transform = get_transforms(target_size=224, central_fraction=0.875)
        return transform(Image.open(path).convert('RGB'))

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        image_id = self.visual_ids[idx]
        v = self.load_image(image_id)
        
        q_inputs, q_attn_mask = self.questions[idx].values()
        a = self.answers[idx]
        
        item = {
            'image': v, 
            'question': q_inputs, 
            'padding_mask': q_attn_mask,
            'labels': a
        }
        
        return item
    
def get_dataset(opt):
    df_train = pd.read_csv(opt.train_path, index_col=0)
    df_test = pd.read_csv(opt.test_path, index_col=0)

    train_dataset = ViVQADataset(df_train, image_path=opt.image_path, answer_path=opt.ans_path)
    test_dataset = ViVQADataset(df_test, image_path=opt.image_path, answer_path=opt.ans_path)

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
    questions = [question.lower() for question in list(df['question'])]
    return questions

def preprocess_answers(df):
    answers = [process_punctuation(answer.lower()) for answer in list(df['answer'])]
    return answers