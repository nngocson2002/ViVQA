from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer
from underthesea import word_tokenize
import re
import json
import torch
import pandas as pd
from omegaconf import OmegaConf
from lavis.common.registry import registry
from typing import List, Optional, Union
import transformers
from transformers.utils import TensorType
from lavis.processors.base_processor import BaseProcessor
from PIL import Image

class Process:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_processor = self.load_preprocess()
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word", use_fast=False)

    def load_preprocess(self):
        config = OmegaConf.load(registry.get_model_class(name="blip2_feature_extractor").default_config_path(model_type="pretrain"))
        preprocess_cfg = config.preprocess
        def _build_proc_from_cfg(cfg):
            return (
                registry.get_processor_class(cfg.name).from_config(cfg)
                if cfg is not None
                else BaseProcessor()
            )
        vis_proc_cfg = preprocess_cfg.get("vis_processor")
        vis_eval_cfg = vis_proc_cfg.get("eval")
        vis_processors = _build_proc_from_cfg(vis_eval_cfg)
        return vis_processors

    def __call__(
        self,
        image,
        text: Union[str, List[str], List[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, transformers.utils.generic.PaddingStrategy] = False,
        truncation: Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ):
        image = self.image_processor(image.convert("RGB"))

        text = word_tokenize(text.lower(), format='text')
        text = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose
        )
        text['padding_mask'] = 1 - text['attention_mask']

        return {
            'image': image.to(self.device),
            'question': text['input_ids'].to(self.device),
            'padding_mask': text['padding_mask'].to(self.device)
        }
    
class ViVQADataset(Dataset):
    def __init__(self, dataframe, processor, image_path, answers_path):
        with open(answers_path, 'r') as f:
            vocab = json.loads(f.read())
        self.vocab_a = vocab['answer']
        self.answers = self.answers2idx(preprocess_answers(dataframe), self.vocab_a)
        
        self.dataframe = dataframe
        self.image_path = image_path
        self.processor = processor

    def answers2idx(self, answers, vocab_a):
        return [vocab_a[answer] for answer in answers]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        question = self.dataframe['question'].iloc[idx]
        img_id = self.dataframe['img_id'].iloc[idx]
        answer = self.answers[idx]

        image = Image.open(f'{self.image_path}/{img_id}.jpg')
        inputs = self.processor(image, question, 
                                return_tensors='pt', 
                                return_token_type_ids=False,
                                return_attention_mask=True, 
                                truncation=True,
                                padding='max_length', 
                                max_length=40)
        inputs |= {'labels': answer.to(self.processor.device)}

        return inputs
    
def get_dataset(opt):
    processor = Process()

    df_train = pd.read_csv(opt.train_path, index_col=0)
    df_test = pd.read_csv(opt.test_path, index_col=0)

    train_dataset = ViVQADataset(df_train, processor, opt.image_path, opt.ans_path)
    test_dataset = ViVQADataset(df_test, processor, opt.image_path, opt.ans_path)

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