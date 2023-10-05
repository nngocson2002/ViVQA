from collections import Counter
from underthesea import word_tokenize
import re
import config
import pandas as pd
import json

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


def extract_vocab(questions, answers):
    words = [word for question in questions for word in word_tokenize(question)]
    words = Counter(words).most_common()
    answers = Counter(answers).most_common()
    vocab_q = {word : i+1 for i, (word,_) in enumerate(words)}
    vocab_a = {answer: i for i, (answer,_) in enumerate(answers)}
    return vocab_q, vocab_a

if __name__ == '__main__':

    df = pd.read_csv(config.__DATASET__)

    questions = preprocess_questions(df)
    answers = preprocess_answers(df)

    vocab_q, vocab_a = extract_vocab(questions, answers)

    vocabs = {
        'question': vocab_q,
        'answer': vocab_a,
    }

    with open(config.__VOCAB__, 'w') as f:
        json.dump(vocabs, f)