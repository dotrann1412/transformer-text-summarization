from transformers import T5Tokenizer
import spacy, torch
from spacy.lang.en.stop_words import STOP_WORDS
from heapq import nlargest
from .models import T5SummarizeModel

import os, json

configs_file = os.environ.get('configs_file', 'configs.json')
configs = {}

if os.path.exists(configs_file):
    with open(configs_file, 'r') as f:
        configs = json.load(f)

t5_pretrained_tokenizer = configs.get('t5_pretrained_tokenizer', 't5-small')

__t5_tokenizer = T5Tokenizer.from_pretrained(t5_pretrained_tokenizer)

__punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
__nlp = spacy.load('en_core_web_sm')
__stopwords = list(STOP_WORDS)

def use_word_frequency(document, keep = 0.3):
    global __t5_tokenizer, __punctuation, __nlp, __stopwords
    tokenized = __t5_tokenizer.tokenize(document)
    frequencies = {}
    
    for w in tokenized:
        if w not in __punctuation \
            and w not in __stopwords:
            if w not in frequencies: frequencies[w] = 1
            else: frequencies[w] += 1
        
    max_frequency = max(frequencies.values())
    
    for key, val in frequencies.items():
        frequencies[key] = val / max_frequency
    
    sentences = [sent for sent in __nlp(document).sents]

    sentences_frequencies = {}
    for sent in sentences:
        for w in __t5_tokenizer.tokenize(' '.join([w.text for w in sent])):
            if w in frequencies:
                if sent not in sentences_frequencies: sentences_frequencies[sent] = 0
                sentences_frequencies[sent] += frequencies[w]
    
    expected_length = int(len(sentences) * keep)
    summary = nlargest(expected_length, sentences_frequencies, key = sentences_frequencies.get)

    return ' '.join([sent.text for sent in summary])

__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
__t5_summarizer = T5SummarizeModel(pretrained = configs.get('t5_pretrained_model', 't5-small')).to(__device)


def use_t5(document, keep = 0.3):
    global __t5_tokenizer, __t5_summarizer, __device
    
    text_encoding = __t5_tokenizer(
        document,
        max_length=configs.get('t5_max_length_embedding', 1024),
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    ).to(__device)

    generated_ids = __t5_summarizer.model.generate(
        input_ids = text_encoding['input_ids'],
        attention_mask = text_encoding['attention_mask'],
        max_length=configs.get('t5_max_length_summarization', 512),
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )

    preds = [
            __t5_tokenizer.decode(gen_id, skip_special_tokens = True, clean_up_tokenization_spaces=True)
            for gen_id in generated_ids
    ]

    return "".join(preds).strip()

def use_bert(document, keep = 0.3):
    pass

def use_gpt(document, keep = 0.3):
    pass