from transformers import AutoTokenizer, T5ForConditionalGeneration
import spacy, torch
from spacy.lang.en.stop_words import STOP_WORDS
from heapq import nlargest

import os, json

configs_file = os.environ.get('configs_file', 'configs.json')
configs = {}

if os.path.exists(configs_file):
    with open(configs_file, 'r') as f:
        configs = json.load(f)

t5_pretrained_tokenizer = configs.get('t5_pretrained_tokenizer', 't5-base')

__t5_tokenizer = AutoTokenizer.from_pretrained(t5_pretrained_tokenizer)

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
__t5_summarizer = T5ForConditionalGeneration.from_pretrained('t5-base').to(__device)

def use_t5(doc, keep = 0.3):
    global __t5_tokenizer, __t5_summarizer, __device
    
    embedded = __t5_tokenizer('summarize: ' + doc, return_tensors="pt")
        
    generated_ids = __t5_summarizer.generate(
        input_ids = embedded.input_ids.to(__device),
        attention_mask = embedded.attention_mask.to(__device),
        do_sample = True, 
        top_k = 1, 
        temperature = 0.7
    )
    
    return __t5_tokenizer.decode(
        generated_ids.squeeze(), skip_special_tokens=True
    )