from transformers import BertTokenizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from heapq import nlargest

__tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
__punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
__nlp = spacy.load('en_core_web_sm')
__stopwords = list(STOP_WORDS)

def use_word_frequency(document, keep = 0.3):
    global __tokenizer, __punctuation, __nlp, __stopwords
    tokenized = __tokenizer.tokenize(document)
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
        for w in __tokenizer.tokenize(' '.join([w.text for w in sent])):
            if w in frequencies:
                if sent not in sentences_frequencies: sentences_frequencies[sent] = 0
                sentences_frequencies[sent] += frequencies[w]
    
    expected_length = int(len(sentences) * keep)
    summary = nlargest(expected_length, sentences_frequencies, key = sentences_frequencies.get)

    return ' '.join([sent.text for sent in summary])

def use_t5(document, keep = 0.3):
    pass

def use_bert(document, keep = 0.3):
    pass

def use_gpt(document, keep = 0.3):
    pass