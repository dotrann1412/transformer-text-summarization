from transformers import AutoTokenizer, T5ForConditionalGeneration
import spacy, torch, string
from spacy.lang.en.stop_words import STOP_WORDS
from heapq import nlargest

import os, json

configs_file = os.environ.get('configs_file', 'configs.json')
configs = {}

if os.path.exists(configs_file):
    with open(configs_file, 'r') as f:
        configs = json.load(f)

t5_pretrained_tokenizer = configs.get('t5_pretrained_tokenizer', 't5-small')

__t5_tokenizer = AutoTokenizer.from_pretrained(t5_pretrained_tokenizer)
__punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
__nlp = spacy.load('en_core_web_sm')
__stopwords = list(STOP_WORDS)
__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
__t5_summarizer = T5ForConditionalGeneration.from_pretrained('ndtran/t5-small_cnn-daily-mails').to(__device)

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



def t5_summarize(text):
    global __t5_summarizer, __t5_tokenizer, __device, configs, __nlp
    
    input_ids = __t5_tokenizer(configs['task_prefix'] + text, return_tensors = 'pt').input_ids
        
    generated_ids = __t5_summarizer.generate(
        input_ids.to(__device), 
        do_sample = True, 
        max_length = 256,
        top_k = 1, 
        temperature = 0.8
    )
    
    doc = __nlp(__t5_tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True))
    sents = [str(sent).lstrip(string.punctuation + ' ').rstrip() for sent in doc.sents]
    
    for i, sent in enumerate(sents):
        if len(sent) > 0:
            sents[i] = sent[0].upper() + sent[1:]
    
    return " ".join(sents)

def use_t5(text, keep = 0.3):
    global __t5_summarizer, __t5_tokenizer, __device, configs, __nlp
    
        
    buffer, tokens_count = '', 0
    nlp_text = __nlp(text)
    
    blocks = []
    
    for sent in nlp_text.sents:
        tokens = __t5_tokenizer.tokenize(str(sent))
                
        if len(tokens) > 512:
            if tokens_count > 0:
                blocks.append(buffer)
                buffer, tokens_count = '', 0
            
            blocks.append(str(sent))
            
        buffer += str(sent)
        tokens_count += len(tokens)
        
        if tokens_count > 512:
            blocks.append(buffer)
            buffer, tokens_count = '', 0
            
    if tokens_count > 0:
        blocks.append(buffer)
                    
    return " ".join(t5_summarize(e) for e in blocks)