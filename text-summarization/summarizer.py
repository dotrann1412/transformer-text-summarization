from transformers import (
    T5TokenizerFast as T5Tokenizer
)

from model import T5SummarizeModel

import os

pretrained_tokenizer = os.environ.get('t5_pretrained_tokenizer', 't5-base')
pretrained_model = os.environ.get('t5_pretrained_model', 't5-base')

__tokenizer = T5Tokenizer.from_pretrained(pretrained_tokenizer)
__model = T5SummarizeModel()

def t5_summarizer(text, device='cpu'):
    global __tokenizer, __model
    
    text_encoding = __tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    ).to(device)

    generated_ids = __model.model.generate(
        input_ids = text_encoding['input_ids'],
        attention_mask = text_encoding['attention_mask'],
        max_length=128,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )

    preds = [
            __tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for gen_id in generated_ids
    ]

    return "".join(preds).strip()