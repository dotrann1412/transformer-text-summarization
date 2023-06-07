import os, pandas as pd, torch, json

root = './text-summarization/dataset/cnn_dailymail'

from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    AdamW
)

with open('./text-summarization/configs/1.json') as f:
    conf = json.load()

tokenizer = T5Tokenizer.from_pretrained('t5-base')
from tqdm import tqdm

# meaningless
for file in os.listdir(root):
    if not file.endswith('.csv'): continue
    
    df = pd.read_csv(os.path.join(root, file))
    cluster = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f'Processing {file}'):
        tokenized_article = tokenizer(
            row['article'],
            max_length =  512,
            padding = 'max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        tokenized_summary = tokenizer(
            row['highlights'],
            max_length= 128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        labels = tokenized_summary['input_ids']
        labels[labels == 0] = -100
        
        cluster.append([
            tokenized_article['input_ids'].flatten(),
            tokenized_article['attention_mask'].flatten(),
            labels.flatten(),
            tokenized_summary['attention_mask'].flatten()
        ])
    
    text_input_ids, text_attention_mask, labels, labels_attention_mask = zip(*cluster)
    text_input_ids, text_attention_mask, labels, labels_attention_mask = torch.stack(text_input_ids), torch.stack(text_attention_mask), torch.stack(labels), torch.stack(labels_attention_mask)
    
    torch.save(
        {
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "labels": labels,
            "labels_attention_mask": labels_attention_mask
        },
        os.path.join(root, file.replace('.csv', '.pt'))
    )