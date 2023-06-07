from transformers import (
    AdamW,
    T5TokenizerFast as T5Tokenizer
)
from datautils import load
from model import T5SummarizeModel

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import os, json
from argparse import ArgumentParser


# import optimizer
import torch
from transformers import AdamW
from tqdm import tqdm

import datetime


def main(options):
    root = os.path.join(os.getcwd(), './text-summarization/dataset')
    
    with open(options.configs, 'r') as f:
        configs = json.load(f)
    
    train_csv, test_csv, val_csv = os.path.join(root, configs['train_csv']), os.path.join(root, configs['test_csv']), os.path.join(root, configs['val_csv'])

    t5_tokenizer = T5Tokenizer.from_pretrained(configs['t5_tokenizer_name'])

    database = load( # CustomizedDataLoader
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        tokenizer=t5_tokenizer,
        batch_size=configs['batch_size'],
        text_embedded_length=configs['text_embedded_length'],
        summary_embedded_length=configs['summary_embedded_length']
    )

    model = T5SummarizeModel(
        pretrained_model=configs['t5_tokenizer_name']
    )

    optimizer = AdamW(
        model.parameters(),
        lr=configs['lr']
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[*] Using device: {}'.format(device))
    
    logs = {
        'train_ ': [],
        'valid_loss': []
    }
    
    for epoch in range(configs['epochs']):
        model.train()
        train_data_loader = database.train_dataloader()
        
        epoch_loss = 0
        for i, batch in tqdm(enumerate(train_data_loader)):
            optimizer.zero_grad()
            
            input_ids, attention_mask, decoder_attention_mask, labels = batch
            input_ids, attention_mask, decoder_attention_mask, labels = input_ids.to(device), attention_mask.to(device), decoder_attention_mask.to(device), labels.to(device)
            
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels
            )
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        
        logs['train_loss'].append(epoch_loss / len(train_data_loader))
        
        valid_data_loader = database.val_dataloader()
        epoch_loss = 0
        for i, batch in tqdm(enumerate(valid_data_loader)):
            input_ids, attention_mask, decoder_attention_mask, labels = batch
            input_ids, attention_mask, decoder_attention_mask, labels = input_ids.to(device), attention_mask.to(device), decoder_attention_mask.to(device), labels.to(device)
            
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels
            )
            
            epoch_loss += loss.item()
            
        logs['valid_loss'].append(epoch_loss / len(valid_data_loader))
        
        
        print('[*] Epoch: {} | Train Loss: {} | Valid Loss: {}'.format(epoch, logs['train_loss'][-1], logs['valid_loss'][-1]))
        
    
    # saving training log and checkpoint
    if not os.path.exists(configs['logs']):
        os.makedirs(configs['logs'])            

    if not os.path.exists(configs['checkpoints']):
        os.makedirs(configs['checkpoints'])
        
    with open(os.path.join(configs['logs'], 'logs.json'), 'w') as f:
        json.dump(logs, f)
        
    torch.save(model.state_dict(), os.path.join(configs['checkpoints'], f'model_{datetime.datetime.now()}.pth'))
        

if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--configs', type=str, default='./text-summarization/configs.json')
    parser.add_argument('--gpus', type=int, default=1)

    main(options = parser.parse_args())