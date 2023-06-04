import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
from transformers import T5Tokenizer

class TextSummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        self.tokenizer = kwargs['tokenizer']
        self.data = kwargs['data']
        self.text_embedded_length = kwargs['text_embedded_length']        
        self.summary_embedded_length = kwargs['summary_embedded_length']
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        text, summary = row['article'], row['highlights']
        
        encoded_text = self.tokenizer(
            text,
            max_length = self.text_embedded_length,
            padding = 'max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        summary_embedded = self.tokenizer(
            summary,
            max_length=self.summary_embedded_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        labels = summary_embedded['input_ids']
        labels[labels == 0] = -100
        
        return {
            "text": text,
            "summary": summary,
            "text_input_ids": encoded_text['input_ids'].flatten(),
            "text_attention_mask": encoded_text['attention_mask'].flatten(),
            "labels": labels.flatten(),
            "labels_attention_mask": summary_embedded['attention_mask'].flatten()
        }
    
    def __len__(self):
        return len(self.data)
    
    
class CustomizedDataLoader(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.batch_size = kwargs['batch_size']
        
        self.train_data = kwargs['train_data']
        self.val_data = kwargs['val_data']
        
        self.test_data = kwargs.get('test_data', None)
        
        self.tokenizer = kwargs['tokenizer']
        
        self.text_embedded_length = kwargs.get('text_embedded_length', 512)
        self.summary_embedded_length = kwargs('summary_embedded_length', 128)
        
    def setup(self, stage = None):
        self.train_ds = TextSummarizationDataset(
            tokenizer=self.tokenizer,
            data=self.train_data,
            text_embedded_length=self.text_embedded_length,
            summary_embedded_length=self.summary_embedded_length
        )
        
        self.val_ds = TextSummarizationDataset(
            tokenizer=self.tokenizer,
            data=self.val_data,
            text_embedded_length=self.text_embedded_length,
            summary_embedded_length=self.summary_embedded_length
        )
        
        if self.test_data is not None:
            self.test_ds = TextSummarizationDataset(
                tokenizer=self.tokenizer,
                data=self.test_data,
                text_embedded_length=self.text_embedded_length,
                summary_embedded_length=self.summary_embedded_length
            )
        else:
            self.test_ds = None
        
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.val_ds if not self.test_ds else self.test_ds, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
def load(train_csv, val_csv, test_csv, *args, **kwargs):
    train_df, val_df, test_df = pd.read_csv(train_csv), pd.read_csv(val_csv), pd.read_csv(test_csv)
    text_embedded_length = kwargs.get('text_embedded_length', 512)
    summary_embedded_length = kwargs.get('summary_embedded_length', 128)
    tokenizer = kwargs.get(
        'tokenizer', 
        T5Tokenizer.from_pretrained(kwargs.get('pretrained_model', 't5-base'))
    )
    batch_size = kwargs.get('batch_size', 8)
    
    return CustomizedDataLoader(
        train_data=train_df,
        val_data=val_df,
        test_data=test_df,
        tokenizer=tokenizer,
        text_embedded_length=text_embedded_length,
        summary_embedded_length=summary_embedded_length,
        batch_size=batch_size
    )    