from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from transformers import (
    T5ForConditionalGeneration,
    AdamW
)
import pytorch_lightning as pl

class T5SummarizeModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        
        super(T5SummarizeModel, self).__init__()

        self.pretrained_model = kwargs.get('pretrained_model', 't5-small')
        
        self.lr = kwargs.get('lr', 1e-5)

        self.model = T5ForConditionalGeneration.from_pretrained(self.pretrained_model) \
            if self.pretrained_model is not None else T5ForConditionalGeneration()
            
    def forward(self, input_ids, attention_mask=None, decoder_attention_mask=None, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        
        return output.loss, output.logits
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask = batch['text_input_ids'], batch['text_attention_mask']
        labels, labels_attention_mask = batch['labels'], batch['labels_attention_mask']
        
        loss, _ = self.forward(
            input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            decoder_attention_mask=labels_attention_mask
        )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']
        
        loss, _ = self.forward(
            input_ids, 
            attention_mask=attention_mask, 
            labels=labels,
            decoder_attention_mask=labels_attention_mask
        )
        
        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']
        
        loss, logits = self.forward(
            input_ids, 
            attention_mask=attention_mask, 
            labels=labels,
            decoder_attention_mask=labels_attention_mask
        )
        
        return loss
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)
    

    
