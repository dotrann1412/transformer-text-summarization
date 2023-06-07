from transformers import T5ForConditionalGeneration
import torch
class T5SummarizeModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.model = T5ForConditionalGeneration.from_pretrained(
            kwargs.get('pretrained', 't5-base'),
            ignore_mismatched_sizes=True
        )
                    
    def forward(self, input_ids, attention_mask=None, decoder_attention_mask=None, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        
        return output.loss, output.logits