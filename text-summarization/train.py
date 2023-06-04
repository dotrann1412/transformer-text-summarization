from transformers import (
    T5ForConditionalGeneration,
    AdamW,
    T5TokenizerFast as T5Tokenizer
)
from datautils import load
from model import T5SummarizeModel
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import os

configs = {
    "batch_size": 8,
    "t5_tokenizer_name": 't5-base',
    "text_embedded_length": 512,
    "summary_embedded_length": 128,
    "checkpoints": os.path.join(os.getcwd(), "checkpoints"),
    'max_epochs': 3,
    'learning_rate': 1e-5
}

root = './text-summarization/dataset/cnn_dailymail'
train_csv, test_csv, val_csv = os.path.join(root, 'train.csv'), os.path.join(root, 'test.csv'), os.path.join(root, 'validation.csv')

t5_tokenizer = T5Tokenizer.from_pretrained(configs['t5_tokenizer_name'])

database = load(
    train_csv=train_csv,
    val_csv=val_csv,
    test_csv=test_csv,
    tokenizer=t5_tokenizer,
    batch_size=configs['batch_size'],
    text_embedded_length=configs['text_embedded_length'],
    summary_embedded_length=configs['summary_embedded_length']
)

model = T5SummarizeModel(
    pretrained_model=configs['t5_tokenizer_name'],
    lr=configs['learning_rate']
)

if not os.path.exists(configs['checkpoints']):
    os.mkdir(configs['checkpoints'])

checkpoint_callback = ModelCheckpoint(
    dirpath = configs['checkpoints'],
    filename = 'best-checkpoint',
    save_top_k = 1,
    verbose = True,
    monitor = 'val_loss',
    mode = 'min'
)

logger = TensorBoardLogger(
    save_dir = configs['checkpoints'],
    name = 'logs'
)

trainer = pl.Trainer(
    logger = logger,
    callbacks = [checkpoint_callback],
    max_epochs = configs['max_epochs'],
    accelerator = "gpu"
)

trainer.fit(model, database)