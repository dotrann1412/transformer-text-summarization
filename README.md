# Transformer Text Summarization
Final project of Statistical Learning Course.

## Status

<p align="center">
  <a href="https://www.codefactor.io/repository/github/dotrann1412/transformer-text-summarization"><img src="https://www.codefactor.io/repository/github/dotrann1412/transformer-text-summarization/badge" alt="CodeFactor" /></a>
</p>

## Students

Here is our's information:

| ID        | Name             | Contact                                                       |
| -------   | -----            | ----                                                          | 
| 20120057  | Trần Ngọc Đô     | [Linkedin](https://www.linkedin.com/in/ndtran11/)             |
| 20120024  | Huỳnh Minh Tuấn  | [Linkedin](https://www.linkedin.com/in/tuan-huynh-b51569122/) |

## Overview
This project is a small application of auto-text summarization using Transformer architecture. The codebase includes 2 parts
- [Backend Server](server/)
- [Frontend Web](web/)

## Techstack
### 1. Backend
**Languages**:
- Python

**Frameworks/Libraries**:
- Django - Build RESTful API
- SpaceOCR - Process OCR Task
- Pytorch - Build model and inference
- HuggingFace - Using for Transformer architecture

### 2. Frontend
**Languages**:
- Javascript
- Typescript
- CSS and HTML (markup language)

**Frameworks/Libraries**:
- VueJS - Web structure
- ElementPlus - UI supports
- Axios - API communicating
- Vite - Optimization

## Getting started
```sh
# setup backend
$user cd server
$user python manage.py runserver <host>:<port>

# setup frontend
$user cd web
$user yarn dev --port <web_port>
```

## Training

Hyper-parameters that were used to train the model:

| Parameter     | Value                |
| ---------     |---------             |
| **No. Epoch**     | 3                    |
| **Learning rate** | 1e-5 (First two epochs), 5e-6 (Last epoch) |
| **Optimizer**    | AdamW                 |
| **Layers**        | Full                 |

The pre-trained model and weight are now available: [here](https://huggingface.co/ndtran/t5-small_cnn-daily-mail). Use this code snippet for inference:

```python
from transformers import T5ForConditionalGeneration, AutoTokenizer

model =  T5ForConditionalGeneration.from_pretrained('ndtran/t5-small_cnn-daily-mail')
model.eval()
tokenizer = AutoTokenizer.from_pretrained('t5-small')
        
generated_ids = model.generate(
      tokenizer(configs['task_prefix'] + text, return_tensors = 'pt').input_ids, 
      do_sample = True, 
      max_length = 256,
      top_k = 1, 
      temperature = 0.8
)
  
output = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens = True)

```

## Web capture

<p align="center">
  <img src="./images/capture.png"/>
</p>

 
Or use this instance app: [Huggingface Space](https://ndtran-t5-small-cnn-daily-mails.hf.space/)

## Issues
Feel free to open any issues.
