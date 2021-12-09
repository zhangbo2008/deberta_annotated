
# 跟bert 区别. 加了相对位置的计算.这个比绝对位置更准确.因为一个句子可以前后加一些废话.而不影响句子本身含义.
# config:https://huggingface.co/microsoft/deberta-base/blob/main/config.json
# vocab:https://huggingface.co/microsoft/deberta-base/blob/main/vocab.json
from transformers import AutoTokenizer, AutoModel


from transformers import DebertaTokenizer, DebertaModel
import torch
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
model = DebertaModel.from_pretrained('microsoft/deberta-base')

with open('vocabfordebug.txt','w',encoding='utf-8') as f:
    f.write(str(tokenizer.get_vocab()))



inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state






from transformers import DebertaTokenizer, DebertaForMaskedLM
import torch
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
model = DebertaForMaskedLM.from_pretrained('microsoft/deberta-base')
inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits








