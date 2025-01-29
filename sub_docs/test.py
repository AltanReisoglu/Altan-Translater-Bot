import re
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torch.nn import functional as F

with open(r"C:\Users\bahaa\Downloads\First Citizen.txt", 'r', encoding='utf-8') as f:
    text = f.read()
sentences = re.split(r'\n+', text.strip())
print(sentences)
next_list=[]
for i in sentences:
    
    next_list.append(i+" <EOS>")
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()  # num_words parametresi kaldırıldı
tokenizer.fit_on_texts(next_list)
word_index = tokenizer.word_index
print(len(word_index))