import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
from torch.nn import functional as F
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_iters = 2000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.0

with open(r"C:\Users\bahaa\Downloads\First Citizen_eos.txt", 'r', encoding='utf-8') as f:
    text = f.read()

strings = text.split("\n")

# Tokenizer işlemi
tokenizer = Tokenizer()
tokenizer.fit_on_texts(strings)
stoi = tokenizer.word_index
itos = dict(zip(stoi.values(), stoi.keys()))
vocab_size = len(stoi)

# Metni sayılara çevirme ve padding
sequences = tokenizer.texts_to_sequences(strings)
padsequences = pad_sequences(sequences, maxlen=block_size, padding='pre')

# Tensor formatına çevirme (flatten ile tek boyuta indirgeme)
data = torch.tensor(padsequences, dtype=torch.long).flatten()

# data loading
def get_batch(split):
    # generate a sequential batch of data of inputs x and targets y

    # Veriyi sıralı almak için sabit bir başlangıç noktası belirleyelim
    global current_index
    if current_index + batch_size >= len(data) - block_size:
        current_index = 0  # Eğer veri biterse başa dön

    ix = torch.arange(current_index, current_index + batch_size)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    
    current_index += batch_size  # Bir sonraki batch için index'i güncelle
    
    return x, y
"""def get_batch(split):
    global current_index
    
    # Veri bitince başa sar
    if current_index + batch_size >= len(data) - block_size:
        current_index = 0  
    
    # Batch için verileri seç
    ix = torch.arange(current_index, current_index + batch_size)
    x_raw = [data[i:i+block_size].tolist() for i in ix]
    y_raw = [data[i+1:i+block_size+1].tolist() for i in ix]

    # Tensor’a çevirme
    x = torch.tensor(x_raw, dtype=torch.long, device=device)
    y = torch.tensor(y_raw, dtype=torch.long, device=device)

    current_index += batch_size  # Batch indeksini güncelle
    
    return x, y"""

# Başlangıç indexi
current_index = 0



class AltanMultiHead(nn.Module):
    def __init__(self,embed_dim,num_head):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_head, dropout=dropout)
    def forward(self, x):
        B, T, C = x.shape  # B = batch_size, T = sequence_length, C = embed_dim

        # Burada query, key, value aynı girdi verisinden alınır
        query = key = value = x.permute(1, 0, 2)  # (sequence_length, batch_size, embed_dim) şeklinde permütasyon yapıyoruz
        
        # Multihead attention hesaplama
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        
       
        
        # Çıkış
        return attn_output.permute(1, 0, 2)  # Çıkışı (B, T, C) şeklinde geri dönüyoruz


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = AltanMultiHead(n_embd,n_head)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
import torch
import numpy as np

def positional_encoding(position, d_model):
    """
    Positional encoding implementation in PyTorch.

    Args:
        position (int): Maximum sequence length.
        d_model (int): Embedding size.

    Returns:
        torch.Tensor: Positional encoding tensor of shape [1, position, d_model].
    """
    # Create a matrix of shape [position, d_model]
    angle_rads = torch.arange(position).unsqueeze(1).float() / torch.pow(
        10000, (2 * (torch.arange(d_model).float() // 2)) / d_model
    )
    
    # Apply sine to even indices in the array; 2i
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    
    # Apply cosine to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads.unsqueeze(0)  # Add batch dimension
    
    return pos_encoding


 # Output: torch.Size([1, 50, 512])

class Encoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, block_size):
        super().__init__()
        # Embedding katmanları
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # Token embedding
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # Positional embedding
        
        # Transformer blokları
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(6)])

        # Son LayerNorm
        self.ln_f = nn.LayerNorm(n_embd)

        # Ağırlıkların başlatılması
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        """
        idx: (B, T) boyutunda bir tensor, integer token indekslerini içerir.
        """
        B, T = idx.shape

        # Token ve pozisyonel embedding'lerin hesaplanması
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        pos_emb = pos_emb.unsqueeze(0).expand(B, T, -1)  # (B, T, C)

        # Embeddinglerin toplanması
        x = tok_emb + pos_emb  # (B, T, C)

        # Transformer bloklarından geçiş
        x = self.blocks(x)  # (B, T, C)

        # Son LayerNorm
        x = self.ln_f(x)  # (B, T, C)

        return x
m=Encoder(vocab_size,n_embd,n_head,block_size).to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
xb, yb = get_batch('train')
output=m(xb).detach()
print(output.shape)

def find_encode(m,sentence):
    encoded=tokenizer.texts_to_sequences(sentence)
    data = torch.tensor(encoded, dtype=torch.long).to(device=device)
    output_sol=m.forward(data)
    return output_sol
#print(find_encode(m,["First Citizen <EOS>"]))
"""class Encoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, block_size):
        super().__init__()
        B,T,C=n_embd/n_head,block_size,n_embd
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = self.position_embedding_table(torch.arange(T, device=device)).unsqueeze(0).expand(B, T, -1)  # (B, T, C)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(6)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
    def forward(self, x):
        B, T = x.shape
        token_embeddings = self.embed(x)  # Token embedding
        position_embeddings = self.pos_embed[:, :T, :]  # Positional encoding
        x = token_embeddings + position_embeddings  # Embeddingler toplanır
        x = self.blocks(x)  # Transformer bloklarından geçir
        x = self.ln(x)  # Son LayerNorm
        return x
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
         # (B,T,vocab_size)

        return x

m=Encoder(vocab_size,n_embd,n_head,block_size).to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
xb, yb = get_batch('train')
output=m(xb).detach()
print(output.shape)"""
