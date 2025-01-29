from encoder import output 
output = output.unsqueeze(0) if output.dim() == 2 else output  # (1, T, C) formatında olmalı
import sub_docs.config as config
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
from torch.nn import functional as F
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.0
# ------------


# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt


with open(r"C:\Users\bahaa\Downloads\Birinci Yurttaş_eos.txt", 'r', encoding='utf-8') as f:
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

    # Pad işlemi
    x_padded = pad_sequences(x_raw, maxlen=block_size, padding='post', value=0)
    y_padded = pad_sequences(y_raw, maxlen=block_size, padding='post', value=0)

    # Tensor’a çevirme
    x = torch.tensor(x_padded, dtype=torch.long, device=device)
    y = torch.tensor(y_padded, dtype=torch.long, device=device)

    current_index += batch_size  # Batch indeksini güncelle
    
    return x, y"""
# Başlangıç indexi
current_index = 0
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
class LayerNorm1d: # (used to be BatchNorm1d)

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    # calculate the forward pass
    xmean = x.mean(1, keepdim=True) # batch mean
    xvar = x.var(1, keepdim=True) # batch variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
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

class CrossAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    """def forward(self, x_1, x_2=output):
        B,T,C=x_1.shape
        queries_1 = self.query(x_1)
        keys_2 = self.key(x_2)
        values_2 = self.value(x_2)
        
        attn_scores = queries_1 @ keys_2.transpose(-2, -1)
        attn_weights = F.softmax(attn_scores / queries_1.size(-1) ** 0.5, dim=-1)
        context_vec = attn_weights @ values_2
        return context_vec"""
    def forward(self, x_1, x_2=output):
        # x_1: (batch_size, target_seq_len, embed_dim)
        # x_2: (batch_size, source_seq_len, embed_dim)
        B, T1, C = x_1.shape
        _, T2, _ = x_2.shape

        # Linearly project input tensors to query, key, and value
        queries = self.query(x_1)  # (B, T1, head_size)
        keys = self.key(x_2)      # (B, T2, head_size)
        values = self.value(x_2)  # (B, T2, head_size)

        # Compute attention scores (scaled dot-product attention)
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))  # (B, T1, T2)
        attn_scores = attn_scores / (C ** 0.5)  # Scale by the square root of head_size

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T1, T2)
        attn_weights = self.dropout(attn_weights)

        # Compute context vectors
        context_vec = torch.matmul(attn_weights, values)  # (B, T1, head_size)

        return context_vec
        
"""class MultiCrossAttention(nn.Module):
    

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([CrossAttention(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out"""
class MultiCrossAttention(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embed dim must be divisible by num_heads"
        head_size = embed_dim // num_heads
        self.heads = nn.ModuleList([CrossAttention(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)  # Combine all heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_1, x_2=output):
        # Multi-Head Cross Attention
        out = torch.cat([head(x_1, x_2) for head in self.heads], dim=-1)  # (B, T, embed_dim)
        out = self.dropout(self.proj(out))  # Final projection
        return out


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
        self.cratt = MultiCrossAttention(n_head,n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.cratt(self.ln3(x),output)
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class AltanTranslator(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(4)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B,T=idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            index_cond = index[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(index_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index
    





model = AltanTranslator()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    
