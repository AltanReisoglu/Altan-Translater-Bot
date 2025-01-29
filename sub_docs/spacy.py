with open(r"C:\Users\bahaa\Downloads\First Citizen.txt", 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars_en = sorted(list(set(text)))
vocab_size = len(chars_en)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars_en) }
itos = {value: key for key, value in stoi.items()}
encode_en = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode_en = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
print(itos)