import re
from collections import Counter

def build_vocab_from_file(path):
    text = open(path, "r", encoding="utf-8").read()
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens = [t.strip() for t in tokens if t.strip()]

    vocab = {"<|unk|>": 0, "<|eot|>": 1}
    for t in sorted(set(tokens)):
        vocab[t] = len(vocab)

    return vocab

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
