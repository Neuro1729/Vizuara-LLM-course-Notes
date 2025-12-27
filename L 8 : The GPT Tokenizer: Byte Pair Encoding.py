# word based tokenizer problem : [boy,boys] may have vastly differed ID 
# BPE algo : most common pair of consecutive bytes of data is replaced with a byte that doesn't occur in data
'''
ex: aaabdaaabac --> ZabdZabac --> ZYdZYac --> WdWac
BPE ensures most common words in vocab are represented by single token ,while rare words are broken down into two or more subwords

tiktoken is a BPE tokenizer by Openai , cl100k_base is Common Language, 100k-token base vocabulary It has about 100,000 learned byte-chunks like: "ing" , "tion" ,"🚀" All discovered automatically by Byte-Pair Encoding.
'''

import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

text = "Hello GPT 🚀"
tokens = enc.encode(text)
decoded = enc.decode(tokens)

print(tokens)
print(decoded)
