import re
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return dataloader

raw_text = "Hello, do you like tea? Yes, I like tea very much. GPT is a great model!"

vocab_size = 50257
output_dim = 256
context_length = 4
batch_size = 2

dataloader = create_dataloader(
    raw_text, 
    batch_size=batch_size, 
    max_length=context_length, 
    stride=2
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

torch.manual_seed(123)
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

token_embeddings = token_embedding_layer(inputs)
pos_indices = torch.arange(context_length)
pos_embeddings = pos_embedding_layer(pos_indices)
input_embeddings = token_embeddings + pos_embeddings

print(f"1. Input IDs Shape:  {inputs.shape}")
print(f"2. Token Emb Shape:  {token_embeddings.shape}") 
print(f"3. Final Emb Shape:  {input_embeddings.shape}")
print("\nFirst sample input embeddings (partial):\n", input_embeddings[0, 0, :5])
