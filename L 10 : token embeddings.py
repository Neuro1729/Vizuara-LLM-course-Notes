vocab_size = 50,000
output_dim = 768

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer(torch.tensor([300])))  # gives embedding of token of ID : 300
input_ids = torch.tensor([300, 5010, 8963, 120])
print(embedding_layer(input_ids)) 
'''
Output:
tensor([[ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-2.8400, -0.7849, -1.4096],
        [ 0.9178,  1.5810,  1.3010]], grad_fn=<EmbeddingBackward0>)
'''
