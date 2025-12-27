'''
GPT uses learned absolute positional embeddings (not relative ones).
'''

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
'''
Token IDs:
 tensor([[   40,   367,  2885,  1464],
        [ 1807,  3619,   402,   271],
        [10899,  2138,   257,  7026],
        [15632,   438,  2016,   257],
        [  922,  5891,  1576,   438],
        [  568,   340,   373,   645],
        [ 1049,  5975,   284,   502],
        [  284,  3285,   326,    11]])

Inputs shape:
 torch.Size([8, 4])
 '''
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)   # torch.Size([8, 4, 256])

pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)

''' 
8 x 4 x 256  + 4 x 256 will broastcast for absolute positional embedding
'''

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape) # torch.Size([8, 4, 256])
