import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.embedding_dim = d_model
        self.tokenEmbedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self,x):
        x = self.tokenEmbedding(x)
        return x