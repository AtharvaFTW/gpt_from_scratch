import torch
from torch import nn

class TokenEmbedding(nn.Module):
    """
    This class converts the tokens into vector space embeddings.
    """
    def __init__(self, vocab_size:int , d_model):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        
        x = self.embed(x)
        return x


class PositionalEncoding(nn.Module):
    """
    The Positional Encoding helps the model to know the sequence of the tokens.
    """
    def __init__(self, seq_len:int, d_model):
        super().__init__()

        z_matrix = torch.zeros(size = (seq_len, d_model))
        pos = torch.arange(seq_len).unsqueeze(1)

        i = torch.arange(d_model//2)
        denominator = torch.pow(10000,(2*i)/ d_model) 

        z_matrix[:,0::2] = torch.sin(pos/denominator)
        z_matrix[:,1::2] = torch.cos(pos/denominator)

        self.register_buffer("pe", z_matrix)
        

    def forward(self,x):
        sequence_length = x.shape[1]
        x = x + self.pe[:sequence_length,:]
        
        return x

class MultiHeadAttention(nn.Module):
    """
    Computes parallell scaled dot-product attention over multiple heads
    to allow the model to jointly attend to information from different representation subspaces.
    """
    def __init__(self, d_model, h:int):
        super().__init__()

    def forward(self,x):
        pass

class FeedForward(nn.Module):
    """
    FeedForward consists of the linear projects with ReLU.
    """
    def __init__(self, d_model):
        super().__init__()

    def forward(self,x):
        pass

class Decoder(nn.Module):
    """
    The decoder has 3 sublayers out of which 2 are same as the encoder with one addition that masks the tokens after the current one to prevent illegal connections.

    """
    def __init__(self, d_model, h:int):
        super().__init__()

    def forward(self,x):
        pass

class LMHead(nn.Module):
    """
    Linear layer that projects from embedding dimension to vocabulary size, converts embeddings back to token probabilities
    """
    def __init__(self, d_model, vocab_size:int):
        super().__init__()

    def forward(self,x):
        pass

class GPT(nn.Module):
    """
    This the high level class that puts all the components together.
    """
    def __init__(self, vocab_size:int, d_model, h, N, context_len: int):
        super().__init__()

    def forward(self,x):
        pass


if __name__ == "__main__":
    import torch
    data = torch.randint(32, (4,8))
    tokenizer = TokenEmbedding(32, 32)
    output = tokenizer(data)

    assert output.shape == (4, 8, 32)
    print("tokeninzer passed")
    
    pe = PositionalEncoding(100, 32)
    pe_output = pe(output)

    assert pe_output.shape == (4, 8, 32)
    print("positional encoder passed")