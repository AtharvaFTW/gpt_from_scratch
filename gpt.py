from torch.ao.nn.quantized import LayerNorm
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

class MaskedMultiHeadAttention(nn.Module):
    """
    Computes parallell scaled dot-product attention over multiple heads
    to allow the model to jointly attend to information from different representation subspaces.
    """
    def __init__(self, d_model, h:int):
        super().__init__()

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.h = h
        self.d_k = d_model//h

    def forward(self,x):
        batch, seq_len, d_model = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(batch, seq_len, self.h, self.d_k)
        K = K.view(batch, seq_len, self.h, self.d_k)
        V = V.view(batch, seq_len, self.h, self.d_k)

        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-1,-2))/torch.sqrt(torch.tensor(self.d_k, dtype = torch.float32))

        upper_mask = torch.triu(torch.ones_like(scores), diagonal= 1) == 1

        scores = scores.masked_fill(upper_mask, float("-inf"))
        scores = torch.softmax(scores, dim =-1)

        attention = torch.matmul(scores, V)
        attention = attention.transpose(1,2).contiguous()
        attention = attention.view(batch, seq_len, d_model)

        multi_head = self.W_O(attention)

        return multi_head

class FeedForward(nn.Module):
    """
    FeedForward consists of the linear projects with ReLU.
    """
    def __init__(self, d_model):
        super().__init__()

        d_ff = 4 * d_model
        self.feed = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
                    )

    def forward(self,x):
        x = self.feed(x)

        return x

class Decoder(nn.Module):
    """
    The decoder has 3 sublayers out of which 2 are same as the encoder with one addition that masks the tokens after the current one to prevent illegal connections.

    """
    def __init__(self, d_model, h:int):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.masked_attention = MaskedMultiHeadAttention(d_model, h)
        self.feed = FeedForward(d_model)

    def forward(self,x):
       x = x + self.masked_attention(x)
       x = self.norm1(x)

       x = x + self.feed(x)
       x = self.norm2(x)

       return x

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
    tokenizer = TokenEmbedding(32, 512)
    output = tokenizer(data)

    assert output.shape == (4, 8, 512)
    print("tokeninzer passed")
    
    pe = PositionalEncoding(100, 512)
    pe_output = pe(output)

    assert pe_output.shape == (4, 8, 512)
    print("positional encoder passed")

    mha = MaskedMultiHeadAttention(512, 8)
    mha_output = mha(pe_output)

    assert mha_output.shape == (4, 8, 512)
    print("mha  passed")

    ff = FeedForward(512)
    ff_output = ff(mha_output)

    assert ff_output.shape == (4, 8, 512)
    print("ff passed")

    deco = Decoder(512, 8)
    deco_output = deco(pe_output)

    assert deco_output.shape == (4, 8, 512)
    print("deco passed")