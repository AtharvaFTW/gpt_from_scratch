import tiktoken
import torch
import string

def get_tokens(path: str):
    """
    This function creates token encodings using tiktoken Byte-pair encoding tokenizer.
    """
    with open (path,"r") as f:
        corpus = f.read()

    enc = tiktoken.get_encoding("gpt2")

    data = enc.encode(corpus)

    split = int(len(data) * 0.9)

    train_data = data[:split]
    val_data = data[split:]

    train_tensor = torch.tensor(train_data, dtype = torch.long)
    val_tensor = torch.tensor(val_data, dtype = torch.long)

    return train_tensor, val_tensor


def get_tokens_character(path):
    """
    This function creates token embeddigns at character level.
    """

    with open(path, "r", encoding= "utf-8") as f:
        corpus = f.read()

    chars = sorted(set(corpus))
    vocab_size = len(chars)

    char_to_idx = {ch: i for i,ch in enumerate(chars)}
    idx_to_char = {i: ch for i,ch in enumerate(chars)}

    def encode(text):
        return [char_to_idx[ch] for ch in text]

    def decode(tokens):
        return "".join([idx_to_char[i] for i in tokens])

    encoded_text = encode(corpus)
    tokens = torch.tensor(encoded_text, dtype = torch.long)

    train_tensor = tokens[:int((0.9) * len(tokens))]
    val_tensor = tokens[int((0.9) * len(tokens)):]

    return train_tensor, val_tensor , vocab_size, encode, decode

def get_tokens_word(path):
    """
    This function creates token embeddings at word level.
    Word level embeddings allow the model to understand better.
    """

    with open(path, "r", encoding= "utf-8") as f:
        corpus = f.read()

    corpus = corpus.lower()
    corpus = corpus.translate(str.maketrans("", "", string.punctuation))

    words = sorted(set(corpus.split()))
    vocab_size = len(words)

    w_to_idx = {w: i for i,w in enumerate(words)}
    idx_to_w = {i: w for i,w in enumerate(words)}

    def encode(text):
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return [w_to_idx[w] for w in text.split()]

    def decode(tokens):
        return " ".join([idx_to_w[i] for i in tokens])

    encoded_text = encode(corpus)
    tokens = torch.tensor(encoded_text, dtype = torch.long)

    train_tensor = tokens[:int((0.9) * len(tokens))]
    val_tensor = tokens[int((0.9) * len(tokens)):]

    return train_tensor, val_tensor , vocab_size, encode, decode



def get_batch(data: torch.Tensor, batch_size:int, context_len:int):
    ix = torch.randint(len(data) - context_len, (batch_size,))

    x = torch.stack([data[i : i + context_len] for i in ix])
    y = torch.stack([data[i + 1: i + context_len + 1] for i in ix])

    return x, y


if __name__ == "__main__":

    path = r"data/weeknd.txt"
    train_tensor,val_tensor, vocab_size, encode, decode = get_tokens_word(path)
    print(f"Vocab size: {vocab_size}")
    print(f"Train Tensor len: {len(train_tensor)}")
    print(f"Val Tensor len: {len(val_tensor)}")
    print(f"Sample encode: {encode("Take me back to LA")}")
    print(f"Sample decode: {decode(encode("Take me back to LA"))}")
    