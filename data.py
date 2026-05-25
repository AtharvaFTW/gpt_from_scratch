import tiktoken
import torch

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


def get_batch(data: torch.Tensor, batch_size:int, context_len:int):
    ix = torch.randint(len(data) - context_len, (batch_size,))

    x = torch.stack([data[i : i + context_len] for i in ix])
    y = torch.stack([data[i + 1: i + context_len + 1] for i in ix])

    return x, y


if __name__ == "__main__":

    path = "shakespeare.txt"
    train, val = get_tokens(path)

    x,y = get_batch(train, 4, 8)
    print("X:", x.shape)
    print("y:", y.shape)