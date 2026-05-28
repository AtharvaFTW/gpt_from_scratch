
import torch
from components import GPT
import tiktoken

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 50257
D_MODEL = 512
H = 8
N = 6
CONTEXT_LEN = 128
BATCH_SIZE = 64




def generate(model, seed_text, max_new_tokens, context_len, device = DEVICE, enc = None):

    context = enc.encode(seed_text)
    context = torch.tensor(context, dtype= torch.long).unsqueeze(dim=0).to(device)
    model.eval()

    for i in range(max_new_tokens):

        logits = model(context[:,-context_len:])
        probs = torch.softmax(logits[:, -1, :], dim = -1)
        next_token = torch.multinomial(probs, num_samples=1)

        context = torch.cat([context, next_token], dim = 1)

    context = context.squeeze(0)
    context = context.tolist()

    res = enc.decode(context)

    return res

def weeknd_generate(model, seed_text, max_new_tokens, context_len, encode, decode, device = DEVICE):

    context = encode(seed_text)
    context = torch.tensor(context, dtype = torch.long).unsqueeze(dim =0).to(device)
    model.eval()

    for i in range(max_new_tokens):

        logits = model(context[:, -context_len:])
        probs = torch.softmax(logits[:, -1, :], dim = -1)
        next_token = torch.multinomial(probs, num_samples=1)

        context = torch.cat([context, next_token], dim = 1)

    context = context.squeeze(0)
    context = context.tolist()

    res = decode(context)

    return res


if __name__ == "__main__":
    from data import get_tokens_character

    _ , _, vc, encode, decode = get_tokens_character(r"data/weeknd.txt")

    model = GPT(vocab_size= vc, d_model= D_MODEL, h= H, N= N, context_len= CONTEXT_LEN).to(DEVICE)
    model.load_state_dict(torch.load("weeknd_weights.pt"))

    
    # enc = tiktoken.get_encoding("gpt2")
    output = weeknd_generate(model, "I am intoxicated", max_new_tokens= 1000, context_len= CONTEXT_LEN, encode = encode, decode = decode)
    print(output)