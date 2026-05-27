from torch.utils._cxx_pytree import args
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

model = GPT(vocab_size= VOCAB_SIZE, d_model= D_MODEL, h= H, N= N, context_len= CONTEXT_LEN).to(DEVICE)
model.load_state_dict(torch.load("weights\weights_at_900.pt"))

enc = tiktoken.get_encoding("gpt2")


def generate(model, seed_text, max_new_tokens, context_len, device = DEVICE):

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

if __name__ == "__main__":
   
    output = generate(model, "To be or not", max_new_tokens= 100, context_len= CONTEXT_LEN)
    print(output)