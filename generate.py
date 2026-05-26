import torch
from components import GPT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 50257
D_MODEL = 512
H = 8
N = 6
CONTEXT_LEN = 128
BATCH_SIZE = 64

model = GPT(vocab_size= VOCAB_SIZE, d_model= D_MODEL, h= H, N= N, context_len= CONTEXT_LEN).to(DEVICE)

model.load_state_dict(torch.load("weights\weights_at_900.pt"))


def generate(model, seed_text, max_new_tokens, context_len):
    model.eval()
    pass 