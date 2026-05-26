import torch
from torch import nn
from components import GPT
from data import get_tokens , get_batch
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_data, val_data = get_tokens("shakespeare.txt")

def training(data_path:str,epochs:int= 3000, device= DEVICE):

    torch.manual_seed(42)
    
    # Initializing Params
    VOCAB_SIZE = 50257
    D_MODEL = 512
    H = 8
    N = 6
    CONTEXT_LEN = 128
    BATCH_SIZE = 64

    train_data, val_data = get_tokens(data_path)
    
    model = GPT(vocab_size= VOCAB_SIZE, d_model= D_MODEL, h= H, N= N, context_len= CONTEXT_LEN).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), desc= f"GPT is Training on {data_path}"):

        model.train()

        x,y = get_batch(train_data, batch_size=BATCH_SIZE, context_len= CONTEXT_LEN)
        x,y = x.to(device), y.to(device)

        y_logits = model(x)

        yp_bz, yp_cl, yp_vc = y_logits.shape
        yt_bz, yt_cl = y.shape


        loss = loss_fn(y_logits.view(yp_bz * yp_cl, yp_vc), y.view(yt_bz * yt_cl))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()

        x_test, y_test = get_batch(val_data, batch_size= BATCH_SIZE, context_len= CONTEXT_LEN)
        x_test, y_test = x_test.to(device), y_test.to(device)

        with torch.inference_mode():

            test_logits = model(x_test)

            typ_bz, typ_cl, typ_vc = test_logits.shape
            tyt_bz, tyt_cl = y_test.shape

            test_loss = loss_fn(test_logits.view(typ_bz * typ_cl, typ_vc), y_test.view(tyt_bz * tyt_cl))

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f} ")
            torch.save(model.state_dict(), f"weights_at_{epoch}.pt")
        torch.save(model.state_dict(),f"final_weights.pt")

if __name__ == "__main__":
    training("shakespeare.txt")



