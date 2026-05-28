from data import get_tokens_word
import torch
from torch import nn
from components import GPT
from data import get_tokens_character , get_batch
from tqdm.auto import tqdm
import wandb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def weeknd_training_character(data_path:str,epochs:int= 3000, device= DEVICE):
    torch.manual_seed(42)
    
    train_data, val_data, vocab_size, encode, decode  = get_tokens_character(data_path)

    # Initializing Params
    VOCAB_SIZE = vocab_size
    D_MODEL = 768
    H = 12
    N = 8
    CONTEXT_LEN = 256
    BATCH_SIZE = 128

    model = GPT(vocab_size= VOCAB_SIZE, d_model= D_MODEL, h= H, N= N, context_len= CONTEXT_LEN).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    wandb.init(project = "WeekndGPT")
    for epoch in tqdm(range(epochs)):

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

        wandb.log({
                "epoch": epoch,
                "train_loss": loss,
                "test_loss": test_loss   
            })

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f} ")
            torch.save(model.state_dict(), f"weights_at_{epoch}.pt")
            artifact_training = wandb.Artifact(f"weeknd-weights_at_{epoch}" ,type = "model")
            artifact_training.add_file(f"weights_at_{epoch}.pt")
            wandb.log_artifact(artifact_training)
    torch.save(model.state_dict(),f"final_weights.pt")
    artifact = wandb.Artifact("weeknd-weights" ,type = "model")
    artifact.add_file('final_weights.pt')
    wandb.log_artifact(artifact)
    wandb.finish()


def weeknd_training_word(data_path:str,epochs:int= 3000, device= DEVICE):
    torch.manual_seed(42)
    
    train_data, val_data, vocab_size, encode, decode  = get_tokens_word(data_path)

    # Initializing Params
    VOCAB_SIZE = vocab_size
    D_MODEL = 768
    H = 12
    N = 8
    CONTEXT_LEN = 128
    BATCH_SIZE = 128

    model = GPT(vocab_size= VOCAB_SIZE, d_model= D_MODEL, h= H, N= N, context_len= CONTEXT_LEN).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    wandb.init(project = "WeekndGPT")
    for epoch in tqdm(range(epochs)):

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

        wandb.log({
                "epoch": epoch,
                "train_loss": loss,
                "test_loss": test_loss   
            })

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f} ")
            torch.save(model.state_dict(), f"weights_at_{epoch}_word.pt")
            artifact_training = wandb.Artifact(f"weights_at_{epoch}_word" ,type = "model")
            artifact_training.add_file(f"weights_at_{epoch}_word.pt")
            wandb.log_artifact(artifact_training)
    torch.save(model.state_dict(),f"final_weights_word.pt")
    artifact = wandb.Artifact("weeknd-weights" ,type = "model")
    artifact.add_file('final_weights_word.pt')
    wandb.log_artifact(artifact)
    wandb.finish()

if __name__ == "__main__":
    weeknd_training_word(r"data/weeknd.txt", epochs= 5000, device = DEVICE)