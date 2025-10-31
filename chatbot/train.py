import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pickle
from seq2seq_attention import Encoder, Decoder, Seq2Seq


class DialogueDataset(Dataset):
    def __init__(self, inputs, targets, vocab_size):
        self.inputs = inputs
        self.targets = targets
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_seq = torch.tensor(self.inputs[idx], dtype=torch.long)
        target_seq = torch.tensor(self.targets[idx], dtype=torch.long)
        input_seq = torch.clamp(input_seq, 0, self.vocab_size - 1)
        target_seq = torch.clamp(target_seq, 0, self.vocab_size - 1)
        return {'src': input_seq, 'trg': target_seq}

def secure_indices(tensor, vocab_size):
    return torch.clamp(tensor, 0, vocab_size - 1)


def train(model, dataloader, optimizer, criterion, device, clip=1.0):
    model.train()
    epoch_loss = 0
    vocab_size = model.encoder.embedding.num_embeddings
    for batch in tqdm(dataloader, desc="Training"):
        src = batch['src'].to(device)
        trg = batch['trg'].to(device)
        src = secure_indices(src, vocab_size)
        trg = secure_indices(trg, vocab_size)
        optimizer.zero_grad()
        output = model(src, trg)
        output_flat = output[:,1:].reshape(-1, vocab_size)
        trg_flat    = trg[:,1:].reshape(-1)
        trg_flat    = secure_indices(trg_flat, vocab_size)
        loss = criterion(output_flat, trg_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    vocab_size = model.encoder.embedding.num_embeddings
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            src = batch['src'].to(device)
            trg = batch['trg'].to(device)
            src = secure_indices(src, vocab_size)
            trg = secure_indices(trg, vocab_size)
            output = model(src, trg, teacher_forcing_ratio=0)
            output_flat = output[:,1:].reshape(-1, vocab_size)
            trg_flat    = trg[:,1:].reshape(-1)
            trg_flat    = secure_indices(trg_flat, vocab_size)
            loss = criterion(output_flat, trg_flat)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


with open("dataset_preprocessed.pkl", "rb") as f:
    datasets = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor_data = pickle.load(f)

tokenizer = preprocessor_data['tokenizer']
vocab_size = preprocessor_data['vocab_size']
print(f"Vocabulary size: {vocab_size}")


train_dataset = DialogueDataset(datasets['train'][0], datasets['train'][1], vocab_size)
val_dataset   = DialogueDataset(datasets['val'][0],   datasets['val'][1],   vocab_size)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=5)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIM  = vocab_size
OUTPUT_DIM = vocab_size
EMBED_DIM  = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT    = 0.2

encoder = Encoder(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
decoder = Decoder(OUTPUT_DIM, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)


N_EPOCHS = 5
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss   = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")


torch.save({
    'model_state_dict': model.state_dict(),
    'tokenizer': tokenizer,
    'vocab_size': vocab_size
}, "seq2seq_dialogue_model.pt")

print("✅ Entraînement terminé et modèle sauvegardé.")
