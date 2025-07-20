import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from model import EntropySentenceBoundaryModel
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm

WINDOW_SIZE = 50
STRIDE = 25
BATCH_SIZE = 128
EPOCHS = 10

class SBDDataset(Dataset):
    def __init__(self, words, labels, vocab, window_size=WINDOW_SIZE, stride=STRIDE):
        X = [vocab.get(w, 0) for w in words]
        y = labels
        self.samples = []
        for i in range(0, len(X) - window_size, stride):
            self.samples.append( (X[i:i+window_size], y[i:i+window_size]) )
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.float32)

# Load data
with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)
words, labels = data["words"], data["labels"]
counter = Counter(words)
max_vocab = 50000
vocab = {w: i+1 for i, (w, _) in enumerate(counter.most_common(max_vocab))}
vocab["<unk>"] = 0

# Dataset and DataLoader
dataset = SBDDataset(words, labels, vocab)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Model, Optimizer, Loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EntropySentenceBoundaryModel(len(vocab)).to(device)

# Class imbalance: pos_weight = (#neg / #pos)
pos_weight = torch.tensor([(len(labels) - sum(labels)) / sum(labels)]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=0.002)

# Training loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for X, y in tqdm(loader, desc=f"Epoch {epoch+1}"):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

torch.save({"model": model.state_dict(), "vocab": vocab}, "model.pt")
print("Model trained and saved.")