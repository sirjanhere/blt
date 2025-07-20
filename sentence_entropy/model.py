import torch.nn as nn

class EntropySentenceBoundaryModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, 128, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(256, 1)
        # DO NOT include self.sigmoid

    def forward(self, x):
        emb = self.emb(x)
        h, _ = self.lstm(emb)
        logits = self.fc(h)
        return logits.squeeze(-1)  # Do not apply sigmoid here!