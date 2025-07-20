import torch
import pickle
from model import EntropySentenceBoundaryModel

WINDOW_SIZE = 50
STRIDE = 25

def load_model():
    checkpoint = torch.load("model.pt", map_location="cpu")
    model = EntropySentenceBoundaryModel(len(checkpoint["vocab"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["vocab"]

def sliding_predict(words, vocab, model, window_size=WINDOW_SIZE, stride=STRIDE):
    X = [vocab.get(w, 0) for w in words]
    result = [0.0] * len(X)
    counts = [0] * len(X)
    for i in range(0, len(X) - window_size, stride):
        x = torch.tensor([X[i:i+window_size]], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)[0]
            probs = torch.sigmoid(logits).cpu().numpy()
        for j in range(window_size):
            result[i+j] += probs[j]
            counts[i+j] += 1
    # average overlapping predictions
    for i in range(len(result)):
        if counts[i] > 0:
            result[i] /= counts[i]
    return result

if __name__ == "__main__":
    with open("data/processed.pkl", "rb") as f:
        data = pickle.load(f)
    words = data["words"]
    sentences = data["sentences"]

    model, vocab = load_model()
    probs = sliding_predict(words, vocab, model)

    idx = 0
    for i, sent in enumerate(sentences[:30]):  # show only first 30 for brevity
        print("Sentence {}: {}".format(i+1, " ".join(sent)))
        entropy_line = " ".join("{:.3f}".format(probs[idx+j]) for j in range(len(sent)))
        print(entropy_line)
        print()
        idx += len(sent)