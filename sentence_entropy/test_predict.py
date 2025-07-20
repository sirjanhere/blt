import pickle
from predict import load_model, predict_sentence_starts

with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)
words, labels, sentences = data["words"], data["labels"], data["sentences"]

model, vocab = load_model()
starts, probs = predict_sentence_starts(words, vocab, model)

tp = sum(labels[i] for i in starts)
fp = len(starts) - tp
fn = sum(labels) - tp

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision+recall) > 0 else 0

print(f"\nPrecision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
print("Predicted sentence starts (word indices):", starts)
for idx in starts:
    print(f"Word {idx}: {words[idx]}")