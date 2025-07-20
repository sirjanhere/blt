import torch
import pickle
from model import EntropySentenceBoundaryModel

# Set the window size for the sliding prediction and the stride (how much to move the window each step)
WINDOW_SIZE = 50
STRIDE = 25

def load_model():
    """
    Loads the trained model checkpoint and vocabulary from disk.
    Returns:
        model: The trained EntropySentenceBoundaryModel ready for inference.
        vocab: The vocabulary dictionary mapping words to indices.
    """
    # Load the saved model checkpoint (includes model weights and vocab)
    checkpoint = torch.load("model.pt", map_location="cpu")
    # Initialize the model with the correct vocabulary size
    model = EntropySentenceBoundaryModel(len(checkpoint["vocab"]))
    # Load the learned parameters into the model
    model.load_state_dict(checkpoint["model"])
    # Set the model to evaluation mode (disables dropout, etc.)
    model.eval()
    return model, checkpoint["vocab"]

def sliding_predict(words, vocab, model, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    Performs sliding window prediction over the input words.
    For each window, computes the probability of each word being a sentence boundary,
    and averages predictions for overlapping words.
    
    Args:
        words: List of words (tokens) in the full text.
        vocab: Dictionary mapping words to indices.
        model: The trained PyTorch model for prediction.
        window_size: Number of words in each window.
        stride: Step size for sliding the window.
    
    Returns:
        result: List of probabilities (one per word), averaged for overlaps.
    """
    # Convert words to indices using the vocabulary (unknown words get index 0)
    X = [vocab.get(w, 0) for w in words]
    result = [0.0] * len(X) # To store the sum of probabilities for each word
    counts = [0] * len(X) # To count how many times each word is included in a window
    # Slide the window over the sequence
    for i in range(0, len(X) - window_size, stride):
        x = torch.tensor([X[i:i+window_size]], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)[0] # Get model's raw output (logits) for the window
            probs = torch.sigmoid(logits).cpu().numpy()  # Convert logits to probabilities
        for j in range(window_size):
            result[i+j] += probs[j]  # Accumulate probabilities
            counts[i+j] += 1  # Accumulate counts for averaging
    # Average the overlapping predictions for each word
    for i in range(len(result)):
        if counts[i] > 0:
            result[i] /= counts[i]
    return result

if __name__ == "__main__":
    # Load the preprocessed data (words, sentences)
    with open("data/processed.pkl", "rb") as f:
        data = pickle.load(f)
    words = data["words"]  # Flat list of all words in the text
    sentences = data["sentences"]  # List of tokenized sentences (each a list of words)
    
    # Load the trained model and vocabulary
    model, vocab = load_model()
    # Run the sliding window prediction for the whole text
    probs = sliding_predict(words, vocab, model)
    
    # Print the first 30 sentences and the predicted probabilities for each word
    idx = 0
    for i, sent in enumerate(sentences[:30]):  # show only first 30 for brevity
        print("Sentence {}: {}".format(i+1, " ".join(sent)))
        entropy_line = " ".join("{:.3f}".format(probs[idx+j]) for j in range(len(sent)))
        print(entropy_line)
        print()
        idx += len(sent)  # Move the index forward by sentence length