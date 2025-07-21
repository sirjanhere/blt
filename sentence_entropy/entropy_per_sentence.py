import torch
import pickle
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

model_name = "gpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_token_entropy(logits):
    # logits: [seq_len, vocab_size]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.cpu().tolist()


def compute_sentence_entropy(sentence):
    """
    Compute token-level entropy for a given sentence.
    Returns: (tokens, entropies)
    """
    text = " ".join(sentence)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    logits = logits[0][:-1]      # Drop last token (no next-token for it)
    input_ids = input_ids[0][1:] # Shift input_ids to align with prediction

    entropy = compute_token_entropy(logits)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    return tokens, entropy


def align_tokens_to_words(words, tokens, entropy):
    """
    Aligns GPT-2 tokens to original words. Handles subwords.
    Returns entropy list, one per word.
    """
    word_entropies = []
    word_idx = 0
    token_idx = 0

    while word_idx < len(words) and token_idx < len(tokens):
        word = words[word_idx]
        collected_entropy = []
        reconstructed = ""
        while token_idx < len(tokens) and len(reconstructed.replace("Ġ", "").replace("▁", "").replace("Ċ", "")) < len(word):
            tok = tokens[token_idx]
            clean_tok = tok.replace("Ġ", "").replace("▁", "").replace("Ċ", "")
            reconstructed += clean_tok
            collected_entropy.append(entropy[token_idx])
            token_idx += 1

        # Fallback if no entropy collected
        if not collected_entropy:
            collected_entropy = [0.0]

        avg_entropy = sum(collected_entropy) / len(collected_entropy)
        word_entropies.append(avg_entropy)
        word_idx += 1

    # If mismatch, pad with zeros
    while len(word_entropies) < len(words):
        word_entropies.append(0.0)

    return word_entropies


def main():
    with open("data/processed.pkl", "rb") as f:
        data = pickle.load(f)

    sentences = data["sentences"][:30]

    for i, sentence in enumerate(sentences):
        print(f"\nSentence {i+1}: {' '.join(sentence)}")
        try:
            tokens, entropy = compute_sentence_entropy(sentence)
            word_entropy = align_tokens_to_words(sentence, tokens, entropy)
            print("Entropies: ", " ".join(f"{e:.4f}" for e in word_entropy))
        except Exception as e:
            print("⚠️ Skipping due to error:", e)


if __name__ == "__main__":
    main()
