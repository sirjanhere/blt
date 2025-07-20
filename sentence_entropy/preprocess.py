import re
import pickle
from pathlib import Path

def get_punct_sentences(text):
    # Remove Gutenberg header/footer
    main_text = re.split(r"\*{3} START OF.*? \*{3}", text, flags=re.DOTALL)[-1]
    main_text = re.split(r"\*{3} END OF.*? \*{3}", main_text, flags=re.DOTALL)[0]
    # Remove empty lines
    main_text = "\n".join([line for line in main_text.split('\n') if line.strip()])
    # Split on sentence-ending punctuation followed by whitespace and a capital letter
    sents = re.split(r"(?<=[\.\!\?])[\s\"\']+(?=[A-ZА-ЯЁ])", main_text)
    return sents

def label_sentence_starts_unpunct(sents):
    words = []
    labels = []
    sentences = []
    for sent in sents:
        # Remove all punctuation, lowercase, re-tokenize
        sent = re.sub(r'[^\w\s\']', '', sent)
        tokens = sent.strip().lower().split()
        if not tokens:
            continue
        sentences.append(tokens)
        for i, tok in enumerate(tokens):
            words.append(tok)
            labels.append(1 if i == 0 else 0)
    return words, labels, sentences

def preprocess_text(input_path, output_path):
    text = Path(input_path).read_text(encoding="utf-8")
    sents = get_punct_sentences(text)
    words, labels, sentences = label_sentence_starts_unpunct(sents)
    with open(output_path, "wb") as f:
        pickle.dump({"words": words, "labels": labels, "sentences": sentences}, f)
    print(f"Processed {len(words)} words, {sum(labels)} sentence starts, {len(sentences)} sentences.")

if __name__ == "__main__":
    preprocess_text("data/war_and_peace.txt", "data/processed.pkl")