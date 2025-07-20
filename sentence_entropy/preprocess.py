import re
import pickle
from pathlib import Path
import spacy

# Load the spaCy English language model.
# The 'en_core_web_sm' model is a small English model for tokenization, tagging, parsing, etc.
nlp = spacy.load("en_core_web_sm")
# Increase the maximum text length spaCy can process to handle large texts like War and Peace.
nlp.max_length = 4_000_000 

def get_punct_sentences(text):
    """
    Given the full text of a book, removes the Project Gutenberg header/footer, 
    cleans up empty lines, and splits the text into sentences using spaCy's
    robust sentence tokenizer.
    Returns a list of sentences as strings.
    """
    # Remove Gutenberg header/footer
    main_text = re.split(r"\*{3} START OF.*? \*{3}", text, flags=re.DOTALL)[-1]
    main_text = re.split(r"\*{3} END OF.*? \*{3}", main_text, flags=re.DOTALL)[0]
    # Remove empty lines
    main_text = "\n".join([line for line in main_text.split('\n') if line.strip()])
    # Use spaCy's sentence tokenizer
    # Use spaCy's built-in sentence boundary detector.
    doc = nlp(main_text)
    sents = [sent.text for sent in doc.sents] # Extract sentences as strings.
    return sents
    return sents

def label_sentence_starts_unpunct(sents):
    """
    Takes a list of sentence strings, and for each sentence:
    - Removes punctuation (except apostrophes), 
    - Converts all words to lowercase,
    - Splits into tokens (words),
    - Assigns a label to each word: 1 if it's the first word of a sentence, 0 otherwise.

    Returns:
        words:    List of all words in the text (flattened).
        labels:   List of 1/0 labels (1 = sentence start).
        sentences:List of tokenized sentences (each a list of words).
    """
    words = []
    labels = []
    sentences = []
    for sent in sents:
        # Remove all punctuation, lowercase, re-tokenize
        sent = re.sub(r'[^\w\s\']', '', sent)
        tokens = sent.strip().lower().split()
        if not tokens:
            continue # Skip any empty sentences.
        sentences.append(tokens)
        for i, tok in enumerate(tokens):
            words.append(tok)
            # Label is 1 for first word in sentence, else 0.
            labels.append(1 if i == 0 else 0)
    return words, labels, sentences

def preprocess_text(input_path, output_path):
    """
    Coordinates the full preprocessing pipeline:
    - Reads raw text from input_path,
    - Splits into sentences,
    - Converts sentences to unpunctuated, lowercased tokens,
    - Labels sentence starts,
    - Saves the processed data as a pickle file for later model training.
    """
    # Read the full raw text file (e.g., War and Peace).
    text = Path(input_path).read_text(encoding="utf-8")
    # Split the text into sentences using spaCy.
    sents = get_punct_sentences(text)
    # Tokenize and label the sentences.
    words, labels, sentences = label_sentence_starts_unpunct(sents)
    # Save the processed data to a pickle file.
    with open(output_path, "wb") as f:
        pickle.dump({"words": words, "labels": labels, "sentences": sentences}, f)
    # Print summary statistics for demo/reporting.
    print(f"Processed {len(words)} words, {sum(labels)} sentence starts, {len(sentences)} sentences.")

if __name__ == "__main__":
    # Run the preprocessing pipeline on the provided input and output file paths.
    preprocess_text("data/war_and_peace.txt", "data/processed.pkl")