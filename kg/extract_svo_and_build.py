import os
import pickle
import csv
import networkx as nx
import spacy
from deepsegment import DeepSegment

def segment_sentences(text):
    segmenter = DeepSegment('en')
    return segmenter.segment_long(text)

def get_subjs(token):
    subjs = []
    for child in token.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            subjs.append(child)
        elif child.dep_ == "conj":
            subjs.extend(get_subjs(child))
    return subjs

def get_objs(token):
    objs = []
    for child in token.children:
        if child.dep_ in ("dobj", "obj", "iobj", "attr", "oprd", "ccomp", "xcomp"):
            objs.append(child)
        elif child.dep_ == "conj":
            objs.extend(get_objs(child))
    return objs

def get_noun_phrase(token):
    """
    Extracts the full noun phrase: includes determiners, compounds, adjectives, "of" prepositional phrases, but
    does NOT include trailing adverbials or unrelated phrases.
    """
    tokens = [token]
    # Add left modifiers (det, amod, compound, poss, nummod)
    tokens += [child for child in token.lefts if child.dep_ in {"det", "amod", "compound", "poss", "nummod"}]
    # Add right modifiers (amod, compound, nummod)
    tokens += [child for child in token.rights if child.dep_ in {"amod", "compound", "nummod"}]
    # For prepositional phrases attached to the noun (e.g., "of the fire")
    for child in token.rights:
        if child.dep_ == "prep" and child.text.lower() == "of":
            tokens += list(child.subtree)
        # If you want *all* attached prep phrases (e.g., "about his health"), include this instead:
        if child.dep_ == "prep":
            tokens += list(child.subtree)
    # Sort and join
    tokens = sorted(set(tokens), key=lambda x: x.i)
    return " ".join([t.text for t in tokens])

def extract_svo_spacy(sentence, nlp):
    doc = nlp(sentence)
    triplets = []
    for token in doc:
        if (token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}) or (token.dep_ == "conj" and token.pos_ in {"VERB", "AUX"}):
            subjs = get_subjs(token)
            objs = get_objs(token)
            if not subjs or not objs:
                continue
            for subj in subjs:
                subj_phrase = get_noun_phrase(subj)
                for obj in objs:
                    # For clausal objects, use full clause as text
                    if obj.dep_ in ("ccomp", "xcomp"):
                        obj_phrase = " ".join([t.text for t in sorted(obj.subtree, key=lambda x: x.i)])
                        triplets.append((subj_phrase, token.lemma_, obj_phrase))
                    else:
                        obj_phrase = get_noun_phrase(obj)
                        triplets.append((subj_phrase, token.lemma_, obj_phrase))
    return triplets

def main():
    input_path = "kg/input.txt"
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    sentences = segment_sentences(raw_text)
    nlp = spacy.load("en_core_web_sm")
    
    triplets = []
    for sent in sentences:
        triplets.extend(extract_svo_spacy(sent, nlp))

    os.makedirs("kg", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    csv_path = "kg/triples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "verb", "object"])
        writer.writerows(triplets)
    print(f"[INFO] Wrote {len(triplets)} triplets to {csv_path}")

    G = nx.DiGraph()
    for subj, verb, obj in triplets:
        G.add_node(subj)
        G.add_node(obj)
        G.add_edge(subj, obj, label=verb)
    
    with open("outputs/kg.pkl", "wb") as f:
        pickle.dump(G, f)
    print(f"[INFO] Knowledge graph saved to outputs/kg.pkl")

if __name__ == "__main__":
    main()