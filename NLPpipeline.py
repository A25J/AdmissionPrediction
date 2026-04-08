import spacy
import textstat
from typing import Dict
import numpy as np
from docx import Document
#load spacy model
nlp = spacy.load('en_core_web_lg')

#feature extraction
def extract_linguistic_features(doc):
    features = {}
    features["readability"] = textstat.flesch_reading_ease(doc.text)
    features["sentence_length"] = np.mean([len(sent) for sent in doc.sents])
    features["oov_ratio"] = sum(t.is_oov for t in doc) / len(doc) #out of vocab
    return features

def extract_semantic_features(doc):
    features = {}

    sentences = list(doc.sents)
    if len(sentences) > 1 :
        sims = [s1.similarity(s2) for s1, s2 in zip(sentences[:-1], sentences[1:])] #0: unrelated meaning
        features["semantic_coherence"] = float(np.mean(sims))
    else:
        features["semantic_coherence"] = 0

    positive_words = ["excellent", "strong", "outstanding", "motivated"]
    negative_words = ["weak", "poor", "problem", "bad"]

    tokens = doc.text.lower().split()
    pos = sum(w in positive_words for w in tokens)
    neg = sum(w in negative_words for w in tokens)

    features["sentiment_score"] = (pos - neg) / max(1, len(tokens))
    return features

def extract_style_features(doc):
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    vocab = set(tokens)

    features = {
        "lexical_diversity": len(vocab) / len(tokens),
        "type_token_ratio": len(vocab) / max(1, len(tokens))
    }
    return features

def extract_structural_features(doc):
    text = doc.text

    # Paragraphs inferred from newlines
    paragraphs = [p for p in text.split("\n") if p.strip()]
    transition_doc = Document("Transitionals.docx")
    transition_text = " ".join(p.text for p in transition_doc.paragraphs if p.text.strip())
    transition_words = [w.strip().lower() for w in transition_text.split(",") if w.strip()]

    # Count transitions in the text
    transitions_count = sum(word in text.lower() for word in transition_words)


    features = {
        "paragraphs": len(paragraphs),
        "transitions": transitions_count
    }
    return features


def extract_all_features(text : str) -> Dict[str, float]:
    doc = nlp(text)

    features = {}
    features.update(extract_linguistic_features(doc))
    features.update(extract_semantic_features(doc))
    features.update(extract_style_features(doc))
    features.update(extract_structural_features(doc))
    return features

#Numerical scoring /5

def normalize_score(x, min_val, max_val):
    """Normalize the value to 0-1 then convert to 0-5 scale"""
    x = max(min(x, max_val), min_val)
    normalized = (x - min_val) / (max_val - min_val)
    return round(normalized*5, 2)
def print_features(features):
    print("{:<25} {:<12}".format("Feature", "Value"))
    print("-" * 40)

    for key, value in features.items():
        if isinstance(value, float):
            print("{:<25} {:<12.4f}".format(key, value))
        else:
            print("{:<25} {:<12}".format(key, value))
def score_document(features: Dict[str, float]) -> float:
    """Convert NLP Features into a single score /5.
    All feature weights are adjustable for the ML Pipeline"""

    weights = {
        "readability": 0.20,
        "semantic_coherence": 0.30,
        "lexical_diversity": 0.20,
        "sentiment_score": 0.10,
        "transitions": 0.10,
        "sentence_length": 0.10
    }
    score = 0
    score += weights["readability"] * normalize_score(features["readability"], 0, 100)
    score += weights["sentence_length"] * normalize_score(features["sentence_length"], 10, 30)
    score += weights["semantic_coherence"] * normalize_score(features["semantic_coherence"], 0, 1)
    score += weights["lexical_diversity"] * normalize_score(features["lexical_diversity"], 0.3, 0.8)
    score += weights["sentiment_score"] * normalize_score(features["sentiment_score"], -0.2, 0.2)
    score += weights["transitions"] * normalize_score(features["transitions"], 0, 5)
    print_features(features)
    return round(score, 2)

#testing the program before using in the final model
###########################################################
# if __name__ == "__main__":
#     print("\n=== SOP + LOR Evaluation System ===")
#     sop_text = input("\nPaste the Statement of Purpose (SOP):\n")
#     lor_text = input("\nPaste the Letter of Recommendation (LOR):\n")
#
#     sop_features = extract_all_features(sop_text)
#     lor_features = extract_all_features(lor_text)
#
#     sop_score = score_document(sop_features)
#     lor_score = score_document(lor_features)
#
#     print("\n=======================================")
#     print("            Final Score                   ")
#     print("=======================================")
#     print(f"SOP Score: {sop_score} /5")
#     print(f"LOR Score: {lor_score} /5")

