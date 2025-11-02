# --------------------------------------------
# Manual SHAP Analysis on Single Text Document
# --------------------------------------------
# Purpose: Load text from manual_shapley_data.txt and run SHAP explanation
# --------------------------------------------

import os
import json
import numpy as np
import shap
from shap.maskers import Text as ShapTextMasker
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import defaultdict
import re
import matplotlib.pyplot as plt
import csv

# Load the text data
def load_manual_data():
    with open("manual_shapley_data.txt", "r", encoding="utf-8") as f:
        return f.read().strip()

# Setup
MODEL_NAME = "all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

# Define a query for comparison
QUERY = "Effect of glucose administration or hyperglycemia management on outcomes after cardiac arrest."

def tokenize(s):
    return [t.lower() for t in re.split(r"\W+", s) if t.strip()]

def remove_stopwords_text(s: str) -> str:
    tokens = tokenize(s)
    filtered = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return " ".join(filtered)

def make_predict_fn(query_emb_vec: np.ndarray):
    def predict(texts):
        embs = model.encode(texts, normalize_embeddings=True)
        sims = util.cos_sim(query_emb_vec, embs).cpu().numpy()
        return sims.reshape(-1, 1)
    return predict

def explanation_to_token_scores(explanation_obj, top_n=15):
    e = explanation_obj[0]
    tokens = list(e.data)
    vals = list(e.values)
    agg = defaultdict(float)
    
    for t, v in zip(tokens, vals):
        if t.strip():
            # Skip stop words in outputs
            if t.lower() in ENGLISH_STOP_WORDS:
                continue
            # Extract scalar value
            try:
                from numpy import ravel
                v_scalar = float(ravel(v)[0])
            except Exception:
                v_scalar = float(v)
            agg[t] += v_scalar
    
    # Sort by absolute contribution
    ranked = sorted(agg.items(), key=lambda x: abs(x[1]), reverse=True)
    return ranked[:top_n]

def save_results(text, explanations, query, similarity_score):
    os.makedirs("manual_shap_outputs", exist_ok=True)
    
    # Save token contributions CSV
    with open("manual_shap_outputs/token_contributions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["token", "shap_value"])
        ranked = explanation_to_token_scores(explanations, top_n=30)
        for tok, val in ranked:
            w.writerow([tok, f"{val:.6f}"])
    
    # Save bar plot
    ranked = explanation_to_token_scores(explanations, top_n=15)
    tokens = [t for t, _ in ranked][::-1]
    values = [v for _, v in ranked][::-1]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(tokens)), values)
    plt.yticks(range(len(tokens)), tokens)
    plt.xlabel("SHAP contribution to similarity")
    plt.title(f"Manual Text SHAP Analysis\nQuery: {query[:60]}...\nSimilarity Score: {similarity_score:.3f}")
    plt.tight_layout()
    plt.savefig("manual_shap_outputs/token_contributions_bar.png", dpi=150)
    plt.close()
    
    # Save JSON with all token values
    try:
        e = explanations[0]
        tokens = [str(t) for t in list(e.data)]
        from numpy import ravel
        values = [float(ravel(v)[0]) if hasattr(v, "__array__") else float(v) for v in list(e.values)]
        payload = {
            "query": query,
            "similarity_score": float(similarity_score),
            "tokens": tokens,
            "shap_values": values
        }
        with open("manual_shap_outputs/full_explanation.json", "w", encoding="utf-8") as jf:
            json.dump(payload, jf, ensure_ascii=False, indent=2)
    except Exception as ex:
        print(f"Error saving JSON: {ex}")
    
    print("Results saved to manual_shap_outputs/")
    print(f"- token_contributions.csv")
    print(f"- token_contributions_bar.png") 
    print(f"- full_explanation.json")

def main():
    # Load the manual text
    text = load_manual_data()
    print(f"Loaded text: {len(text)} characters")
    print(f"Preview: {text[:100]}...")
    
    # Clean text and query
    cleaned_text = remove_stopwords_text(text)
    cleaned_query = remove_stopwords_text(QUERY)
    
    print(f"\nQuery: {QUERY}")
    print(f"Cleaned query: {cleaned_query}")
    
    # Calculate similarity score
    query_emb = model.encode([cleaned_query], normalize_embeddings=True)
    text_emb = model.encode([cleaned_text], normalize_embeddings=True)
    similarity_score = util.cos_sim(query_emb, text_emb).cpu().numpy()[0][0]
    
    print(f"Similarity score: {similarity_score:.3f}")
    
    # Run SHAP explanation
    print("\nRunning SHAP explanation...")
    predict_fn = make_predict_fn(query_emb)
    masker = ShapTextMasker(r"\W+")
    explainer = shap.Explainer(predict_fn, masker)
    
    # Explain the cleaned text
    exp = explainer([cleaned_text], max_evals=500)  # type: ignore[arg-type]
    
    # Get top contributing tokens
    ranked = explanation_to_token_scores(exp, top_n=10)
    print(f"\nTop 10 contributing tokens:")
    for i, (token, value) in enumerate(ranked, 1):
        print(f"{i:>2}. {token:<15} {value:>8.4f}")
    
    # Save all results
    save_results(cleaned_text, exp, QUERY, similarity_score)
    
    # Show lexical overlap for comparison
    query_terms = set([t for t in tokenize(QUERY) if t not in ENGLISH_STOP_WORDS])
    text_terms = set([t for t in tokenize(text) if t not in ENGLISH_STOP_WORDS])
    overlap = sorted(query_terms.intersection(text_terms))
    print(f"\nLexical overlap: {overlap}")

if __name__ == "__main__":
    main()
