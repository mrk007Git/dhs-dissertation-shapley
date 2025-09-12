# --------------------------------------------
# XAI for Retrieval: SHAP on a Semantic Model
# --------------------------------------------
# Purpose: Explain why a semantic model ranked a document highly
# Query: Your PICO Query 1 (ACLS training and skill retention)
# Outputs: SHAP text plots (Jupyter), bar plots, CSV of token contributions
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

# -----------------------
# 1) Inputs: query + docs
# -----------------------
QUERY = (
    "Effect of structured ACLS training on skill retention and performance among medical students "
    "and early-career healthcare providers."
)

# Replace this toy list with the top 10 for your query from your 424 ACLS articles.
DOCS = [
    "Randomized simulation-based ACLS training improves skill retention in senior medical students after three months.",
    "Basic life support refresher courses in emergency trainees: knowledge decay and competence in megacode scenarios.",
    "Effectiveness of structured advanced cardiac life support certification on clinical performance during in-hospital cardiac arrest.",
    "Surgical rotation exposure and non-cardiac operating skills among interns: a cross-sectional analysis.",
    "High-fidelity simulation for resuscitation: impact on performance, retention, and confidence among early-career providers."
]

DOC_IDS = [f"D{i+1}" for i in range(len(DOCS))]  # handy labels

# -----------------------
# 2) Semantic model setup
# -----------------------
# You can swap to a biomedical model like "pritamdeka/S-BioBert-snli-scinli" or similar
MODEL_NAME = "all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

# Compute similarity scores
query_emb = model.encode([QUERY], normalize_embeddings=True)  # 1 x d
doc_embs = model.encode(DOCS, normalize_embeddings=True)      # n x d
scores = util.cos_sim(query_emb, doc_embs).cpu().numpy().flatten()  # relevance scores

# Rank docs by semantic score
ranked_idx = np.argsort(scores)[::-1]
TOP_K = min(3, len(DOCS))  # explain top 3 by default
top_idx = ranked_idx[:TOP_K]

print("Top results by semantic similarity:")
for r, i in enumerate(top_idx, start=1):
    print(f"{r:>2}. {DOC_IDS[i]}  score={scores[i]:.3f}  {DOCS[i][:90]}...")

# ------------------------------------------
# 3) Wrap the scoring function for SHAP use
# ------------------------------------------
# SHAP expects a function f(text_list) -> array of prediction scores
# For retrieval, the "prediction" is the cosine similarity to the fixed query.
def predict(texts):
    embs = model.encode(texts, normalize_embeddings=True)
    sims = util.cos_sim(query_emb, embs).cpu().numpy()
    # shape: (1, n_texts) -> return shape (n_texts,) or (n_texts,1)
    return sims.reshape(-1, 1)

# Text masker splits on non-word characters; change pattern if you want finer control
masker = ShapTextMasker(r"\W+")

# SHAP Explainer: model-agnostic text explainer around our predict() function
explainer = shap.Explainer(predict, masker)

# ------------------------------------
# 4) Explain the top-ranked documents
# ------------------------------------
# Keep explanations quick by limiting evaluations; increase if time allows
EXPLANATIONS = []
for i in top_idx:
    text = DOCS[i]
    exp = explainer([text], max_evals=500)  # type: ignore[arg-type]
    EXPLANATIONS.append((DOC_IDS[i], text, exp))

# -------------------------------
# 5) Visualise and save artefacts
# -------------------------------
# If running in Jupyter, shap.plots.text(EXPLANATIONS[0][2][0]) will render inline.
# Also create a bar plot of token contributions for each explained doc.

import matplotlib.pyplot as plt

os.makedirs("shap_outputs", exist_ok=True)

def explanation_to_token_scores(explanation_obj, top_n=15):
    # Convert SHAP text explanation to a sorted list of (token, shap_value)
    # This relies on explanation_obj[0] since we passed a single text each time
    e = explanation_obj[0]
    # e.values is per-token contribution, e.data are tokens
    tokens = list(e.data)
    vals = list(e.values)
    # Aggregate repeated tokens by sum of contributions
    agg = defaultdict(float)
    for t, v in zip(tokens, vals):
        if t.strip():
            # v can be a 0-d/1-d numpy array; extract a scalar robustly
            try:
                from numpy import ravel
                v_scalar = float(ravel(v)[0])
            except Exception:
                v_scalar = float(v)
            agg[t] += v_scalar
    # Sort by absolute contribution, then sign
    ranked = sorted(agg.items(), key=lambda x: abs(x[1]), reverse=True)
    return ranked[:top_n]

# Save CSV with top token contributions
import csv
with open("shap_outputs/top_token_contributions.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["doc_id", "token", "shap_value"])
    for doc_id, text, exp in EXPLANATIONS:
        ranked = explanation_to_token_scores(exp, top_n=30)
        for tok, val in ranked:
            w.writerow([doc_id, tok, f"{val:.6f}"])

print("\nSaved: shap_outputs/top_token_contributions.csv")

# Bar plots per doc
for doc_id, text, exp in EXPLANATIONS:
    ranked = explanation_to_token_scores(exp, top_n=15)
    tokens = [t for t, _ in ranked][::-1]
    values = [v for _, v in ranked][::-1]

    plt.figure(figsize=(8, 5))
    plt.barh(range(len(tokens)), values)
    plt.yticks(range(len(tokens)), tokens)
    plt.xlabel("SHAP contribution to similarity")
    plt.title(f"{doc_id}: top token contributions")
    plt.tight_layout()
    outpath = f"shap_outputs/{doc_id}_bar.png"
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")

# Optional: export token-level attributions for text explanations
# Note: shap.save_html only supports force plots; text plots are notebook-only.
# We'll save per-token SHAP values to JSON for downstream viewing.
for doc_id, text, exp in EXPLANATIONS:
    try:
        e = exp[0]
        tokens = [str(t) for t in list(e.data)]
        # Ensure values are plain floats
        from numpy import ravel
        values = [float(ravel(v)[0]) if hasattr(v, "__array__") else float(v) for v in list(e.values)]
        payload = {"doc_id": doc_id, "tokens": tokens, "values": values}
        json_path = f"shap_outputs/{doc_id}_text.json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, ensure_ascii=False, indent=2)
        print(f"Saved: {json_path}")
    except Exception as ex:
        print(f"Skipping text JSON export for {doc_id}: {ex}")

# -------------------------------------------------------
# 6) Simple lexical baseline explanation for comparison
# -------------------------------------------------------
# Show which query terms overlap with each document.
# This is deliberately simple to make the interpretability contrast visible.
def tokenize(s):
    # Simple non-word split matching masker pattern; keep alphanumerics
    return [t.lower() for t in re.split(r"\W+", s) if t.strip()]

query_terms = set([t for t in tokenize(QUERY) if t not in ENGLISH_STOP_WORDS])
print("\nLexical baseline term overlaps:")
for r, i in enumerate(top_idx, start=1):
    doc_terms = set([t for t in tokenize(DOCS[i]) if t not in ENGLISH_STOP_WORDS])
    overlap = sorted(query_terms.intersection(doc_terms))
    print(f"{r:>2}. {DOC_IDS[i]} overlap: {overlap}")
