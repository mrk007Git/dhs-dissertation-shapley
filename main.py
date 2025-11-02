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
from infrastructure.article_repository import load_article_from_db
from tqdm import tqdm

# -----------------------
# 1) Inputs: query + docs
# -----------------------
QUERIES = {
    1: "Impact of Advanced Cardiac Life Support courses on knowledge retention and competency in undergraduate medical education in the UK",
    2: "Impact of first-pass defibrillation on survival and neurological outcome in out-of-hospital ventricular fibrillation.",
    3: "Effect of glucose administration or hyperglycemia management on outcomes after cardiac arrest."
}


def read_pmc_ids_for_query(qnum: int):
    path = os.path.join("data", f"query_{qnum}_pmc_ids.txt")
    ids = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                pmc = line.strip()
                if pmc:
                    ids.append(pmc)
    return ids

def fetch_docs_from_db(pmc_ids: list[str], limit_per_id: int = 1):
    docs = []
    doc_ids = []
    for pmc in tqdm(pmc_ids, desc="Fetching articles"):
        df = load_article_from_db(pmc, limit=limit_per_id)
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                content = str(row.get("content") or "").strip()
                if content:
                    docs.append(content)
                    # Prefer explicit PmcId if present, otherwise fall back to PMC
                    doc_ids.append(str(row.get("PmcId") or pmc))
    return docs, doc_ids

# -----------------------
# 2) Semantic model setup
# -----------------------
MODEL_NAME = "all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

def rank_and_print(query_text: str, docs: list[str], doc_ids: list[str], top_k: int = 3):
    if not docs:
        print("No documents to rank for this query.")
        return [], []
    query_emb = model.encode([query_text], normalize_embeddings=True)
    doc_embs = model.encode(docs, normalize_embeddings=True)
    scores = util.cos_sim(query_emb, doc_embs).cpu().numpy().flatten()
    ranked_idx = np.argsort(scores)[::-1]
    top_idx = ranked_idx[: min(top_k, len(docs))]
    print("Top results by semantic similarity:")
    for r, i in enumerate(top_idx, start=1):
        preview = docs[i][:90].replace("\n", " ")
        print(f"{r:>2}. {doc_ids[i]}  score={scores[i]:.3f}  {preview}...")
    return top_idx, scores

# ------------------------------------------
# 3) Wrap the scoring function for SHAP use
# ------------------------------------------
def make_predict_fn(query_emb_vec: np.ndarray):
    # Returns predict(texts) -> sims to the provided query embedding
    def predict(texts):
        embs = model.encode(texts, normalize_embeddings=True)
        sims = util.cos_sim(query_emb_vec, embs).cpu().numpy()
        return sims.reshape(-1, 1)
    return predict

# Text masker splits on non-word characters; change pattern if you want finer control
masker = ShapTextMasker(r"\W+")

# SHAP Explainer: model-agnostic text explainer around our predict() function
def explain_texts(predict_fn, texts: list[str]):
    explainer = shap.Explainer(predict_fn, masker)
    EXPLANATIONS = []
    for idx, text in enumerate(texts):
        exp = explainer([text], max_evals=500)  # type: ignore[arg-type]
        EXPLANATIONS.append((idx, text, exp))
    return EXPLANATIONS

# ------------------------------------
# 4) Explain the top-ranked documents
# ------------------------------------
# Keep explanations quick by limiting evaluations; increase if time allows
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -------------------------------
# 5) Visualise and save artefacts
# -------------------------------
# If running in Jupyter, shap.plots.text(EXPLANATIONS[0][2][0]) will render inline.
# Also create a bar plot of token contributions for each explained doc.

import matplotlib.pyplot as plt

ensure_dir("shap_outputs")

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
            # Skip stop words in outputs (case-insensitive)
            if t.lower() in ENGLISH_STOP_WORDS:
                continue
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
def save_token_csv(out_dir: str, explanations):
    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, "top_token_contributions.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "token", "shap_value"])
        for doc_id, text, exp in explanations:
            ranked = explanation_to_token_scores(exp, top_n=30)
            for tok, val in ranked:
                w.writerow([doc_id, tok, f"{val:.6f}"])
    print(f"\nSaved: {out_csv}")

def save_bar_plots(out_dir: str, explanations):
    ensure_dir(out_dir)
    for doc_id, text, exp in explanations:
        ranked = explanation_to_token_scores(exp, top_n=15)
        tokens = [t for t, _ in ranked][::-1]
        values = [v for _, v in ranked][::-1]

        plt.figure(figsize=(8, 5))
        plt.barh(range(len(tokens)), values)
        plt.yticks(range(len(tokens)), tokens)
        plt.xlabel("SHAP contribution to similarity")
        plt.title(f"{doc_id}: top token contributions")
        plt.tight_layout()
        outpath = os.path.join(out_dir, f"{doc_id}_bar.png")
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"Saved: {outpath}")

# Optional: export token-level attributions for text explanations
# Note: shap.save_html only supports force plots; text plots are notebook-only.
# We'll save per-token SHAP values to JSON for downstream viewing.
def save_text_jsons(out_dir: str, explanations):
    ensure_dir(out_dir)
    for doc_id, text, exp in explanations:
        try:
            e = exp[0]
            tokens = [str(t) for t in list(e.data)]
            from numpy import ravel
            values = [float(ravel(v)[0]) if hasattr(v, "__array__") else float(v) for v in list(e.values)]
            payload = {"doc_id": doc_id, "tokens": tokens, "values": values}
            json_path = os.path.join(out_dir, f"{doc_id}_text.json")
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

def remove_stopwords_text(s: str) -> str:
    tokens = tokenize(s)
    filtered = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return " ".join(filtered)

def process_query(qnum: int, qtext: str, top_k: int = 3, limit_per_id: int = 1):
    print(f"\n=== Processing Query {qnum}: {qtext[:80]}... ===")
    pmc_ids = read_pmc_ids_for_query(qnum)
    print(f"Loaded {len(pmc_ids)} PMC IDs from data/query_{qnum}_pmc_ids.txt")
    docs, doc_ids = fetch_docs_from_db(pmc_ids, limit_per_id=limit_per_id)
    if not docs:
        print("No documents fetched for this query; skipping SHAP.")
        return

    # Remove stop words from query and docs before ranking/SHAP
    cleaned_qtext = remove_stopwords_text(qtext)
    cleaned_docs = []
    cleaned_ids = []
    for did, d in zip(doc_ids, docs):
        cd = remove_stopwords_text(d)
        if cd:
            cleaned_docs.append(cd)
            cleaned_ids.append(did)
    if not cleaned_docs:
        print("All documents empty after stopword removal; skipping SHAP.")
        return

    top_idx, scores = rank_and_print(cleaned_qtext, cleaned_docs, cleaned_ids, top_k=top_k)
    if not len(top_idx):
        print("No top documents to explain.")
        return

    query_emb = model.encode([cleaned_qtext], normalize_embeddings=True)
    predict_fn = make_predict_fn(query_emb)
    # Prepare explanations for selected top docs
    selected_texts = [cleaned_docs[i] for i in top_idx]
    selected_ids = [cleaned_ids[i] for i in top_idx]
    exps = []
    for did, txt in zip(selected_ids, selected_texts):
        exp = shap.Explainer(predict_fn, masker)([txt], max_evals=500)  # type: ignore[arg-type]
        exps.append((did, txt, exp))

    out_dir = os.path.join("shap_outputs", f"query_{qnum}")
    save_token_csv(out_dir, exps)
    save_bar_plots(out_dir, exps)
    save_text_jsons(out_dir, exps)

    # Lexical overlap for context
    query_terms = set([t for t in tokenize(qtext) if t not in ENGLISH_STOP_WORDS])
    print("\nLexical baseline term overlaps:")
    for r, i in enumerate(top_idx, start=1):
        # Use original docs here for lexical context display
        original_doc = docs[i] if i < len(docs) else ""
        doc_terms = set([t for t in tokenize(original_doc) if t not in ENGLISH_STOP_WORDS])
        overlap = sorted(query_terms.intersection(doc_terms))
        print(f"{r:>2}. {selected_ids[r-1]} overlap: {overlap}")


if __name__ == "__main__":
    # Iterate every query
    for qnum, qtext in QUERIES.items():
        process_query(qnum, qtext, top_k=3, limit_per_id=1)
