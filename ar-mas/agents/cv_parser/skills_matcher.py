from typing import List, Dict, Any, Optional
import os 
import re
import functools 

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import numpy as np
import pinecone

# Global cache for the SentenceTransformer model
_EMBEDDING_MODEL: Optional[SentenceTransformer] = None

# Global cache: (model_name, skill_tiple) -> embedding matrix
_SKILL_EMB_CACHE: Dict[Tuple[str, Tuple[str, ...]], np.ndarray] = {}

def init_embeddings_model(model_name: str ="all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Initialize (and cache) a SentenceTransformer model.
    
    model_name: Hugging Face / SentenceTransformer model id.
    Returns a SentenceTransformer instance.
    """
    global _EMBEDDING_MODEL

    # If already loaded with same name, resue
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(model_name)
    
    return _EMBEDDING_MODEL

def get_skill_embeddings(skill_vocabulary: List[str],
                         model: SentenceTransformer,
                         model_name: str) -> np.ndarray:
    """
    Given a list of skill names and a SentenceTransformer model,
    return a matrix of embeddings (one row per skill).
    Uses an in-memory cache so we only compute embeddings once
    per (model_name, exact skill list).
    """
    key = (model_name, tuple(skill_vocabulary))
    if key not in _SKILL_EMB_CACHE:
        emb = model.encode(skill_vocabulary, normalize_embeddings=True)
        _SKILL_EMB_CACHE[key] = emb
    return _SKILL_EMB_CACHE[key]

# Load spaCy model once
_NLP = spacy.load("en_core_web_sm")

# Helper: extract simple candidate phrases from text
def _extract_candidate_phrases(text: str, max_ngram: int = 3) -> List[str]:
    """
    Use SpaCy to extract candidate noun phrases from the text.
    These are potential skill candidates. (e.g. 'python', 'deep learning').
    """

    if not text:
        return []
    
    doc = _NLP(text)
    candidates = []

    # Noun chunks: "machine learning", "data science", etc.
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()

        # Basic clean: collapse spaces, remove trailing punctuation
        phrase = re.sub(r"\s+", " ", phrase)
        pharse = phrase.strip(" .,:;()[]{}")
        if len(phrase) >= 2:
            candidates.append(phrase)
        
    # Single tokens that are nouns or PROPN (proper nouns) - e.g. "Python"
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and token.is_alpha:
            t = token.text.strip()
            if len(t) >= 2:
                candidates.append(t)
    
    # Deduplicate, preserve original case
    unique = sorted(set(candidates), key=lambda s: s.lower())
    return unique

def match_skills(
    text: str,
    skill_vocabulary: List[str],
    top_k: int = 20,
    min_score: float = 0.6,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[Dict[str, Any]]:
    """
    Hyrid skills matcher:
    - Extract candidate noun-phrases from text (using spaCy).
    - Embed candidates & skill vocabulary with SentenceTransformer.
    - Compute cosine similarity.
    - For each candidate phrase, return top-k skills above threshold.

    Returns a list of objects:
    [
      {
        "candidate": "python",
        "matches": [
          {"skill": "Python", "score": 0.89},
          {"skill": "Python (Programming Language)", "score": 0.87}
        ]
      },
      ...
    ]
    """
    if not text or not skill_vocabulary:
        return []
    
    # Extract candidate phrased 
    candidates = _extract_candidate_phrases(text)
    if not candidates:
        return []
    
    # Loading embeddings model 
    model = init_embeddings_model(model_name)

    # Get/compute skill embeddings
    skill_emb = _get_skill_embeddings(skill_vocabulary, model, model_name)

    # Compute candidate embeddings
    cand_emb = model.encode(candidates, normalize_embeddings=True)

    # Consine similarity matrix [C, S]
    sims = cosine_similarity(cand_emb, skill_emb)

    results: List[Dict[str, Any]] = []

    # For each candidate, get its best skill matches
    for i, cand in enumberate(candidates):
        row = sims[i]   # similarities to all skills -> shape [S]

        # indices of top_k highest scores
        top_idx = np.argsort(row)[::-1][:top_k]

        matches = []
        for j in top_idx:
            score = float(row[j])
            if score < min_score:
                continue 
            
            matches.append({
                "skill": skill_vocabulary[j],
                "score": round(score, 4)
            })
        
        if matches:
            results.append({
                "candidates": cand,
                "matches": matches
            })
    
    return results
