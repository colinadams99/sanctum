# retrieval for chunking

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple
import sklearn
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix



def _clean(s: str) -> str:
    s = s.replace('\r', '\n')
    s = re.sub(r'[ \t]+', " ", s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    adding  character-based chunking with overlap
    """
    text = _clean(text)
    if not text:
        return []

    # initializes
    chunks = []
    start = 0

    # length of text
    n = len(text)

    # loops through and appends text for chunking
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

# term frequency index
@dataclass
class TfidfIndex:
    vectorizer: TfidfVectorizer
    matrix: csr_matrix
    chunks: List[str]


def build_tfidf_index(chunks: List[str]) -> TfidfIndex:
    vec = TfidfVectorizer(
        stop_words = 'english',
        max_features = 50000,
        ngram_range = (1, 2),
    )
    mat = vec.fit_transform(chunks)
    return TfidfIndex(vectorizer = vec, matrix = mat, chunks = chunks)

# retrieves top k terms from the tfidf index
def retrieve_top_k(index: TfidfIndex, query: str, k: int = 6) -> List[Tuple[int, float, str]]:
    qv = index.vectorizer.transform([query])
    sims = cosine_similarity(qv, index.matrix).flatten() # gets the cosine similarity
    top = sims.argsort()[::-1][:k] # sorts the terms

    return [(int(i), float(sims[i]), index.chunks[i]) for i in top]


