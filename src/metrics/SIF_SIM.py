import numpy as np
import sys
sys.path.append(__file__)
from utils import embeddings
import math
from operator import mul


def compute_sif_similarity(content, summary, index_name="emb_fasttext"):
    c_emb = embeddings.get_sif_embedding(content, index_name)
    s_emb = embeddings.get_sif_embedding(summary, index_name)
    inner = (c_emb*s_emb).sum()
    c_emb_norm = np.sqrt((c_emb*c_emb).sum())
    s_emb_norm = np.sqrt((s_emb * s_emb).sum())
    sim_score = inner / c_emb_norm / s_emb_norm
    return sim_score

def _dot_product(v1, v2):
    return sum(map(mul, v1, v2))

def _vector_product(v1, v2):
    prod = _dot_product(v1, v2)
    len1 = math.sqrt(_dot_product(v1, v1))
    len2 = math.sqrt(_dot_product(v2, v2))
    return prod / (len1 * len2)

def compute_cosine_similarity(content, summary, index_name="emb_fasttext"):
    c_emb = embeddings.get_document_embedding(content)
    s_emb = embeddings.get_document_embedding(summary)
    return _vector_product(c_emb,s_emb)

