from __future__ import division
import collections
import numpy as np
import sys
sys.path.append(__file__)
from utils import embeddings
from utils import preprocessing
from rouge import Rouge
import six
from scipy.spatial import distance



def _find_closest(ngram, counter, index_name):
    if len(counter) == 0:
        return "", 0, 0
    #If no embedding can be found, try lexical matching
    if not(embeddings.has_ngram_embedding(ngram, index_name)):
        if ngram in counter:
            return ngram, counter[ngram], 1
        else:
            return "",0,0
    ranking_list = []
    ngram_emb = embeddings.get_ngram_embedding(ngram, index_name)
    for k,v in six.iteritems(counter):
        if k == ngram:
            ranking_list.append((k,v,1.))
            continue
        if not(embeddings.has_ngram_embedding(k,index_name)):
            ranking_list.append((k,v,0.))
            continue
        k_emb = embeddings.get_ngram_embedding(k,index_name)
        ranking_list.append((k,v,1 - distance.cosine(k_emb, ngram_emb)))
    ranked_list = sorted(ranking_list, key= lambda tup: tup[2], reverse=True)
    return ranked_list[0]

def _safe_divide(numerator, denominator):
    if denominator > 0:
        return numerator / denominator
    else:
        return 0

def _safe_f1(matches, recall_total, precision_total, alpha):
    recall_score = _safe_divide(matches, recall_total)
    precision_score = _safe_divide(matches, precision_total)
    denom = (1.0 - alpha) * precision_score +  alpha * recall_score
    if denom > 0.0:
        return (precision_score * recall_score) / denom
    else:
        return 0.0

def _soft_overlap(peer_counter, model_counter, index_name):
    SIMILARITY_THRESHOLD = 0.8
    result = 0
    for k,v in six.iteritems(peer_counter):
        closest, count, sim = _find_closest(k,model_counter,index_name)
        if sim < SIMILARITY_THRESHOLD:
            continue
        if count <= v:
            del model_counter[closest]
            result += count
        else:
            model_counter[closest] -= v
            result += v
    return result


def compute_rouge(peer, model):
    rouge = Rouge()
    scores = rouge.get_scores(peer, model)
    return scores

def compute_rouge_we(peer, model, n_grams, alpha, index_name):
    peer_tokens = preprocessing.get_words(peer)
    model_tokens = preprocessing.get_words(model)
    peer_counter = preprocessing.get_ngram_counts(peer,n_grams)
    model_counter = preprocessing.get_ngram_counts(model, n_grams)
    matches = _soft_overlap(peer_counter, model_counter, index_name)
    recall_total = preprocessing.get_ngram_count(model_tokens,n_grams)
    precision_total = preprocessing.get_ngram_count(peer_tokens,n_grams)
    return _safe_f1(matches, recall_total, precision_total, alpha)
