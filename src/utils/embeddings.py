from __future__ import absolute_import
import sys
import os
sys.path.append(__file__)
from elasticsearch import Elasticsearch
from utils import frequencies
from utils import preprocessing
import numpy as np
import json
import itertools


global es
es = Elasticsearch(['141.54.132.147'])

def _get_query_body(query, retrieve_content=True):
    """Create query body for elasticsearch
    
    Arguments:
        query {[string]} -- term to search for
        retrieve_content {[bool]} -- if the complete source must be returned (default: {True})
    """
    query_body = {
        "query": {
            "query_string": {
                "query": query,
                "fields": ["word"]
            }
        },
        "_source":retrieve_content
    }
    return query_body


def has_embedding(word, index_name="emb_fasttext"):
    """Query elastic search index to check for docstring
    
    Arguments:
        word {[string]} -- term to search for
        index_name {[string]} -- index to query
    """

    query_body = _get_query_body(word,retrieve_content=False)
    response = es.search(index=index_name,doc_type='document',body=query_body)
    hits = response['hits']['hits']
    if len(hits)>0:
        return True
    else:
        return False

def has_ngram_embedding(ngram, index_name="emb_fasttext"):
    has = False
    for word in ngram:
        has = has_embedding(word,index_name)
    return has
    

def get_word_embedding(word, index_name="emb_fasttext"):
    """Get word embedding from index
    
    Arguments:
        word {[string]} -- term to get embedding for
        index_name {[string]} -- index to query
    """
    query_body = _get_query_body(word,retrieve_content=True)
    if has_embedding(word, index_name):
        response = es.search(index=index_name, doc_type='document', body=query_body)
        doc = response['hits']['hits'][0]
        _res = list(doc.values())[4]
        return _res['vector']
    else:
        return np.zeros(shape=300)

def get_ngram_embedding(ngram, index_name='emb_fasttext'):
    words = list(ngram)
    vec = []
    for word in words:
        vec.append(get_word_embedding(word, index_name))
    ngram_vec = np.mean(vec, axis=0)
    return ngram_vec



def get_document_embedding(document, index_name="emb_fasttext"):
    """Get embedding for a document. Returns the average of individual word embeddings.
    
    Arguments:
        document {[string]} -- document text
        index_name {[string]} -- index to query
    
    """
    vec=[]
    normalized_document = preprocessing.normalize_text(document)
    for token in normalized_document.split():
        vec.append(get_word_embedding(token,index_name))
    doc_vec = np.mean(vec, axis=0)
    return doc_vec


def get_sif_embedding(text, index_name="emb_fasttext"):
    normalized_text = preprocessing.normalize_text(text)
    words = normalized_text.split()
    text_length = len(words)
    sif_vec = np.zeros(shape=300)
    for word in words:
        word_weight = frequencies.get_weighted_document_frequency(word)
        word_vector = get_word_embedding(word, index_name)
        sif_vec = np.add(sif_vec, np.multiply(word_weight, word_vector))
    sif_vec = np.divide(sif_vec, text_length)
    return sif_vec





