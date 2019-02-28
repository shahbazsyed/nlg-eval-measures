import pickle
import numpy as np
import sys
import os
src_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(__file__)

enwiki_doc_file = os.path.join(src_dir,'../data/enwiki_vocab_min200.txt')


def get_enwiki_doc_frequencies():
    """Returns a dict containing document frequencies of words in English Wikipedia, with minimum frequency of 200
    """
    try:
        with open(enwiki_doc_file,encoding='utf-8') as file:
            frequencies={}
            for line in file:
                tokens = line.strip().split()
                frequencies[tokens[0]] = float(tokens[1])
        return frequencies

    except FileNotFoundError:
        return None

def get_enwiki_weighted_doc_frequencies(a=1e-3):
    """Returns weighted document frequencies according to the paramter
    
    Keyword Arguments:
        a {[float]} -- [weighing parameter] (default: {1e-3})
    """
    try:
        with open(enwiki_doc_file, encoding='utf-8') as file:
            word2weight={}
            if a <=0:
                a=1.0
            N=0
            for line in file:
                tokens = line.strip().split()
                word2weight[tokens[0]] = float(tokens[1])
                N+=float(tokens[1])
            for word, freq in word2weight.items():
                word2weight[word] = a/(a+freq/N)
            return word2weight 
    except FileNotFoundError:
        return None


def get_weighted_document_frequency(word_text):
    enwiki_weighted_doc_frequencies = get_enwiki_weighted_doc_frequencies()
    if word_text in enwiki_weighted_doc_frequencies:
        return enwiki_weighted_doc_frequencies[word_text]
    else:
        return 1.0