import sys
sys.path.append(__file__)
from utils import preprocessing
from nltk.metrics import jaccard_distance

def compute_jaccard_distance(peer, model):
    peer_tokens = set(preprocessing.get_words(peer))
    model_tokens = set(preprocessing.get_words(model))
    return jaccard_distance(peer_tokens, model_tokens)