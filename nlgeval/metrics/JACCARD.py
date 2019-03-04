from nltk.metrics import jaccard_distance
from nlgeval.utils import preprocessing
import sys
sys.path.append(__file__)


class JaccardScorer():
    def compute_score(self, peer, model):
        peer_tokens = set(preprocessing.get_words(peer))
        model_tokens = set(preprocessing.get_words(model))
        return jaccard_distance(peer_tokens, model_tokens)

    def method(self):
        return "JACCARD"
