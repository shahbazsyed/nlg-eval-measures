from nlgeval.utils import preprocessing
import math
import numpy as np

def _KL_divergence(peer_distribution, model_distribution):
    sum_val = 0
    for w,f in peer_distribution.items():
        if w in model_distribution:
            sum_val += f*math.log(f/float(model_distribution[w]))
    if np.isnan(sum_val):
        return 0.0
    return sum_val

def _compute_average_of_two_distributions(dist_1, dist_2):
    average = {}
    keys = set(dist_1.keys()) | set(dist_2.keys())

    for k in keys:
        s_1 = dist_1.get(k,0)
        s_2 = dist_2.get(k,0)
        average[k] = (s_1 + s_2) / 2.
    return average

def _JS_divergence(peer_distribution, model_distribution):
    average = _compute_average_of_two_distributions(peer_distribution, model_distribution)
    js = (_KL_divergence(peer_distribution, average) + _KL_divergence(model_distribution, average)) / 2.

    if np.isnan(js):
        return 0.
    return js


class JsdScorer():

    def compute_score(self, peer, model, as_ngrams=2):
        peer_distribution = preprocessing.compute_term_frequency_in_text(peer,as_ngrams)
        model_distribution = preprocessing.compute_term_frequency_in_text(model,as_ngrams)
        avg = 0.
        avg = _JS_divergence(peer_distribution, model_distribution)
        return avg
    
    def method(self):
        return "JSD"
