import sys
sys.path.append(__file__)
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import bleu_score
from utils import preprocessing
import json

def compute_blue_score(peer, model):
    peer_tokens = preprocessing.get_words(peer)
    model_tokens = preprocessing.get_words(model)
    smooth = bleu_score.SmoothingFunction().method3
    scores = {}
    scores["cumulative_1"] = sentence_bleu([peer_tokens],model_tokens,weights=(1,0,0,0), smoothing_function=smooth)
    scores["cumulative_2"] = sentence_bleu([peer_tokens],model_tokens,weights=(0.5,0.5,0,0), smoothing_function=smooth)
    scores["cumulative_3"] = sentence_bleu([peer_tokens],model_tokens,weights=(0.33,0.33,0.33,0), smoothing_function=smooth)
    scores["cumulative_4"] = sentence_bleu([peer_tokens],model_tokens,weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)
    return json.dumps(scores)
