from rouge import Rouge

def compute_rouge_score(peer, model):
    rouge = Rouge()
    scores = rouge.get_scores(peer, model)
    return scores

