from rouge import Rouge


class RougeScorer():
    def compute_score(self, peer, model):
        rouge = Rouge()
        scores = rouge.get_scores(peer, model)
        return scores

    def method(self):
        return "ROUGE"
