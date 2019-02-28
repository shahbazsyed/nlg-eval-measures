# import sys
# sys.path.append(__file__)
# from metrics import BLEU, JSD, ROUGE_WE, METEOR, JACCARD
# import pickle
# from tqdm import tqdm



# def compute_scores(peer,model, index_name="emb_fasttext"):
#     all_scores = {}
#     sif_score = SIF_SIM.compute_sif_similarity(peer,model,index_name)
#     bleu_scores = BLEU.compute_blue_score(peer, model)
#     js_divergence = JSD.compute_jensen_shannon_divergence(peer, model,1)
#     rouge_score = ROUGE_WE.compute_rouge(peer, model)
#     meteor_score = METEOR.compute_meteor_score(peer, model)
#     all_scores['SIF'] = sif_score
#     all_scores['BLEU'] = bleu_scores
#     all_scores['JSD'] = js_divergence
#     all_scores['ROUGE'] = rouge_score
#     all_scores['METEOR'] = meteor_score
#     return all_scores
# if __name__ == "__main__":
#     #from timeit import Timer
#     content = '''I, m24, have been dating a wonderful girl 26 for 8 months now.  I am very much in love with her.  However today when eating lunch in my girlfriends apartment I went on youtube and noticed a wierd link to guys pissing their pants.  Somthing didn't ring right to me so I pulled up her history, I know horrible, and found a forum with stories about guys pissing their pants.  I quickly sum up that this is her porn.  Somthing called a desperation fetish.  I have no idea how to handle this.  I have many concerns.  If I can't satisfy her needs how long before its more than just internet browsing.  I don't understand fetishes at all not having any.  I don't like that she hid it from me, it makes me think there could be more'''
#     summary = "I invaded my gfs  privacy and discovered a fetish that I find mildly disturbing what the hell should I do?"
#     model = "I looked into my gfs private stuff and discovered that she has a disturbing fetish."
#     model_1 = "This is some random text not related to the content at all."
#     print(METEOR.compute_meteor_score(model, model_1))
#     #t = Timer(lambda : compute_scores(summary, model,"emb_social"))
#     # print(t.timeit(number=1))
   
from nlgeval.metrics import JSD

if __name__== "__main__":
    model = "I looked into my gfs private stuff and discovered that she has a disturbing fetish."
    model_1 = "This is some random text not related to the content at all."
    print(JSD.compute_jensen_shannon_divergence(model, model_1))