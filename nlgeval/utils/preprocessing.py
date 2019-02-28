from __future__ import division
import re
import string
from collections import Counter
import nltk
from nltk.util import ngrams
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

global stemmer, stopset, regex
regex = re.compile('[^a-zA-z ]')
stemmer = SnowballStemmer("english")
stopset = frozenset(stopwords.words("english"))

def convert_to_unicode(object):
    if isinstance(object, str):
        return object
    elif isinstance(object,bytes):
        return object.decode("utf-8")


def get_stem(word):
    return stemmer.stem(word)

def _remove_punctuation(text):
    return re.sub('['+string.punctuation+']', '', text)

def normalize_text(text):
    no_punctuations = _remove_punctuation(text)
    normalized_text = regex.sub('',no_punctuations)
    return convert_to_unicode(normalized_text).lower()

def get_ngrams(text, N=2):
    normalized_text = normalize_text(text)
    tokens = word_tokenize(normalized_text)
    return [gram for gram in ngrams(tokens, N)]

def get_words(text, stem=False):
    normalized_text = normalize_text(text)
    tokens = word_tokenize(normalized_text)
    if stem:
        return [get_stem(token) for token in tokens]
    else:
        return tokens

def get_ngram_counts(text, N=2):
    return Counter(get_ngrams(text))

def get_ngram_count(words, N):
    return max(len(words)-N+1, 0)

def get_total_ngram_count(text, N=2):
    return max(len(get_words(text))-N+1, 0)

def is_ngram_content(ngram):
    words = list(ngram)
    for word in words:
        if not(word in stopset):
            return True
    return False

def _compute_word_frequency(words):
    word_freq = {}
    for word in list(words):
        word_freq[word] = word_freq.get(word,0)+1
    return word_freq

def compute_term_frequency_in_text(text, as_ngrams=2):
    words = get_content_words(text,as_ngrams)
    num_words = len(words)
    word_freq = _compute_word_frequency(words)
    word_tf = dict((w, f/float(num_words)) for w, f in word_freq.items())
    return word_tf

def get_content_words(text, as_ngrams=2):
    words = get_words(text)
    if as_ngrams >1:
        return [gram for gram in ngrams(words,as_ngrams) if is_ngram_content(gram)]
    else:
        return [word for word in words if not word in stopset]