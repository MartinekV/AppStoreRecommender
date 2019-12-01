from nltk import WordNetLemmatizer, word_tokenize, SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.sbs = SnowballStemmer('english')

    def __call__(self, doc):
        doc = re.sub('[0-9]+', '', doc)
        lemmas = [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        return [self.sbs.stem(l) for l in lemmas if len(l) > 2]


class TfIdfTransformer(object):
    @staticmethod
    def fit(corpus):
        print("processing {} documents...".format(len(corpus)))

        vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words="english")
        vectorizer.fit(corpus)
        pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))
        print("processing done")
