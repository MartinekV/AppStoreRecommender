from nltk import WordNetLemmatizer, word_tokenize, SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.sbs = SnowballStemmer('english')

    def __call__(self, doc):
        doc = re.sub('[0-9*=･～+【】◆◇●•…■□・ー▼▲△▽►▻▶▷◁◈◉◎①○⋆＊★☆≈∞⇒※©~®™▬「」『』ㆍ（［］]+', '', doc)
        doc = re.sub('(?:[a-zA-Z])_+\s', ' ', doc)
        doc = re.sub('\s_+(?:[a-zA-Z])', ' ', doc)
        doc = re.sub('(?:[a-zA-Z])-+\s', ' ', doc)
        doc = re.sub('\s-+(?:[a-zA-Z])', ' ', doc)
        doc = re.sub('\s_+\s', ' ', doc)
        doc = re.sub('//(?:www)', ' ', doc)
        lemmas = [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        return [self.sbs.stem(l) for l in lemmas if len(l) > 2]


class TfIdfTransformer(object):
    @staticmethod
    def fit(corpus):
        print("processing {} documents...".format(len(corpus)))

        vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words=stopwords.words("english"))
        vectorizer.fit(corpus)
        with open("vectorizer.pickle", "wb") as f:
            pickle.dump(vectorizer, f)
        print(vectorizer.get_feature_names())
        print("processing done")

    @staticmethod
    def transform(apps):
        with open("vectorizer.pickle", "rb") as f:
            print("computing tf-idf table...")
            vectorizer = pickle.load(f)
            ids = list(apps.values_list("id", flat=True))
            descriptions = list(apps.values_list("app_desc", flat=True))
            tfidf_dict = {}

            for id, tfidf in zip(ids, vectorizer.transform(descriptions).toarray()):
                tfidf_dict[id] = tfidf

            with open("tfidf_table.pickle", "wb") as ft:
                pickle.dump(tfidf_dict, ft)
