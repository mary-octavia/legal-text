import codecs
import string
import unicodedata
import numpy as np
import gensim
import sklearn.feature_selection as fs
from gensim.models import doc2vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer
from sklearn.pipeline import Pipeline
from sklearn import naive_bayes
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.cross_validation import KFold, LeaveOneOut, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.dummy import DummyClassifier
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

fclean="court_ruling_complete_113k-utf8-tab-clean3-100fold-clean.csv"
court_ruling = "court_rulings_task2_8classes_train.csv"


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii

def get_preprocessor(suffix=''):
    def preprocess(unicode_text):
        return unicode_text.strip().lower() + suffix
    return preprocess

class MyDocs(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in codecs.open(self.fname, 'r', encoding="utf-8"):
            aux = line.split("\t")
            yield aux[4].split(" ")

def preprocess_data(X, n, suffix='', binarize=True):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 1),
                                 preprocessor=get_preprocessor(suffix))
    X = vectorizer.fit_transform(X)
    X = Binarizer(copy=False).fit_transform(X) if binarize else X
    return X

def train_d2v(fname):
    print "entered d2v training"
    sentences = doc2vec.TaggedLineDocument(fname)
    model_court = gensim.models.Doc2Vec(sentences, size=200, workers =10, window=20)


    model_court.save("court-task2-8_train.d2v")
    # pkl = open("court-task2-8_train.d2v", 'wb')
    # pickle.dump(model_court, pkl)
    # pkl.close()



if __name__ == '__main__':
    # train_d2v(court_ruling)

    model = doc2vec.Doc2Vec.load("court-task1-8_test.d2v2")
    model2 = doc2vec.Doc2Vec.load("court-task2-8_test.d2v2")
    
    