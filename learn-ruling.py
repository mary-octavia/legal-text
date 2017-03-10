import codecs
import string
import unicodedata
import numpy as np
import sklearn.feature_selection as fs
from nltk import word_tokenize 
from nltk.stem import SnowballStemmer    
from gensim.models import doc2vec    
from sklearn import naive_bayes
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.cross_validation import KFold, LeaveOneOut, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.neighbors import KNeighborsClassifier


rl = "juri-all-non-empty.csv"
# model = doc2vec.Doc2Vec.load("court-task1-8_test.d2v2")
model2_tr = doc2vec.Doc2Vec.load("court-task2-8_train.d2v2")
# model2_ts = doc2vec.Doc2Vec.load("court-task2-8_test.d2v2")

class LemmaTokenizer(object):
    def __init__(self):
        self.stemmer = SnowballStemmer("french")

    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in word_tokenize(doc)]


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii


def get_preprocessor(suffix=''):
    def preprocess(unicode_text):
        return unicode_text.strip().lower() + suffix
    return preprocess


def preprocess_data(X, n, suffix='', binarize=True):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1),
                                 preprocessor=get_preprocessor(suffix), tokenizer=LemmaTokenizer())
    X = vectorizer.fit_transform(X)
    X = Binarizer(copy=False).fit_transform(X) if binarize else X
    return X


def reduce_classes(y):
    for i in range(len(y)):
        y[i] = ((y[i].encode("utf8")).translate(None, string.punctuation)).decode("utf8")
        y[i] = y[i].replace("nonlieu", "non-lieu")
        y[i] = y[i].replace("non lieu", "non-lieu")
        y[i] = y[i].split(" ")[0]
    return y

def load_data2(fin):
    legal, y = [], []
    with codecs.open(fin, 'r', encoding="utf-8") as f:
        ct = f.read()
    ct = ct.split("\n")
    print "len ct", len(ct)
    for i in range(len(ct)):
        aux = ct[i].split("\t")
        if len(aux) == 2:
            dsc, dec = aux[0], aux[1]
            legal.append(remove_accents(dsc))
            y.append(remove_accents(dec))
        else:
            print i
    print len(y), len(legal)
    # legal = np.array(legal)
    # y = np.array(y)
    set_y = list(set(y))

    for i in range(len(legal)):
        legal[i] = legal[i].lower()
        legal[i] = ((legal[i].encode("utf8")).translate(None, string.punctuation)).decode("utf8")
        legal[i] = ((legal[i].encode("utf8")).translate(None, "1234567890")).decode("utf8")
        legal[i] = legal[i].replace("annule", "")
        legal[i] = legal[i].replace("casse", "")
        legal[i] = legal[i].replace("rejeter", "")
        legal[i] = legal[i].replace("rejette", "")
        legal[i] = legal[i].replace("irrecevable", "")
        legal[i] = legal[i].replace("irrecevabilite", "")
        legal[i] = legal[i].replace("recevable", "")
        legal[i] = legal[i].replace("recevabilite", "")            
    for j in range(len(set_y)):
            legal[i] = legal[i].replace(set_y[j], "")
    # y = reduce_classes(set_y)
    print "reduced classes", len(set(set_y))
    return legal, y

def load_data(filename=rl):
    legal, y = [], []
    with codecs.open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            a = line.split("\t")
            if (line != "") and (len(a)==7):
                idx, loc, dec, date, dsc, art, law  = line.split("\t")
                legal.append(remove_accents(dsc))
                y.append(remove_accents(dec))
    print len(y), len(legal)
    legal = np.array(legal)
    y = np.array(y)
    set_y = list(set(y))

    for i in range(len(legal)):
        legal[i] = legal[i].lower()
        legal[i] = ((legal[i].encode("utf8")).translate(None, string.punctuation)).decode("utf8")
        legal[i] = ((legal[i].encode("utf8")).translate(None, "1234567890")).decode("utf8")
        legal[i] = legal[i].replace("annule", "")
        legal[i] = legal[i].replace("casse", "")
        legal[i] = legal[i].replace("rejeter", "")
        legal[i] = legal[i].replace("rejette", "")
        legal[i] = legal[i].replace("irrecevable", "")
        legal[i] = legal[i].replace("irrecevabilite", "")
        legal[i] = legal[i].replace("recevable", "")
        legal[i] = legal[i].replace("recevabilite", "")            
	for j in range(len(set_y)):
            legal[i] = legal[i].replace(set_y[j], "")
    y = reduce_classes(y)
    print "reduced classes", len(y), set(y)
    return legal, y


def get_best_features(X, y, vectorizer):
    '''get names of best features in X from vectorizer'''
    print "entered get_best_features"
    f = codecs.open("results.txt", "w", encoding="utf-8")
    fnames = vectorizer.get_feature_names()
    b = fs.SelectKBest(fs.f_classif, k=50) #k is number of features.
    X_n = b.fit_transform(X, y)
    index_v =  b.get_support()

    print "best unigrams:"
    for i in range(len(index_v)):
        if index_v[i] == True:
            f.write(fnames[i])
            f.write("\n")
    f.close()
    print "exited get_best_features"


class get_docvec(BaseEstimator, TransformerMixin):   
        
    def fit(self):
        return self
    
    def transform(self):
        # X_ft = create_occ_matrix(X, stwords)

        # elif self.dist == ''
        # X_dst = np.array(X_dst)
        # print "X_rn", X_dst.shape
        new_X = []
        for i in range(len(model2_tr.docvecs)):
            new_X.append(model2_tr.docvecs[i])
        new_X = np.array(new_X)
        print "new_X type", type(new_X)
        return new_X


def get_docvecs_np(vecs):
    np_vecs = np.zeros((len(vecs), len(vecs[0])), dtype=vecs[0].dtype)
    for i in range(len(vecs)):
        np_vecs[i] = vecs[i]
    print "types", type(np_vecs)
    return np_vecs


if __name__ == '__main__':
    filename = 'court_rulings_task2_8classes_train.csv'
    f = codecs.open("results.txt", "a", encoding="utf-8")
    X, y = load_data2(filename)

    '''get best unigram features with ANOVA'''
    # vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), tokenizer=LemmaTokenizer())
    # X_new = vectorizer.fit_transform(X)
    # get_best_features(X_new, y, vectorizer)

    # model = doc2vec.Doc2Vec.load("court-task1-8_test.d2v2")
    # model2 = doc2vec.Doc2Vec.load("court-task2-8_test.d2v2")

    # extract_lexical_features(X)


    '''cross-validation block'''
    skf = StratifiedKFold(y, n_folds=10)
    # X_new = preprocess_data(X, n=4, suffix="", binarize=False)
    # X_new = get_docvec.transform(X)
    X_new = get_docvecs_np(model2_tr.docvecs)
    # clf = LinearSVC()
    clf = KNeighborsClassifier(n_neighbors=3)
    dummy = DummyClassifier(strategy="stratified")

    accuracy, recall, precision, f1 = [], [], [], []
    dummy_acc, dummy_rec, dummy_prec, dummy_f1 = [], [], [], []
    for train_index, test_index in skf:
        X_train, X_test = X_new[train_index], X_new[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print "fitting the classifier"
        clf.fit(X_train, y_train)
        dummy.fit(X_train, y_train)

        print "predicting"
        y_pred = clf.predict(X_test)
        y_dummy = dummy.predict(X_test)

        print "svm report:\n", classification_report(y_test, y_pred)
        print "dummy report: \n", classification_report(y_test, y_dummy)
        f.write(classification_report(y_test, y_pred))
        f.write("\n")

        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred, average='weighted'))
        recall.append(recall_score(y_test, y_pred, average='weighted'))
        f1.append(f1_score(y_test, y_pred, average='weighted'))

        dummy_acc.append(accuracy_score(y_test, y_dummy))
        dummy_prec.append(precision_score(y_test, y_dummy, average='weighted'))
        dummy_rec.append(recall_score(y_test, y_dummy, average='weighted'))
        dummy_f1.append(f1_score(y_test, y_dummy, average='weighted'))

    print "accuracy mean ", np.mean(accuracy), " accuracy std ", np.std(accuracy)
    print "precision mean ", np.mean(precision), " and std ", np.std(precision)
    print "recall mean ", np.mean(recall), " and std ", np.std(recall)
    print "f1 mean ", np.mean(f1), " and std ", np.std(f1)
    print "dummy accuracy mean ", np.mean(dummy_acc), " accuracy std ", np.std(dummy_acc)
    print "dummy precision mean ", np.mean(dummy_rec), " and std ", np.std(dummy_prec)
    print "dummy recall mean ", np.mean(dummy_rec), " and std ", np.std(dummy_rec)
    print "dummy f1 mean ", np.mean(dummy_f1), " and std ", np.std(dummy_f1)

    f.write("accuracy mean:  " + np.mean(accuracy).astype(str) +"accuracy std: "+ np.std(accuracy).astype(str) + "\n")
    f.write("precision mean: " + np.mean(precision).astype(str) + "precision std: " + np.std(precision).astype(str) + "\n")
    f.write("recall mean:" + np.mean(recall).astype(str) + "recall std: "+ np.std(recall).astype(str) + "\n")
    f.write("f1 mean:" +np.mean(f1).astype(str) + "f1 std: " + np.std(f1).astype(str)) 


    f.close()
