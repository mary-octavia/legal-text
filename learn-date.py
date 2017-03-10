import codecs
import string
import unicodedata
import numpy as np
import sklearn.feature_selection as fs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer
from sklearn.pipeline import Pipeline
from sklearn import naive_bayes
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.cross_validation import KFold, LeaveOneOut, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.dummy import DummyClassifier
from datetime import datetime

print "entered learn-date.py at ", datetime.now()

fl = "juri-all-non-empty-200fold.csv"

def get_preprocessor(suffix=''):
    def preprocess(unicode_text):
        return unicode_text.strip().lower() + suffix
    return preprocess


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii


def preprocess_data(X, n, suffix='', binarize=True):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 1),
                                 preprocessor=get_preprocessor(suffix))
    X = vectorizer.fit_transform(X)
    X = Binarizer(copy=False).fit_transform(X) if binarize else X
    return X


def remove_duplicate_entries(fname):
    f = codecs.open(fname, "r", encoding="utf-8")
    dsc, new_c = [], []

    ct = f.read()
    f.close()
    ct = ct.split("\n")

    for i in range(len(ct)):
        ct[i] = ct[i].split("\t")
        dsc.append(ct[i][4])

    if len(dsc) != len(ct):
        print "err in unpacking ct"
        #return 0
    else:
        idx = np.zeros(len(dsc))
        for i in range(len(dsc)):
            if dsc[i] in dsc[i+1:len(dsc)]:
                idx[i] = 1
        for i in range(len(ct)):
            if idx[i] == 0:
                new_c.append("\t".join(ct[i]))
    return new_c


def load_data(filename=fl):
    legal, y = [], []
    ct = remove_duplicate_entries(filename)
    print "len(ct) in load_data: ", len(ct)
 
    #with codecs.open(filename, 'r', encoding="utf-8") as f:
    for line in ct:
        a = line.split("\t")
        if (line != "") and (len(a)==7):
            idx, loc, dec, date, dsc, art, law  = line.split("\t")
            legal.append(remove_accents(dsc))
            y.append(int(date.split("-")[0])/10) # get the century and decade only
    print len(y), len(legal)
    legal, y = np.array(legal), np.array(y)

    # for i in range(len(y)): #replace year with decade
    #     aux = y[i]
    #     aux = aux[0:len(aux)-1] + "0"
    #     y[i] = aux
        
    set_y = list(set(y))
    print "unique labels: ", set_y 

    for i in range(len(legal)):
        legal[i] = legal[i].lower()
        legal[i] = ((legal[i].encode("utf8")).translate(None, string.punctuation)).decode("utf8")       
        legal[i] = ((legal[i].encode("utf8")).translate(None, "1234567890")).decode("utf8")

    return legal, y


def get_best_features(X, y, vectorizer):
    '''get names of best features in X from vectorizer'''
    print "entered get_best_features"
    f = codecs.open("results.txt", "a", encoding="utf-8")
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


if __name__ == '__main__':
    start_time = str(datetime.now())
    filename = 'juri-all-non-empty.csv'
    f = codecs.open("results.txt", "a", encoding="utf-8")
    f.write("run started at: " + start_time + "\n input file: " + filename)
    X, y = load_data(filename)

    '''get best unigram features with ANOVA'''
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1))
    X_new = vectorizer.fit_transform(X)
    get_best_features(X_new, y, vectorizer)

    # extract_lexical_features(X)


    '''cross-validation block'''
    skf = StratifiedKFold(y, n_folds=10)
    # X_new = preprocess_data(X, n=4, suffix="", binarize=False)
    clf = LinearSVC()
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
