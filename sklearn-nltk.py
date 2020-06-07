# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 10:54:50 2020

@author: Machachane
"""


from sklearn.datasets import fetch_20newsgroups
categories_train = fetch_20newsgroups(subset='train', shuffle=True)


cat_names = categories_train.target_names
print('\n\n',categories_train.target_names)     #prints all the categories
print('\n',categories_train.data[0])          #prints the whole first article (data set)
print('\n'.join(categories_train.data[0].split('\n')[:3])) #prints first 3 lines of the first data file

#-----------------------------------------------------------------------

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(categories_train.data)
X_train_counts.shape

#-----------------------------------------------------------------------

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

#-----------------------------------------------------------------------

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, categories_train.target)

#-----------------------------------------------------------------------

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])

text_clf = text_clf.fit(categories_train.data, categories_train.target)

#-----------------------------------------------------------------------

import numpy as np
categories_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict(categories_test.data)
np.mean(predicted == categories_test.target)

#-----------------------------------------------------------------------

from sklearn.linear_model import SGDClassifier

text_clf_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge'))]) #, penalty='12' alpha=1e-3, n_iter=5, random_state=42

_= text_clf_svm.fit(categories_train.data, categories_train.target)

predicted_svm = text_clf_svm.predict(categories_test.data)
np.mean(predicted_svm == categories_test.target)

#-----------------------------------------------------------------------
"""
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1,1), (1,2)],
              'tfidf__ise_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3)}


gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(categories_train.data, categories_train.target)
gs_clf.best_score_
gs_clf.best_params_
"""
#-----------------------------------------------------------------------

from sklearn.model_selection import GridSearchCV
parameters_svm = {'vect__ngram_range': [(1,1), (1,2)],
                  'tfidf__use_idf': (True, False),
                  'clf-svm__alpha': (1e-2, 1e-3)}


gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(categories_train.data, categories_train.target)
gs_clf_svm.best_score_
gs_clf_svm.best_params_

#-----------------------------------------------------------------------

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), 
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB())])

#-----------------------------------------------------------------------

import nltk
nltk.download()

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english', ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer(), self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
    
stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect),
                             ('tfidf', TfidfTransformer()),
                             ('mnb', MultinomialNB(fit_prior=False))])

text_mnb_stemmed = text_mnb_stemmed.fit(categories_train.data, categories_train.target)

predicted_mnb_stemmed = text_mnb_stemmed.predict(categories_test.data)

np.mean(predicted_mnb_stemmed == categories_test.target)


"""
https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
"""