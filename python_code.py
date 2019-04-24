import warnings
warnings.filterwarnings('ignore')

###DELETE ABOVE LATER!!

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
#from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
#from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

# ------ functions ----- #

class AppTextClassifier():

    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.snowball = SnowballStemmer('english')
        self.eng_stopwords = set(stopwords.words("english"))


    def _prepare_words(self, df, text_col_name):
        #remove stop words from original
        df['text_clean'] = [[word for word in original_text.split(" ") if word not in self.eng_stopwords] for original_text in df[text_col_name]]

        #stem words
        df['text_clean'] = [[self.snowball.stem(word) for word in words_list] for words_list in df['text_clean']]

        #join all words again to prepare for tf-idf
        df['text_clean'] = [" ".join(words_list) for words_list in df['text_clean']]

        #create sparse matrix, make available for entire class
        return df

    def fit(self, df, text_col_name):
        df = self._prepare_words(df, text_col_name)
        tfidf_train = self.tfidf.fit_transform(df['text_clean'])
        return tfidf_train

    def transform(self, df, text_col_name):
        df = self._prepare_words(df, text_col_name)
        tfidf_transform = self.tfidf.transform(df['text_clean'])

        return tfidf_transform



if __name__  == "__main__":

    #load data
    X = pd.read_csv("../data/train.txt", header = None, names = ["original_text"])
    y = pd.read_csv("../data/labels.txt", header = None)

    #create training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, test_size = .3, random_state = 14)

    #estimate final roc_auc score for multinomial model
    atc = AppTextClassifier()
    X_train_tfidf = atc.fit(X_train, "original_text")
    model = MultinomialNB()
    estimated_roc_auc = cross_val_score(model, X_train_tfidf, y_train, cv = 3, scoring='roc_auc')
    print("Average Estimated ROC AUC score based on cross validation:  ", estimated_roc_auc.mean() )

    #final ROC AUC, assuming we have the final model we're going to use
    model.fit(X_train_tfidf, y_train)

    X_test_tfidf = atc.transform(X_test, "original_text")
    y_test_pred = model.predict_proba(X_test_tfidf)
    print("Final ROC AUC score of test set: ", roc_auc_score(y_test, y_test_pred[:,1]))
