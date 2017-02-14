import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from numpy import array
from sklearn.svm import SVC


class AnalyzeReview(object):
    """
    Analyze Review from tsv file class.
    """
    def __init__(self):
        """
        Init Method
        """
        self.train_x = []
        self.train_y = []
        self.model = None
        self.vectorizer = None

    def get_cleaned_review(self, raw_review):
        """

        :param raw_review:
        :return:
        """
        # Remove formatting elements
        clean_review = BeautifulSoup(raw_review, "html5lib").get_text()
        # Remove everything except alphabets
        clean_review = re.sub("[^a-zA-Z]", " ", clean_review)
        clean_review = clean_review.lower().split()
        # Pick out stop words from NLTK corpus
        stops = set(stopwords.words("english"))
        # Remove all the stop words
        clean_words = [w for w in clean_review if w not in stops]
        return " ".join(clean_words)

    def read_file(self, filename):
        """

        :param filename: Pass the tab separated file as input
        :return: Returns a Pandas Dataframe with all the fields
        """
        train = pd.read_csv(filename, quoting=3, header=0, delimiter="\t", index_col=0)
        train_x = train["review"].values
        self.train_y = train["sentiment"].values
        l = len(self.train_y)
        for i, sample in enumerate(train_x):
            if i % 1000 == 0:
                print " Cleaning words {0} / {1} ".format(i, l)

            clean_sample = self.get_cleaned_review(sample)
            self.train_x.append(clean_sample)

    def vectorize_words(self):
        """
        Converts words into vectors using Bag of Words
        :return:
        """
        vectorizer = CountVectorizer(analyzer='word', tokenizer=None,
                                     preprocessor=None, stop_words=None,
                                     max_features=5000)
        self.train_x = vectorizer.fit_transform(self.train_x).toarray()
        self.vectorizer = vectorizer
        print self.train_x, self.train_y

    def train_svm(self):
        """

        :return: Saves trained SVM model
        """
        model = SVC()
        model.fit(self.train_x, self.train_y)
        self.model = model

    def predict(self, reviews):
        """

        :param reviews: Enter movie review
        :return: Predicts sentiment
        """
        reviews = self.get_cleaned_review(reviews)
        vectorizer = self.vectorizer
        vector = vectorizer.transform(reviews).toarray()
        model = self.model
        sentiment = model.predict(vector)
        return sentiment


def main():
    """

    :return: Run the program
    """
    model = AnalyzeReview()
    model.read_file('labeledTrainData.tsv')
    model.vectorize_words()
    model.train_svm()
    print "The sentiment of the movie review is".format(model.predict("The movie was really bad, I did not like it one bit"))

if __name__ == "__main__":
    main()
