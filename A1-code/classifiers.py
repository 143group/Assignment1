import numpy as np
from math import log
# You need to build your own model here instead of using well-built python packages such as sklearn

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# You can use the models form sklearn packages to check the performance of your own models

class HateSpeechClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPreditZero(HateSpeechClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

# TODO: Implement this
class NaiveBayesClassifier(HateSpeechClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        self.pos = {}
        self.neg = {}
        self.features = 0
        self.count_pos = 0
        self.count_neg = 0
        self.pos_prior = 0
        self.neg_prior = 0

    def fit(self, X, Y):
        # Add your code here!
        for i in range(len(Y)):
            curr_table = self.pos if Y[i] == 1 else self.neg
            for word,occur in enumerate(X[i]):
                curr_table[word] = occur + curr_table.get(word, 0)

        self.count_pos = sum(self.pos.values())
        self.count_neg = sum(self.neg.values())
        self.pos_prior = self.count_pos/(self.count_pos + self.count_neg)
        self.neg_prior = self.count_neg/(self.pos_prior + self.count_neg)
        self.features = len(self.pos)

        for key in range(len(self.pos)):
            self.pos[key] = (self.pos[key]+1)/(self.count_pos+self.features)
            self.neg[key] = (self.neg[key]+1)/(self.count_neg+self.features)
    
    def predict(self, X):
        # Add your code here!
        results = []
        for input in X:
            pos,neg = log(self.pos_prior),log(self.neg_prior)
            for word,occur in enumerate(input):
                pos += log(self.pos[word] ** occur)
                neg += log(self.neg[word] ** occur)
            estimate = 1 if pos/(pos+neg) < neg/(pos+neg) else 0 
            results.append(estimate)
        return results


# TODO: Implement this
class LogisticRegressionClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self):
        # Add your code here!
        raise Exception("Must be implemented")
        

    def fit(self, X, Y):
        # Add your code here!
        raise Exception("Must be implemented")
        
    
    def predict(self, X):
        # Add your code here!
        raise Exception("Must be implemented")


# you can change the following line to whichever classifier you want to use for bonus
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
