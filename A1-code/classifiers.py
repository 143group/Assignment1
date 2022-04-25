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
        # get the given classification of each sentence
        # and store the sentence and count into feature dictionary
        for i in range(len(Y)):
            curr_table = self.pos if Y[i] == 1 else self.neg
            for word, occur in enumerate(X[i]):
                curr_table[word] = occur + curr_table.get(word, 0)

        # get count of hate sentences and not hate(pos) sentences
        self.count_pos = sum(self.pos.values())
        self.count_neg = sum(self.neg.values())
        self.features = len(self.pos)

        # calculate the prior probability of hate and not hate sentences 
        self.pos_prior = self.count_pos / (self.count_pos + self.count_neg)
        self.neg_prior = self.count_neg / (self.pos_prior + self.count_neg)
        
        # add-1 smoothing 
        for key in range(len(self.pos)):
            self.pos[key] = (self.pos[key]+1) / (self.count_pos+self.features)
            self.neg[key] = (self.neg[key]+1) / (self.count_neg+self.features)
    
    def predict(self, X):
        # Add your code here!
        # Naive Bayes' algorithim
        results = []
        for input in X:
            pos, neg = log(self.pos_prior), log(self.neg_prior)
            for word, occur in enumerate(input):
                pos += log(self.pos[word] ** occur)
                neg += log(self.neg[word] ** occur)
            estimate = 1 if pos / (pos+neg) < neg / (pos+neg) else 0 
            results.append(estimate)
        return results


# TODO: Implement this
class LogisticRegressionClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self):       
        self.learning_rate = 0.03
        self.num_iterates = 1400

    def fit(self, X, Y):
        # X columns is the size of the vocabulary
        # coeffiecent vector must match the # of columns of X
        self.betas = np.zeros(X.shape[1]).T

        for _ in range(self.num_iterates):
            # sigmoid function 
            y_i = 1 / (1 + np.exp(-(X @ self.betas)))

            #  gradient = sigma x_i * beta 
            gradient_descent = self.learning_rate * np.dot((y_i - Y), X)

            # need to transpose the rightside vector to match the betas vector
            # in order to subtract the 2
            self.betas = self.betas - gradient_descent.T

    def predict(self, X):
        # binary logistic regression equation
        # if predicted value >= 0.5, classify as hate
        log_regress = (1 / (1 + np.exp(-(X @ self.betas)))).round()
        return log_regress
        


# you can change the following line to whichever classifier you want to use for bonus
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
