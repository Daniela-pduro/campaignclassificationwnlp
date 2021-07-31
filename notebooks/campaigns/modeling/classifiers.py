# Sklearn Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Serializing
import pickle

class MultinomialNBClassifier(object):
    """
    This python class takes 'features_test' as input and:
    
    - Transforms test data into the same feature vectors as the training data.
    - Trains the classifier and predicts for test data.
    
    """
    
    # Initializer
    
    def __init__(self, features_test, vectorizer, vec_features_train, target_train):
        
        self.features_test = features_test
        self.vectorizer = vectorizer
        self.vec_features_train = vec_features_train
        self.target_train = target_train
    
    # Instance methods
    
    def vectorizer_test(self):
        """
        - Transforms test data using training data's features.
        """
        
        vec_features_test = self.vectorizer.transform(self.features_test)
        
        return vec_features_test
    
    def get_multinomial_nb_classifier(self):
        """
        - Instantiates a Multinomial Naive Bayes classifier.
        """
        
        classifier = MultinomialNB()
        
        return classifier
    
    def classifier_training(self, classifier):
        """
        - Trains the model.
        """
        
        classifier.fit(self.vec_features_train, self.target_train)
        
    def classifier_predict(self, classifier, vec_features_test):
        """
        - Predicts sector for bow_features_test. 
        """
        
        predictions = classifier.predict(vec_features_test)
        
        return predictions
    
        
    def classifier_save(self, classifier, path="../../../datasets/multinominal_nb_classifier"):
        """
        - Receives classifier and path as input.
        - Saves the classifier in a file.
        """
    
        with open(path, 'wb') as f:
            
            pickle.dump(classifier, f)
    
    def trained_classifier_load(self, path="../../../datasets/multinomial_nb_classifier"):
        """
        - Receives path as input.
        - Loads the classifier in an object.
        """
        
        with open(path, 'rb') as f:
            
            self.classifier = pickle.load(f)


class LogisticRegressionClassifier(object):
    """
    This python class takes 'features_test', 'bow_vectorizer','bow_features_train' ad 'target_train' as input and:
    
    - Transforms test data into the same feature vectors as the training data.
    - Creates a logistic regression classifier.
    - Trains the classifier and predicts for test data.
    
    """
    
    # Initializer
    
    def __init__(self, features_test, vectorizer, features_train, target_train):
        
        self.features_test = features_test
        self.vectorizer = vectorizer
        self.features_train = features_train
        self.target_train = target_train
        self.target_01_train = self.target_train['target_01'].values
        self.target_02_train = self.target_train['target_02'].values
        self.target_03_train = self.target_train['target_03'].values
        self.classifier = self.get_logistic_regression_classifier()
    
    # Instance methods
    
    def vectorizer_test(self):
        """
        - Transforms test data using training data's features.
        """
        
        features_test = self.vectorizer.transform(self.features_test)
        
        return features_test
    
    def get_logistic_regression_classifier(self):
        """
        - Instantiates a Logistic Regression classifier.
        """
        
        classifier = LogisticRegression()
        
        return classifier
    
    def classifier_training(self, target_train):
        """
        - Receives the classifier and the desired target_train as input.
        - Trains the model.
        """
        
        self.classifier.fit(self.features_train, target_train)
        
        
    def classifier_save(self, path="../../../datasets/logistic_regression_classifier"):
        """
        - Receives classifier and path as input.
        - Saves the classifier in a file.
        """
    
        with open(path, 'wb') as f:
            pickle.dump(self.classifier, f)
    
    def trained_classifier_load(self, path="../../../../datasets/logistic_regression_classifier"):
        """
        - Receives path as input.
        - Loads the classifier in an object.
        """
        
        with open(path, 'rb') as f:
            
            self.classifier = pickle.load(f)
    
    def classifier_predict(self, bow_features_test):
        """
        - Predicts sector for bow_features_test. 
        """
        
        predictions = self.classifier.predict(bow_features_test)
        
        return predictions