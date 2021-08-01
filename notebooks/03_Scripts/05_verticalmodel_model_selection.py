import sys
sys.path.append('..')


import warnings
warnings.filterwarnings('ignore')

# Parser
import argparse

# Get file
from campaigns.getinputdata import  LanguageFileGetInfo

# Texts Preprocessing
from campaigns.modeling.nlp import VerticalMessagePreprocessor, VerticalModelFeatureExtractor, StopWordsRemover 

# Feature Engineering
from campaigns.modeling.preprocessing import VerticalEncoder

# Texts Feature Engineering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Texts Classification
from campaigns.modeling.classifiers import MultinomialNBClassifier

# Evaluation
from campaigns.modeling.evaluation import VerticalModelEvaluator

if __name__ == "__main__":
    
    # Making a parser object
    parser = argparse.ArgumentParser(description='Add vectorizer and language arguments')
    
    # Creating variables
    parser.add_argument('--input_root', type=str, help='Input root')
    parser.add_argument('--input_file', type=str, help='Input file')
    parser.add_argument('--vectorizer', type=str, help='Input vectorizer')
    parser.add_argument('--stemmer', type=str, help='Stemmer Language')
    
    # Setting the 'args' variable to the values of the parsed arguments
    args = parser.parse_args()
    #print(args.stemmer)
    
    # Getting the df
    # Creating an instance
    language_instance = LanguageFileGetInfo(dataroot=args.input_root, datafile=args.input_file)
    
    # Feature Extraction
    # Preproceseeing texts
    # Creating an instance
    language_instance = VerticalMessagePreprocessor(language_instance.language_df, args.stemmer)
    print(args.stemmer)
    
    # Checking the number of labeled messages
    print('Labeled Messages:' , len(language_instance.language_df))
    
    # Preprocessing texts
    language_instance.preprocess_message(flag_stemming=True)
    
    
    # Encoding sector
    # Creating an instance
    language_instance = VerticalEncoder(language_instance.language_df)
    
    language_instance.change_names()
    language_instance.small_samples_remover(10)
    language_instance.label_encoding()
    sector_id_df = language_instance.label_encoding_df()
    
    
    # Vectorization
    
    # Selecting vectorizer
    vectorizers = {'CountVectorizer': CountVectorizer, 'TfidfVectorizer': TfidfVectorizer}
    vectorizer = vectorizers.get(args.vectorizer, TfidfVectorizer) 

    # Creating an instance    
    language_instance = VerticalModelFeatureExtractor(vectorizer, language_instance.language_df, sector_id_df)
    
    # Splitting the data
    features_train, features_test, target_train, target_test = language_instance.splitter()
    
    # Obtaining the vectorizer
    vectorizer = language_instance.get_vectorizer()
    print(vectorizer)

    # Training the vectorizer
    vec_features_train, vectorizer = language_instance.vectorizer_train(features_train)
    
    # SMOTE oversampling
    vec_features_train, target_train = language_instance.target_smotter(vec_features_train, target_train)  
   
    # Detecting Verticals
    
    # Transforming test data into the same feature vector as training data
    # Creating an instance
    model = MultinomialNBClassifier(features_test, vectorizer, vec_features_train, target_train)
    
    # Transforming test data
    vec_features_test = model.vectorizer_test()

    print(f'The dimension of our feature vector is {vec_features_train.shape[1]}.')
    
    
    # Training the classifier and predicting for test data
    # Creating an instance
    classifier = model.get_multinomial_nb_classifier()
    
    # Training the model
    model.classifier_training(classifier)
    
    # Predicting vertical
    predictions = model.classifier_predict(classifier, vec_features_test)
    
    # Evaluation
    evaluation = VerticalModelEvaluator(target_test, predictions, features_test)
    
    results_df = evaluation.get_results_df()
    
    print('Misclassified:', evaluation.get_misclassified(results_df))