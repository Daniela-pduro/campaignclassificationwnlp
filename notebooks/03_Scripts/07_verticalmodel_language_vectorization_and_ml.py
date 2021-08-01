import sys
sys.path.append('..')


import warnings
warnings.filterwarnings('ignore')

# Parser
import argparse

# Get file
from campaigns.getinputdata import CleanCampaignsFileGetInfo

# Feature Engineering
from campaigns.modeling.preprocessing import VerticalEncoder, MergePredicted

# Texts Preprocessing
from campaigns.modeling.nlp import VerticalModelTFIDFVectorizer

# Texts Classification
from campaigns.modeling.classifiers import MultinomialNBClassifier

if __name__ == "__main__":
    
    # Making a parser object
    parser = argparse.ArgumentParser(
        description='Add language files, and trained vectorizer and classifier arguments')
    
    # Creating variables
    parser.add_argument('--input_root', type=str, help='Input root')
    parser.add_argument('--input_file', type=str, help='Input file')
    parser.add_argument('--output_root', type=str, help='Output root')
    parser.add_argument('--output_file', type=str, help='Output file')

    parser.add_argument('--save_trained_vectorizer', type=str, help='Path')
    parser.add_argument('--save_trained_classifier', type=str, help='Path')
    
    # Setting the 'args' variable to the values of the parsed arguments
    args = parser.parse_args()
    
    # Getting the df
    # Creating an instance
    language = CleanCampaignsFileGetInfo(args.input_root, args.input_file)

    # Encoding vertical
    # Creating an instance
    language = VerticalEncoder(language.language_df)
    
    # Changing names
    language.change_names()
    
    # Removing small samples
    language.small_samples_remover(10)
    
    # Label enconding
    language.label_encoding()
    
    # Saving the encoding df in an object
    sector_id_df = language.label_encoding_df()
    
    # Splitting
    # Creating an instance
    language = VerticalModelTFIDFVectorizer(language.language_df, sector_id_df)

    # Dividing between train and test
    # Assigining test to the unlabeled messages
    features_train, features_test, target_train, target_test = language.labeled_unlabeled_partitioner()
    
    # Vectorizing using TFIDF Approach
    tfidf_vectorizer = language.tfidf_vectorizer()
    print(tfidf_vectorizer)
    
    # Getting features from train
    tfidf_features_train, vectorizer = language.tfidf_vectorizer_train(tfidf_vectorizer, features_train)
    print("Features train shape: ", tfidf_features_train.shape)
    
    # SMOTE oversampling
    tfidf_features_train, target_train = language.target_smotter(tfidf_features_train, target_train)
    
    
    # Saving the trained vectorizer
    language.vectorizer_save(
        tfidf_vectorizer, path="../../../datasets/vectorizers/" + args.save_trained_vectorizer)
    
    # Vectorizing test using train features and predicting
    # Creating an instance
    model = MultinomialNBClassifier(features_test, tfidf_vectorizer, tfidf_features_train, target_train)
    
    # Vectorizing test (unlabeled messages)
    tfidf_features_test =  model.vectorizer_test()
    
    # Observing the dimension of the resulting feature vector
    print('Labeled Messages:', tfidf_features_train.shape)
    print('Unabeled Messages:', tfidf_features_test.shape)
    
    print(f'The dimension of our feature vector is {tfidf_features_train.shape[1]}.')
    
    # Training and test
    # Creating an instance
    classifier = model.get_multinomial_nb_classifier()
    print(classifier)
    
    # Training the model
    model.classifier_training(classifier)
    
    # Saving the trained classifier
    model.classifier_save(
        classifier, path="../../../datasets/classifiers/" + args.save_trained_classifier)
    
    # Predicting Vertical
    predictions = model.classifier_predict(classifier, tfidf_features_test)
   
    # Merging predictions with the original messages
    language_predictions = MergePredicted(predictions, language.language_df, language.test_df)
    unique, counts = language_predictions.get_predicted_frequency()
    results_df = language_predictions.merge_predicted()
    vertical_classified_df = language_predictions.get_vertical_classified_df()
    
    # Saving file
    language_predictions.save_final_df(vertical_classified_df, args.output_root, args.output_file)