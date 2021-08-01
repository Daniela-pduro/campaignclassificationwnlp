import sys
sys.path.append('..')


import warnings
warnings.filterwarnings('ignore')

# Parser
import argparse

# Get file
from campaigns.getinputdata import CleanClassifiedCampaignsFileGetInfo

# Preprocessing
from campaigns.modeling.preprocessing import ClassifiedCampaignsGetPipeline

# Evaluation
from campaigns.modeling.evaluation import MainModelEvaluator

import pickle


if __name__ == "__main__":
    
    # Making a parser object
    parser = argparse.ArgumentParser(description='Selects features and target')
    
    # Creating variables
    parser.add_argument('--input_root', type=str, help='Input root')
    parser.add_argument('--input_file', type=str, help='Input file')
    
    parser.add_argument('--features', nargs="+", help='Features')
    parser.add_argument('--target', nargs="+", help='Target')
    parser.add_argument('--text_column', type=str, help='Subject or Message')
    parser.add_argument('--pickle_file', type=str, help='Save the model')

    # Setting the 'args' variable to the values of the parsed arguments
    args = parser.parse_args()

    # Getting the df
    # Creating an instance
    campaigns = CleanClassifiedCampaignsFileGetInfo(args.input_root, args.input_file)
    campaigns
    
   
    # Creating a scikit-learn Pipeline
    # Creating an instance
    campaigns = ClassifiedCampaignsGetPipeline(campaigns.campaigns_w_vertical_df,
                                          features=campaigns.campaigns_w_vertical_df[args.features],
                                          target = campaigns.campaigns_w_vertical_df[args.target])
    
    # Getting transformers
    cat_transformer = campaigns.get_cat_transformer()
    
    text_transformer = campaigns.get_text_transformer()
    
    # Applying the transformers to features using "ColumnTransformer"
    ct = campaigns.get_column_transformer(cat_transformer, text_transformer, args.text_column)
    
    # Splitting the dataset and fitting ColumnTransformer to the Pipeline
    features_train, features_test, target_train, target_test = campaigns.train_test_splitter()
    
    pipeline = campaigns.get_pipeline(ct)
    
    # Predicting targets
    model = pipeline.fit(features_train, target_train)
    
    predictions = pipeline.predict(features_test)
    
    # Evaluation
    evaluation = MainModelEvaluator(target_test, predictions)
    
    print("Predictions mean: ", evaluation.predictions_mean)
    print("Target Test: ", target_test[args.target].values.mean())
    
    if evaluation.predictions_mean < target_test[args.target].values.mean():
        
        print("Predictions mean is not greater than target test mean")
        
    else:
        
        print("Predictions mean is greater than target test mean.")
        print("We should improve the classifier.")
        
    # Accuracy
    print('accuracy %s' % evaluation.get_accuracy_score(evaluation.target_test))
    
    # Confusion Matrix
    tn, fp, fn, tp = evaluation.get_confusion_matrix(evaluation.target_test)
    
    print("True Negatives: ", tn)
    print("True Positives: ", tp)
    print("False Negatives ", fn)
    print("False Positives: ", fp)
    
    # Recall score
    print("Recall Score: ", evaluation.get_recall_score(evaluation.target_test))
    
    # Precision score
    print("Precision Score: ", evaluation.get_precision_score(evaluation.target_test))
    
    # Save model
    with open(args.pickle_file, 'wb') as f:
            pickle.dump(model, f)