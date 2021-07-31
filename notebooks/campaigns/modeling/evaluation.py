# Directiories Handle
import os

import warnings
warnings.filterwarnings('ignore')

# Data Preprocessing
import pandas as pd 

# Evaluation Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score

class VerticalModelEvaluator(object):
    """
    This python class gets 'target_test' and 'predictions' as input and:
    
    - Displays a classification report.
    - Displays a confusion matrix.
    - Creates a dataframe with the actual label of each message and its prediction.
    - Displays the misclassified messages percentage.
    - Creates a document with the classification report and saves it for future reference.
    """
    
    def __init__(self, target_test, predictions, features_test):
        
        self.target_test = target_test
        self.predictions = predictions
        self.features_test = features_test
        
    def get_classification_report(self):
        
        classification_report_str = classification_report(self.target_test, self.predictions)
        
        print('\nClassification Report')
        print('======================================================')
        print('\n', classification_report_str)

        
    def get_confusion_matrix(self):
        
        confusion = confusion_matrix(self.target_test, self.predictions)
        
        print('\nConfusion Matrix')
        print('======================================================')
        print('\n', confusion)
        
    def get_results_df(self):
        
        target_test_df = pd.DataFrame(self.target_test, columns = ['actual_label'])
        features_test_df = pd.DataFrame(self.features_test, columns =['clean_message'])
        predictions_df = pd.DataFrame(self.predictions, columns = ['predicted_class'])
        results_df = pd.merge(features_test_df, target_test_df, how='outer', left_index=True, right_index=True)
        results_df = pd.merge(results_df, predictions_df, how='outer', left_index=True, right_index=True)
        
        return results_df


    def get_misclassified(self, results_df):

        misclassified_0 = len(results_df[(results_df['actual_label']==0) & (results_df['predicted_class']!=0)])
        misclassified_1 = len(results_df[(results_df['actual_label']==1) & (results_df['predicted_class']!=1)])
        misclassified_2 = len(results_df[(results_df['actual_label']==2) & (results_df['predicted_class']!=2)])
        misclassified_3 = len(results_df[(results_df['actual_label']==3) & (results_df['predicted_class']!=3)])
        misclassified_4 = len(results_df[(results_df['actual_label']==4) & (results_df['predicted_class']!=4)])
        misclassified_5 = len(results_df[(results_df['actual_label']==5) & (results_df['predicted_class']!=5)])
        misclassified_6 = len(results_df[(results_df['actual_label']==6) & (results_df['predicted_class']!=6)])
        misclassified_7 = len(results_df[(results_df['actual_label']==7) & (results_df['predicted_class']!=7)])
        misclassified_8 = len(results_df[(results_df['actual_label']==8) & (results_df['predicted_class']!=8)])
        misclassified_9 = len(results_df[(results_df['actual_label']==9) & (results_df['predicted_class']!=9)])
        misclassified_10 = len(results_df[(results_df['actual_label']==10) & (results_df['predicted_class']!=10)])
        misclassified_11 = len(results_df[(results_df['actual_label']==11) & (results_df['predicted_class']!=11)])
        misclassified_12 = len(results_df[(results_df['actual_label']==12) & (results_df['predicted_class']!=12)])
        misclassified_13 = len(results_df[(results_df['actual_label']==13) & (results_df['predicted_class']!=13)])
        
        
        total_misclassified = misclassified_0 + misclassified_1 + misclassified_2 + misclassified_3 +\
                            misclassified_4 + misclassified_5 + misclassified_6 + misclassified_7 +\
                            misclassified_8 + misclassified_9 + misclassified_10 + misclassified_11 +\
                            misclassified_12 + misclassified_13
        
        return total_misclassified / len(results_df) * 100
    
    def save_classification_report(self, path, name):
        
        path = path
        
        classification_report_dict = classification_report(self.target_test, self.predictions, output_dict=True)
        
        report_df = pd.DataFrame.from_dict(classification_report_dict).transpose()
        
        report_df.to_csv(os.path.join(path, name))
        

class MainModelEvaluator(object):
    """
    This python class gets 'target_test' and predictions' as input and:
    
    - Displays predictions, target_test and target_train mean values.
    - Displays a confusion matrix.
    - Returns true negative, true positive, false negative and false postive values.
    - Displays recall score.
    - Disaplys precision score.
    """
    
    def __init__(self, target_test, predictions):
        
        self.target_test = target_test
        #self.target_01_test = self.target_test['target_01'].values
        #self.target_02_test = self.target_test['target_02'].values
        #self.target_03_test = self.target_test['target_03'].values
        self.predictions = predictions
        self.predictions_mean = self.predictions.mean()
        self.target_test_mean = self.target_test.mean()
        #self.target_01_test_mean = self.target_test['target_01'].mean()
        #self.target_02_test_mean = self.target_test['target_02'].mean()
        #self.target_03_test_mean = self.target_test['target_03'].mean()
        
        
        
    def get_accuracy_score(self, target_test):
        
        my_accuracy_score = accuracy_score(target_test, self.predictions) * 100
        
        return my_accuracy_score

        
    def get_confusion_matrix(self, target_test):
        
        confusion = confusion_matrix(target_test, self.predictions)
        
        print('\nConfusion Matrix')
        print('======================================================')
        print('\n', confusion)
        
        tn, fp, fn, tp = confusion_matrix(target_test, self.predictions).ravel()
        
        
        return tn, fp, fn, tp
    
    def get_recall_score(self, target_test):
        
        my_recall_score = recall_score(target_test, self.predictions) * 100
        
        return my_recall_score
    
    def get_precision_score(self, target_test):
        
        my_precision_score = precision_score(target_test, self.predictions) * 100
        
        return my_precision_score