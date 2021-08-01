# Directories Handle
import os


# Serializing and Deserializing
import pickle

# Data Processing
import pandas as pd

class ModelApplier(object):
    
    def __init__(self, classifier_path, vectorizer_path):
        
        self.load_vectorizer_from_file(vectorizer_path)
        self.load_classifier_from_file(classifier_path)
        
        
    def load_vectorizer_from_file(self, path):
        """
        - Receives path as input.
        - Loads the vectorizer in an object.
        """
        
        with open(path, 'rb') as f:
            
            self.vectorizer = pickle.load(f)
            
        print(self.vectorizer)
    
    def load_classifier_from_file(self, path):
        """
        - Receives path as input.
        - Loads the classifier in an object.
        """
        
        with open(path, 'rb') as f:
            
            self.classifier = pickle.load(f)
        
        print(self.classifier)

    
    def classify_message(self, message):
        """
        - Receives message as input.
        - 
        """
        
        message = [message]
        message = self.vectorizer.transform(message)
        prediction = self.classifier.predict(message)
        
        return prediction
    

class ModelUnwrapper(object):
    
    def __init__(self, radio, model_path='../../../datasets/pipelines/main_model_target_01_pipeline'):
        
        self.model = self.load_wrapped_model_from_file(model_path)
        self.radio = radio
        
    def load_wrapped_model_from_file(self, path):
        """
        - Receives path as input.
        - Loads the model in an object.
        """
        
        with open(path, 'rb') as f:
            
            model = pickle.load(f)
            
            return model
            
    def make_predictions_with_subject(self, message_input):
        
        vertical = self.radio
        
        aux_dic = {'clean_subject': message_input, 'vertical': vertical}
            
        columns = ['clean_subject', 'vertical']
            
        index = ['a']
            
        aux_df = pd.DataFrame(aux_dic, columns=columns, index=index)

        prediction = self.model.predict(aux_df)
        
        return prediction
    
    def make_predictions_with_message(self, message_input):
        
        vertical = self.radio
        
        aux_dic = {'clean_message': message_input, 'vertical': vertical}
            
        columns = ['clean_message', 'vertical']
            
        index = ['a']
            
        aux_df = pd.DataFrame(aux_dic, columns=columns, index=index)

        prediction = self.model.predict(aux_df)
        
        return prediction


class ExtractKeywordsFromFile(object):
    """
    - Gets keywords from file.
    """
    
    def __init__(self, keyword_vertical, keyword_kpi):
        
        self.keyword_vertical = keyword_vertical
        self.keyword_kpi = keyword_kpi
        
    def keywords_retriever(self):
        
        file_path = '../../../keywords/' + self.keyword_vertical + '/' + self.keyword_kpi + '/keywords.txt'
        
        keywords_list = []
        
        with open(file_path,'r') as keywords:
            
            for phrase in keywords:
                
                keywords_list.append(phrase)
                
        return keywords_list
  