# Directories Handle
import os


# Data Processing
import pandas as pd
import numpy as np

# Cleaning Texts
from string import punctuation
import re

# Detecting Language
import langdetect

# NLP
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

# Partition
from sklearn.model_selection import train_test_split

# Preparing for deep learning
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

# Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec

# Target Preparation
from collections import Counter
from imblearn.over_sampling import SMOTE

# Deep Learning
from keras import preprocessing
from keras.preprocessing.sequence import pad_sequences

# Data visualization
import matplotlib.pyplot as plt

# Timing
import time

# Serializing
import pickle


class CampaignsCleaner(object):
    """
    This python class reads the tabular data contained in a dataframe and:
    
    - Removes campaigns from users with several accounts.
    - Removes campaigns with empty messages.
    
    """
    # Initializer
    def __init__(self, campaigns_df):
        
        self.campaigns_df = campaigns_df
    
    # Instance Methods    
    def empty_messages_remover(self):
        """
        - Removes campaigns with empty messages.
        
        """
        
        self.campaigns_df.dropna(subset=['message'], inplace=True)
        self.campaigns_df.dropna(subset=['subject'], inplace=True)
        
        print('Number of campaigns within the dataset after removing empty messages: ', self.campaigns_df.shape[0])
    
    def duplicated_users_remover(self):
        """
        - Removes duplicated users' campaigns.
        
        """
        to_remove = [82045, 82171, 84179, 82391, 82116, 83709, 83710, 82398, 82169,
                     83711, 82393, 82929, 82394, 83707, 84177, 85641, 83708, 82120,
                     84178, 84169, 85642, 82173, 84506, 84460, 84458, 84457, 84505,
                     84497, 84442, 84456, 84452, 84443, 84504, 84451, 84455, 84453,
                     84454, 84461, 84437, 84446, 84459, 84503, 84447, 84440, 84439,
                     84438, 84171, 84176, 84172, 82121, 84445, 84441]
        
        for code in to_remove:
            self.campaigns_df.drop(self.campaigns_df[self.campaigns_df['sender'] == code].index, inplace= True)
            
        print('Number of campaigns within the dataset after removing users with duplicated accounts: ', 
              self.campaigns_df.shape[0])


class TextsCleaner(object):
    """
    This python class reads the tabular data contained in a dataframe and:
   
   - Removes nonwords.
    
    """
    
    # Initializer
    def __init__(self, campaigns_df):
        
        self.campaigns_df = campaigns_df
        self.allowed_chars = ' AaÁáÀàÂâBbCcÇçDdEeÉéÈèÊêFfGgHhIiÍíÌìÎîJjKkLlMmNnÑñOoÓóÒòÔôPpQqRrSsTtUuÚúÙùÜüÛûVvWwXxYyZz'
        self.non_words = list(punctuation)
        self.non_words.extend(map(str,range(0,10)))
        self.non_words.extend(['¿', '¡', '"','♦','•','·','€','®','✔','©','ª','º','→','↑','⇢','🚗',
                               '🎉','👉','📌','🔥','⚽','💰','🏆','✌️','🙁','🙃','🔻','⚔','🤑',
                               '🚨','👀','⚠️','🎰','✅','🔄','⭐','😍','💛','👈','🎊','🎄','🎅',
                               '🤷🏻','👕','🤯','💝','🔝','💸','⏳','♻️','🤘','🏴','🎁','😎','🔴',
                               '⚫','📆','🚀','🔔','🏅','🎾','👌','📲','🎟️','🥳','🎙','📷','👇🏼',
                               '🏖️ ','🤔','💥','🎃', '👇','🛍️','🦋','👣','🎬','😉','💎','⏰',
                               '👋🏻','🎶','🍀','🔹','🤗','✈️','⬇️','⚡️','📍','🐶', '👍','🎟️',
                               '🌲', '🌍','✨','😊', '”', '“','⚖', '👟','¨', '►','🌟','«','»',
                               '🕺🏻','📞', '’','°', '❤️','♥','´','…', '☎️', '✉️', '–', '●',
                               '‘','’','🎓','☘️','🌈','😃','💻', '❤', '—', '➜', '🤩', '📣',
                               '°','⇒', '™', '👆', '²', '✓','★', '🌹', '🌺','📺', '🛋️', '📖',
                               '🤫', '😀', '☰','✕', '🥇','✆','✉','📣','🗓', '🏖️','💥', '🚚',
                               '🍁', '🍷','【', '】', '🔁', '◆', '🌐', '😏', '💘', '💙', '👋',
                               '🥗', 'ℹ️', '📚', '✏️', '🥕', '⭕️', '🏻', '❌', '🥘', '😜','❓',
                               '🍠', '🥔', '👧', '🖤', '🍂', '🌷', '🍅', '🌀', '🤤', '🍆', '🌞',
                               '💶','🥑', '📝', '🌸', '🙏', '🏼', '🎧', '💦', '🍓', '🍉', 
                               '♣', '♠', '🍌', '🧡', '💪', '◆', '🙌🏼','🍃','🏼', '💚', '🔅', '👋',
                               '😳', '📧', '🥪', '🌎', '🕘', '➡', '💉', '📄', '💡', '🌐', '🙂',
                               '☎', '🚙', '🎒', '💳', '💠', '▶', '💖', '📰', '❀', '☀️', '🏻',
                               '🏻', '🍇', '🍎', '🎥', '➡️', '🌰', '🎈', '⚡', '👨', '🌽', '🎯', 
                               '☺️', '💬', '🥬', '👩🏼', '🇪🇸', '🗺️', '️⃣', '🍲', '💕', '😅',
                               '😬', '😥', '🍏', '🦃', '🌿', '❄️', '😘', '😮', '🔁', '⚡', '📰',
                               '🌻', '💨', '👨','⚕️', '📚', '▶', '♡', '🙌'])

    
    # Instance Methods
    
    def word_counter(self, subject, message):
        """
        - Counts the total number of words in the column `subject`, including nonwords (e.g. emojis, numbers)
        - Counts the total number of words in the column `message`, including nonwords (e.g. emojis, numbers)
        """
        try:
            
            subject_ttl_words = self.campaigns_df[subject].apply(lambda x: len(x.split(' '))).sum()
            message_ttl_words = self.campaigns_df[message].apply(lambda x: len(x.split(' '))).sum()
        
        except Exception as e:
            
            print(str(e))
        
        print('Total number of words in subject: ', subject_ttl_words)
        print('Total number of words in message: ', message_ttl_words)
    
    
    
    def text_cleaner(self, message):
        """
        - Receives message as input and returns a clean version.
        """
        
        try:
            
            # Replace Encodings
            message = message.replace('\xa0', ' ')
            message = message.replace('\u200c', ' ')
            message = message.replace('\ufeff', ' ')
            message = message.replace('\xad', ' ')
            message = message.replace('\u200b', ' ')
            message = message.replace('\ufeff', ' ')
            message = message.replace('\u001000', ' ')
            message = message.replace('\u200e', ' ')
            message = message.replace('\u200c', ' ')

            # Changing to lowecase
            message = message.lower().strip()
            
            # Padding and removing non words
            message = ''.join([' ' + char + ' ' if char in self.non_words else char for char in [char for char in re.sub(r'http\S+','http', message, flags=re.MULTILINE) if char in self.allowed_chars]])
            
            # Tokenization
            message = message.split(' ')

            # Removing extra empty tokens
            message = ' '.join(message).split()

            # Back to string
            message = ' '.join(message)

            return message

        except:
        
            return message
           
    def clean_texts(self):
        """
        - Applies text_cleaner method over the dataframe.
        - Counts the total number of words in the column `message` after cleaning, and prints out the result.
        - Counts the total number of words in the column `subject` after cleaning, and prints out the result.
        """
        
        self.campaigns_df['clean_subject'] = self.campaigns_df['subject'].apply(self.text_cleaner)
        self.campaigns_df['clean_message'] = self.campaigns_df['message'].apply(self.text_cleaner)
        
    
    def save_final_df(self, path, filename):
        """
        -  Saves the df in the specified path.
        """
        self.campaigns_df.to_csv(os.path.join(path,filename), header=None)
        
        
class LanguageDetector(object):
    """
    This python class reads the texts contained in a dataframe and:
    
   - Counts the number of words in the whole set.   - Removes empty messages.
   - Detects the main language in which the texts are written.
   - Filters the dataframe by language.
   - Plots each language percentage within the dataframe.
   - Returns a new dataframe per language.
    """

    # Initializer
    def __init__(self, campaigns_df):
        
        start = time.time()
        self.campaigns_df = campaigns_df
        self.empty_messages_remover()
        print('1 Number of seconds since it started running: ' + str(time.time()-start))
        self.detect_lang()
        print('2 Number of seconds since it started running: ' + str(time.time()-start))
        self.languages_list = self.list_languages()
        print('3 Number of seconds since it started running: ' + str(time.time()-start))
        
    # Instance Methods
    
    def word_counter(self):
        """
        - Counts the total number of words in message texts.
        """
        
        return  self.campaigns_df['clean_message'].apply(lambda x: len(x.split(' '))).sum()
    
    def empty_messages_remover(self):
        """
        - Removes empty messages from the dataframe.
        """
        self.campaigns_df.dropna(subset=['clean_message'], inplace=True)
        
        return self.campaigns_df.shape
    
    
    def language_detector(self, message):
        """
        - Receives message as input and detects the language in which it is written.
        """
        
        try:
            # Detect language
            
            return langdetect.detect(message)
        
        except:
            
            return message
    
    def detect_lang(self):
        """
        - Applies language_detector method over the dataframe.

        """
        
        self.campaigns_df['language'] = self.campaigns_df['clean_message'].apply(self.language_detector)
        
    def plot_languages(self):
        """
        - Plots a bar chart with the proportion of messages in each language.
        
        """
        
        fig, ax = plt.subplots(1, figsize=(12,5))
        plt.suptitle('Number of campaigns per language')
        
        ax.bar(self.campaigns_df['language'].value_counts().index,
        self.campaigns_df['language'].value_counts().values)
        plt.xlabel("Language") 
        plt.ylabel("Count");
        
        
    def list_languages(self):
        """
        - Lists the language code of all the languages appearing on the dataframe.
        """
        
        languages_list = self.campaigns_df['language'].unique()
        
        return languages_list
    
    def replace_language(self, original_language_code, new_language_code='es'):
        """
        - Changes the language label of a message to Spanish by default, 
        or to any other language.
    
        """
        
        self.campaigns_df.loc[(self.campaigns_df['language'] == original_language_code), 'language']= new_language_code
        
    
    def language_filter(self, path):
        """
        -  Filters and creates one csv file per language.
        -  Drops the column 'language'.
        -  Saves the file in the specified path.

        """
        
        for language_code in self.languages_list:
            
            df = self.campaigns_df[self.campaigns_df['language'] == language_code].\
            drop('language', axis=1).\
            to_csv(os.path.join(path, str(language_code) + '.csv'), header=None)            


class StopWordsRemover(object):
    """
    This Python class receives as input:
    
    - The tabular data contained in a specific language file.
    - The name of a specific language.
    
    And performs the following task:
    
    - Removing stopwords of the language mentioned in the input argument.
    
    """
    
    # Initializer
    def __init__(self, language_df, language):
        
        self.language_df = language_df
        self.stopwords_list = stopwords.words(language)
        self.stopwords_list.extend(['datos', 'información', 'modificar', 'leyendoseguir', 'correo',
                                    'adjuntos', 'mail', 'online', 'web', 'info', 'email', 'newsletter',
                                    'suscripción'])
        

    # Instance Methods
    
    def stopwords_remover(self, message):
        """
        - The function gets a text sample and:
        
        - Applies the NLP preprocessing technique of removing stopwords to email messages.
        - Returns a processed version.
        """
        
        try:
            
            # Tokenize
            message = message.split(' ')

            # Remove empty strings
            message = ' '.join(message).split()
            
            # Remove stopwords
            
            if self.stopwords_list is not None:
                
                message = [word for word in message if word not in self.stopwords_list]
                
            # Back to string
            message = " ".join(message)
                
            return message
        
        except Exception as e:
            
            print(e)
            
            return message
        
    def remove_stopwords(self):
        """
        - Applies stopwords_remover method over the dataframe
        
        """
        
        self.language_df['clean_message'] = self.language_df['clean_message'].apply(self.stopwords_remover)
        
        
    def ngrams_remover(self, message):
        """
        - The fucntion gets a list of ngrams and removes them for the messages passed as argument.
        """
        try:
        
            for sentence in self.ngrams_list:
            
                # removing n-grams
                message = message.replace(sentence,' ')
            
                # tokenization
                message = message.split(' ')
            
                # removing extra empty tokens
                message = ' '.join(message).split()
            
                # back to string
                message = ' '.join(message)
            
            return message
            
        except:
        
            return message
    
    def remove_ngrams(self, ngrams_list):
        """
        - - Applies ngrams_remover method over the dataframe.
        """
        
        self.ngrams_list = ngrams_list
        
        self.language_df['clean_message'] = self.language_df['clean_message'].apply(
            self.ngrams_remover)
        
    def add_ngrams_to_file(self, ngrams_list):
        """
        - Adds ngram to file.
        """
        
        with open('../../../corpora/irrelevant_40.txt','a') as irrelevant_40:
            for sentence in ngrams_list:
                irrelevant_40.write(sentence + '\n') 
        
    def get_ngrams_from_file(self):
        """
        - Gets ngram list from file
        """
        
        with open('../../../corpora/irrelevant_40.txt', 'r') as irrelevant_40:
            
            irrelevant_list = []
            
            for sentence in irrelevant_40:
                
                irrelevant_list.append(sentence)
                
            irrelevant_list = [re.sub('\n','', sentence) for sentence in irrelevant_list]
            
        return irrelevant_list    
    
    def save_final_df(self, path, filename):
        """
        -  Saves the df in the specified path.
        """
        self.language_df.to_csv(os.path.join(path,filename), header=None)
            

class VerticalMessagePreprocessor(object):
    """
    This python class receives as input:
    
    - The tabular data contained in a specific language file.
    - The name of a specific language.
    - The argument 'remove_unlabeled'.
    
    And performs the following tasks:
    
    - Removes rows with empty messages.
    - Keeps labeled messages only by default, or keeps both labeled and unlabeled messages if
        `remove_unlabeled` is set to False.
    - Preprocesses messages applying NLP techniques: tokenization and stemming.  
    """
    
    # Initializer
    def __init__(self, language_df, language, remove_unlabeled=True):
        
        self.language_df = language_df
        self.language_df.dropna(subset=['clean_message'], inplace=True)
        if remove_unlabeled:
            self.language_df = self.language_df[self.language_df['sector'].notnull()]
        self.stemmer = SnowballStemmer(language)


    # Instance Methods
    
    def message_preprocessor(self, message, flag_stemming=False):
        """
        - The function gets a text sample and:
        
        - Applies NLP preprocessing techniques to email messages.
        - Returns a processed version.
        """
        
        try:
            
            # Tokenize
            message = message.split(' ')

            # Remove empty strings
            message = ' '.join(message).split()
        
            #Stemming
            if flag_stemming == True:

                message = [self.stemmer.stem(word) for word in message]

            # Back to string
            message = " ".join(message)

            return message

        except Exception as e:
            print(e)
            return message
            
        
    def preprocess_message(self, flag_stemming=False):
        """
        - Applies message_preprocessor method over the dataframe.
        """
        
        self.language_df['clean_message'] = self.language_df['clean_message'].apply(lambda x: self.message_preprocessor(x, flag_stemming))
        
    def save_final_df(self, path, filename):
        """
        -  Saves the df in the specified path.
        """
        self.language_df.to_csv(os.path.join(path,filename), header=None)


class MainModelTextsPreprocessor(object):
    """
    This python class receives as input:
    
    - The tabular data contained in a specific language file.
    - The name of a specific language.
    - The argument 'remove_unlabeled'.
    
    And performs the following tasks:
    
    - Removes rows with empty subjects and/or empty messages.
    - Keeps labeled messages only by default, or keeps both labeled and unlabeled messages if
        `remove_unlabeled` is set to False.
    - Preprocesses messages applying NLP techiniques: tokenization and stemming.  
    """
    
    # Initializer
    def __init__(self, campaigns_w_vertical_df, language, remove_unlabeled=True):
        
        self.campaigns_w_vertical_df = campaigns_w_vertical_df
        self.campaigns_w_vertical_df.dropna(subset=['clean_message'], inplace=True)
        self.campaigns_w_vertical_df.dropna(subset=['clean_subject'], inplace=True)
        if remove_unlabeled:
            self.campaigns_w_vertical_df = self.campaigns_w_vertical_df[self.campaigns_w_vertical_df['sector'].notnull()]
        self.stemmer = SnowballStemmer(language)

    
    # Instance Methods
    def message_preprocessor(self, message, flag_stemming=False):
        """
        - The function gets a text sample and:
        
        - Applies NLP preprocessing techniques to email messages.
        - Returns a processed version.
        """
        
        try:
            
#             if language in self.snowball_languages:
            
             # Tokenize
            message = message.split(' ')

            # Remove empty strings
            message = ' '.join(message).split()

            #Stemming
            if flag_stemming == True:

                message = [self.stemmer.stem(word) for word in message]

            # Back to string
            message = " ".join(message)

            return message

        except Exception as e:
            print(e)
            return message
            
        
    def preprocess_message(self, flag_stemming=False):
        """
        - Applies message_preprocessor method over the dataframe.
        """
        
        self.campaigns_w_vertical_df['clean_message'] = self.campaigns_w_vertical_df['message'].apply(lambda x: self.message_preprocessor(x, flag_stemming))
        
        self.campaigns_w_vertical_df['clean_subject'] = self.campaigns_w_vertical_df['subject'].apply(lambda x: self.message_preprocessor(x, flag_stemming))
    
    
        
    def save_final_df(self, path, filename):
        """
        -  Saves the df in the specified path.
        """
        self.campaigns_w_vertical_df.to_csv(os.path.join(path,filename), header=None)


class VerticalModelFeatureExtractor(object):
    """
    This python class receives the name of a feature vectorization method as input, a language-filtered df and the vertical-encoded df and:
    
    - Splits the data into train and test.
    - Transforms messages from the training set into numerical features using the stated vectorization method.    
    
    """
    
    # Initializer
    
    def __init__(self, cls, language_df, sector_id_df):
        
        self.language_df = language_df
        self.sector_id_df = sector_id_df
        self.features = self.language_df['clean_message'].values
        self.target = self.language_df['sector_cat_id'].values
        self.cls = cls
        self.vectorizer = self.get_vectorizer()
    
    # Instance Methods
    
    def target_plotter(self, target):
        """
        - Plots target variable distribution.
        """
        
        # summarize the distribution
        counter = Counter(target)
        
        for key, value in counter.items():
            pct = value/len(target) * 100
            print('Vertical=%d, n=%d (%.3f%%)' % (key, value, pct))
            
        # plot the distribution
        
        plt.bar(counter.keys(), counter.values())
        plt.show()
        
    def splitter(self):
        
        """
        - Splits labeled data data in 90% training and 10% test
        """
        
        features_train, features_test, target_train, target_test = train_test_split(self.features, self.target, test_size=0.1)       
    
        return features_train, features_test, target_train, target_test
    
    def target_smotter(self, features_train, target_train):
        """
        - Applies oversampling techniques.
        """
        
        oversample = SMOTE()
        
        features_train, target_train = oversample.fit_resample(features_train, target_train)
        
        return features_train, target_train
    
    
    def labeled_unlabeled_partitioner(self):
        """
        - Partitions off the specific language dataset to create a set with the labeled messages and 
        a set with the unlabeled ones.
        """
        
        # Labeled messages (removing unlabeled)
        train = self.language_df[self.language_df['sector_cat_id'] != -1]
        
        # Defining train and test
        features_train = train['clean_message'].values
        features_test = self.test_df['clean_message'].values 
        target_train = train['sector_cat_id'].values
        target_test = self.test_df['sector_cat_id'].values
        
        return features_train, features_test, target_train, target_test
    
    def get_vectorizer(self):
        """
        Takes 'features_train' as parameter and:
        
        - Instantiates a vectorizer using the selected feature extraction method.
        
        """
        
        vectorizer = self.cls()
        
        return vectorizer
    
    def vectorizer_train(self, features_train):
        """
        - Extracts features from the training data contained in 'features_train'.
        """
        
        vec_features_train = self.vectorizer.fit_transform(features_train).toarray()
        
        return vec_features_train, self.vectorizer
    

class VerticalModelTFIDFVectorizer(object):
    """
    This python class reads the tabular data contained in a specific language file and:
    
    - Splits the data into train and test.
    - Transforms messages from the training set into numerical features using TFIDF.
    - Saves the trained vectorizer.
    - Loads the trained vectorizer.
    """
    
    # Initializer
    def __init__(self, language_df, sector_id_df):
        
        self.language_df = language_df
        self.sector_id_df = sector_id_df
        self.test_df = self.language_df[self.language_df['sector_cat_id'] == -1]
        self.features = self.language_df['clean_message'].values
        self.target = self.language_df['sector_cat_id'].values

    # Instance Methods
    
    def target_plotter(self, target):
        """
        - Plots target variable distribution.
        """
        
        # summarize the distribution
        counter = Counter(target)
        
        for key, value in counter.items():
            pct = value/len(target) * 100
            print('Vertical=%d, n=%d (%.3f%%)' % (key, value, pct))
            
        # plot the distribution
        
        plt.bar(counter.keys(), counter.values())
        plt.show()
    
    def splitter(self):
        """
        - Splits labeled data data in 90% training and 10% test
        """
        
        features_train, features_test, target_train, target_test = train_test_split(self.features, self.target, test_size=0.1)       
    
        return features_train, features_test, target_train, target_test
    
    def labeled_unlabeled_partitioner(self):
        """
        - Partitions off the specific language dataset to create a set with the labeled messages and 
        a set with the unlabeled ones.
        """
        
        # Labeled messages (removing unlabeled)
        train = self.language_df[self.language_df['sector_cat_id'] != -1]
        
        # Defining train and test
        features_train = train['clean_message'].values
        features_test = self.test_df['clean_message'].values 
        target_train = train['sector_cat_id'].values
        target_test = self.test_df['sector_cat_id'].values
        
        return features_train, features_test, target_train, target_test
    
    
    def target_smotter(self, features_train, target_train):
        """
        - Applies oversampling techniques.
        """
        
        oversample = SMOTE()
        
        features_train, target_train = oversample.fit_resample(features_train, target_train)
        
        return features_train, target_train
    
    def tfidf_vectorizer(self):
        """
        Takes 'features_train' as parameter and:
        
        - Instantiates a vectorizer using CountVectorizer.
        
        """
        
        tfidf_vectorizer = TfidfVectorizer(max_features=10000)
        
        return tfidf_vectorizer
    
    def tfidf_vectorizer_train(self, tfidf_vectorizer, features_train):
        """
        - Extracts features from the training data contained in 'features_train'.
        """
        
        tfidf_features_train = tfidf_vectorizer.fit_transform(features_train).toarray()
        
        return tfidf_features_train, tfidf_vectorizer
    
    def vectorizer_save(
        self, tfidf_vectorizer, path="../../../datasets/vertical_model_tfidf_vectorizer"):
        """
        - Receives path as input.
        - Saves vectorizer.vocanulary_ in a file.
        """
        
        with open(path, 'wb') as f:
            
            pickle.dump(tfidf_vectorizer, f)
            
    def trained_vectorizer_load(self, path="../../../datasets/vertical_model_tfidf_vectorizer"):
        """
        - Receives vectorizer path as input.
        - Loads the classifier as an object.
        """
        
        with open(path, 'rb') as f:
            
            tfidf_vectorizer = pickle.load(f)
            
        return tfidf_vectorizer
    
class MainModelFeatureExtractor(object):
    """
    This python class receives the name of a feature vectorization method as input and the clean campaigns df; and:
    
    - Splits the data into train and test.
    - Transforms messages from the training set into numerical features using the stated vectorization method.    
    
    """
    
    # Inititalizer
    
    def __init__(self, cls, campaigns_w_vertical_df, features):
        self.campaigns_w_vertical_df = campaigns_w_vertical_df
        self.features = features
        #self.features = self.campaigns_w_vertical_df['message'].values
        self.target_01 = self.campaigns_w_vertical_df['open_rate_result'].values
        self.target_02 = self.campaigns_w_vertical_df['ctr_result'].values
        self.target_03 = self.campaigns_w_vertical_df['ctor_result'].values
        self.cls = cls
        self.vectorizer = self.get_vectorizer()
        
    # Instance methods
    
    def splitter(self):
        
        features_train, features_test, target_train, target_test = train_test_split(self.features, pd.DataFrame({'target_01': self.target_01, 'target_02': self.target_02,
                                       'target_03': self.target_03}), test_size=0.1)       
    
        return features_train, features_test, target_train, target_test
    
    def get_vectorizer(self):
        """
        Takes 'features_train' as parameter and:
        
        - Instantiates a vectorizer using the selected feature extraction method.
        
        """
        
        vectorizer = self.cls(max_features=10000)
        
        return vectorizer
    
    
    def vectorizer_train(self, features_train):
        """
        - Extracts features from the training data contained in 'features_train'.
        """
        
        vec_features_train = self.vectorizer.fit_transform(features_train).toarray()
        
        return vec_features_train, self.vectorizer
    


    def vectorizer_save(self, bow_vectorizer, path="../../../datasets/bow_vectorizer"):
        """
        - Receives path as input.
        - Saves vectorizer.vocanulary_ in a file.
        """
        
        with open(path, 'wb') as f:
            
            pickle.dump(bow_vectorizer, f)
            
    def trained_vectorizer_load(self, path="../../../datasets/bow_vectorizer"):
        """
        - Receives vectorizer path as input.
        - Loads the classifier as an object.
        """
        
        with open(path, 'rb') as f:
            
            bow_vectorizer = pickle.load(f)
            
        return bow_vectorizer    


class VerticalModelGetWordVectors(object):
    """
    This python class receives a dataframe as input a the name of the language, and:
    
    - Prepares the target converting the column into categorical (one-hot encoding).
    - Prepares texts from messages as features: tokenization, removing empty tokens, removing stopwords.
    - Splits between train and test.
    - Transforms these tokens into numerical features using a Word2Vec model.    
    
    """
    
    # Initializer 
    
    def __init__(self, language_df, language):
        
        self.language_df = language_df
        self.features = self.language_df['clean_message'].values
        self.target = self.language_df['sector_cat_id'].values
        print('Reshaping the target with one-hot encoding')
        self.vertical_to_categorical()
        self.stopwords_list = stopwords.words(language)
        print('Converting our texts into lists of tokens')
        self.tokenize_messages()
        print('Removing empty tokens')
        self.remove_empty_tokens()
        print('Removing stopwords')
        self.remove_stopwords()
        self.sentences = self.language_df['clean_message']
        
    # Instance Methods
    
    def vertical_to_categorical(self):
        """
        - Reshapes the target from a vector that contains values for each sector to a matrix with a boolean (one-hot encoding).
        """
        
        encoder = LabelEncoder()
        encoder.fit(self.target)
        encoded_target = encoder.transform(self.target)
        self.target = np_utils.to_categorical(encoded_target)

    def message_tokenizer(self, message):
        """
        - Tokenizes texts from messages.
        """
        
        try:
            
            if message != None:
                
                message = message.split(' ')
                
            else:
                
                message = 0
                
            return message
        
        except:
            
            return 0
        
    def tokenize_messages(self):
        """
        - Applies message_tokenizer method over the dataframe.
        """
            
        self.language_df['clean_message'] = self.language_df['clean_message'].apply(self.message_tokenizer)
        

    def empty_tokens_remover(self, message):
        """
        - Removes empty tokens.
        """
        
        try:
            
            if message != None:
                
                message = ' '.join(message_sample).split()
                
            else:
                
                message = message
            
            return message
        
        except:
            
            return message
        
        
    def remove_empty_tokens(self):
        """
        - Applies empty_tokens_remover method over the dataframe.
        """
        
        self.language_df['clean_message'] = self.language_df['clean_message'].apply(self.empty_tokens_remover)
        
    
    def stopwords_remover(self, message):
        """
        - Removes stopwords.
        """
        
        try:
            
            if self.stopwords_list is not None:
                
                message = [word for word in message if word not in self.stopwords_list]
                
            else:
                
                message = message
                
            return message
        
        except:
            
            return message
        
    def remove_stopwords(self):
        """
        - Applies stopwords_remover method over the dataframe.
        """
        
        self.language_df['clean_message'] = self.language_df['clean_message'].apply(self.stopwords_remover)
        
    def splitter(self):
        
        """
        - Splits labeled data data in 90% training and 10% test
        """
        
        
        features_train, features_test, target_train, target_test = train_test_split(self.sentences, self.target, test_size=0.1, 
                                                                                    random_state=0, stratify=self.target)       
    
        return features_train, features_test, target_train, target_test
    
    def getting_embedding_model(self, features_train):
        """
        - Fits Word2Vec algorithm on training data (our corpus).
        - Learns a vector representaion fo each word.
        """
        
        corpus = features_train
        print('Corpus shape: ', corpus.shape)
        
        model = Word2Vec(corpus, window=5, min_count=1)
        
        return corpus, model
    
    def save_wv_model(self, model, file = '../../../../models/', full=True):
        """
        - Saves trained word vectors.
        """
        
        if full == True:
            
            model.save(file + 'stored_model.wv')
        
        else:
            
            model.wv.save_word2vec_format(file + 'stored_model.txt', binary='False')
        
    
    def retrieving_wv_model(self, file = '../../../../models/stored_model.wv'):
        
        retrieved_model = Word2Vec.load(file)
        
        return retrieved_model

class VerticalModelGetPaddedSeqs(object):
    """
    This Python class takes a preprocessed corpus (lists of lists of n-grams) and:
    
    - Converts it into a list of sequences using tensorflow.
    - Pads out each sequence to the same length create a feature matrix.
    - Applies same feature enginnering techniques to the test set.    
    """
    
    # Initializer
    
    def __init__(self, corpus, features_test):
        
        self.corpus = corpus
        self.features_test = features_test
        
    # Instance methods

    def corpus_get_list_of_lists(self):
        """
        - Converts our corpus into a list of lists.
        """
        
        corpus_list = []
        
        for tokens_list in self.corpus:
            
            tokens_lists_list = [" ".join(tokens_list[token:token+1]) for token in range(0, len(tokens_list), 1)]
        
            corpus_list.append(tokens_lists_list)
            
        return corpus_list
    
    
    def check_corpus_size(self, corpus_list):
        """
        - Checks the size of our vocabulary.
        """
        
        all_words = []
        
        for each_list in corpus_list:
            for word in each_list:
                all_words.append(word)
        
        all_words = set(all_words)
        vocab_size = len(all_words)
        
        return vocab_size
    
    def get_vocab_index(self, corpus_list):
        """
        - Creates a vocabulary index from our corpus.
        """
        
        tokenizer = preprocessing.text.Tokenizer()
        
        tokenizer.fit_on_texts(corpus_list)
        
        dic_vocabulary = tokenizer.word_index
        
        return dic_vocabulary, tokenizer
    
    def integer_transformer(self, corpus_list, tokenizer):
        """
        - Takes each token from corpus_list and replaces it with its corresponding integer value from word_index dictionary.
        """
        
        encoded_corpus_seqs = tokenizer.texts_to_sequences(corpus_list)
        
        return encoded_corpus_seqs
    
    
    def get_messages_max_length(self, encoded_corpus_seqs):
        """
        - Receives encoded_corpus_seqs as input and calculates the average length of all the sequences.
        """
        
        messages_length = []
        
        for sequence in encoded_corpus_seqs:
            
            messages_length.append(len(sequence))
            
        max_length= round(sum(messages_length)/len(messages_length))
        
        return max_length
    
    def get_padded_seqs(self, encoded_corpus_seqs, max_length):
        """
        - Receives encoded sequences and pads out each sequence to the same length.
        """
        
        padded_sequences = pad_sequences(encoded_corpus_seqs, maxlen=max_length, padding='post')
        
        return padded_sequences
    
    
    def features_test_get_list_of_lists(self):
        """
        - Converts features_test into a list of lists.
        """
        
        features_test_list = []
        
        for tokens_list in self.features_test:
            
            features_test_lists_list = [" ".join(tokens_list[token:token+1]) for token in range(0, len(tokens_list), 1)]
            
            features_test_list.append(tokens_list)
            
        return features_test_list                               

class VerticalModelGetWeightMatrix(object):
    """
    This Python class gets a vector dimension as input and:
    
    - Creates a word embedding matrix that will be used as a weight matrix in the neural network classifier.
    """
    
    # Initializer
    
    def __init__(self, vector_dimension, dic_vocabulary, model_wv):
        
        self.vector_dimension = vector_dimension
        self.dic_vocabulary = dic_vocabulary
        self.model_wv = model_wv
        
    # Instance methods
    
    def embedding_matrix_creator(self):
        """
        - Creates an embedding matrix.
        """
        # startint the matrix with all 0s.
        embedding_matrix = np.zeros((len(self.dic_vocabulary) + 1, self.vector_dimension))
        
        for word, index in self.dic_vocabulary.items():
            
            try:
                
                embedding_matrix[index] = self.model_wv[word]
                
            except:
                
                pass
            
        return embedding_matrix
        