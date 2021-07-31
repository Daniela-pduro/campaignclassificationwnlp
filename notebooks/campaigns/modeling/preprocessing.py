# Directories Handle
import os

# Timing
import time

# Data Processing
import numpy as np
import pandas as pd
import functools

# NLP
import spacy

# Data Preprocessing Pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Cleaning Texts
import re

# Data Visalization
import matplotlib.pyplot as plt
import seaborn as sns


class KPIcreator(object):
    """
    This python class reads the tabular data contained in a dataframe and:
    
    - Creates marketing KPIs variables: `open_rate`, `ctr` & `ctor`. 
    - Plots variables distribution.
    - Detects outliers.
    - Handles outliers.
    - Filters by vertical.
    - Finds the campaigns with the best results by vertical.
    - Extracts special keywords from best results messages.
    
    """
    
    # Initializer
    def __init__(self, campaigns_df):
        
        self.campaigns_df = campaigns_df
        
    # Instance Methods
    
    def kpi_creator(self):
        """
        - Creates a news variables called 'open_rate', 'ctr' and 'ctor'
        """
        
        # Creating New Variables
        
        self.campaigns_df['open_rate'] = self.campaigns_df['opens'] / self.campaigns_df['total_sent']
        
        self.campaigns_df['ctr'] = self.campaigns_df['clicks'] / self.campaigns_df['total_sent']
        
        self.campaigns_df['ctor'] = self.campaigns_df['clicks'] / self.campaigns_df['opens']
          
    
    def kpi_describe(self):
        """
        - Displays 'open_rate', 'ctr' and 'ctor' stats by customer category
        """
        
        # Showing stats by customer category
        
        open_rate_table = self.campaigns_df.groupby('customer_cat')[['open_rate']].describe()
        
        ctr_table = self.campaigns_df.groupby('customer_cat')[['ctr']].describe()
        
        ctor_table = self.campaigns_df.groupby('customer_cat')[['ctor']].describe()
        
        return open_rate_table, ctr_table, ctor_table
    
    def plot_outliers(self):
        """
        - Displays Boxplots to identify outliers.
        """
        
        # Plotting box plots
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(1, figsize=(8,5))
        
        plt.suptitle("Main Variables Boxplots - Outliers Detection", fontsize=16)
        
        total_sent = self.campaigns_df['open_rate']
        opens = self.campaigns_df['ctr']
        clicks = self.campaigns_df['ctor']
        
        colors = ['#E69F00','#56B4E9','#2CA02C']
        colors_total_sent = dict(color=colors[0])
        colors_opens = dict(color=colors[1])
        colors_clicks = dict(color=colors[2])
        labels = ['Open Rate','CTR', 'CTOR']
        
        ax.boxplot(total_sent, positions=[0],labels=[labels[0]], 
                   boxprops=colors_total_sent, medianprops=colors_total_sent,
                   whiskerprops=colors_total_sent, capprops=colors_total_sent,
                   flierprops=dict(markeredgecolor=colors[0]))
        
        ax.boxplot(opens, positions=[1], labels=[labels[1]],
                   boxprops=colors_opens, medianprops=colors_opens,
                   whiskerprops=colors_opens, capprops=colors_opens,
                   flierprops=dict(markeredgecolor=colors[1]))
        
        ax.boxplot(clicks, positions=[2], labels=[labels[2]],
                   boxprops=colors_clicks, medianprops=colors_clicks,
                   whiskerprops=colors_clicks, capprops=colors_clicks,
                   flierprops=dict(markeredgecolor=colors[2]))
    
  
    def bivariate_plotter(self,variable):
        """
        - Performs a bivariate analysis - Customer Cat vs. Total Sent
        - Displays a box plot
        - Displays a bar chart.
        """
        
        # Plotting box plots
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(1, figsize=(8,5))
        
        plt.suptitle(str(variable) + ' by customer category', fontsize=16)
                
        colors_dict = {'open_rate': '#E69F00', 'ctr':'#56B4E9', 'ctor': '#2CA02C'}
        
        color = [colors_dict[variable]]
        colors_customer = dict(color=color[0])
        labels = ['Category 01','Category 02', 'Category 03', 'Category 04', 'Category 05']
        
        for customer in range(1,6):
        
            ax.boxplot(self.campaigns_df[self.campaigns_df['customer_cat']==customer][variable], 
                   positions= [customer-1], labels=[customer], 
                   boxprops=colors_customer, 
                   medianprops=colors_customer,
                   whiskerprops=colors_customer, 
                   capprops=colors_customer,
                   flierprops=dict(markeredgecolor=color[0]))
        
        # Plotting bar charts
    
        for customer in range(1,6):
        
            fig, ax = plt.subplots(1, figsize=(8,5))
            plt.suptitle('Category ' + str(customer), fontsize=16)
        
            ax.hist(self.campaigns_df[self.campaigns_df['customer_cat']==customer][variable],
                    bins=15, color=colors_dict[variable], label=variable, alpha=0.75)
            

    def detect_outliers(self, variable):
        """
        - Displays outliers from a given variable passed as argument to the function.
        - Returns fence values.
        """
        
        q1 = self.campaigns_df[variable].quantile(0.25)
        q3 = self.campaigns_df[variable].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        print(f'\nOutliers:')
        print(len(self.campaigns_df[(self.campaigns_df[variable] < lower_bound)|
                                 (self.campaigns_df[variable] > upper_bound)]))
        
        
        return lower_bound, upper_bound
        
    
    def handle_outliers(self, variable, lower_bound, upper_bound):
        """
        - Receives fence values.
        - Substitutes outliers by fence values.
        - Displays mean value after handling outliers.
        """

        self.campaigns_df.loc[(self.campaigns_df[variable] < lower_bound), variable] = lower_bound
        self.campaigns_df.loc[(self.campaigns_df[variable] > upper_bound), variable] = upper_bound

        print(f'Mean value after outliers treatment: {self.campaigns_df[variable].mean():.2f}')

                
class BenchmarkCreator(object):
    """
    This python class reads the tabular data contained in a dataframe and:
    
    - Creates benchmark variables by customer category and vertical: `open_rate_benchmark`, 
    `ctr_benchmark` & `ctor_benchmark` by customer category.
    
    """
     
    # Initializer
    def __init__(self, campaigns_w_vertical_df):
        
        self.campaigns_w_vertical_df = campaigns_w_vertical_df
        pd.options.display.max_columns = None
        
    # Instance Methods
    
    def benchmarks_aux_creator(self):
        """
        - Calculates the mean of 'open_rate', 'ctr' and 'ctor' variables by
        customer_category and vertical and saves the result in an auxiliary dataframe.
        
        """
        
        aux = self.campaigns_w_vertical_df[['customer_cat','vertical','open_rate','ctr','ctor']]\
        .groupby(['vertical', 'customer_cat'], as_index=False).mean()
        
        return aux
    
     
    def benchmark_variable_creator(self, aux):
        """
        - Creates new variables: 'open_rate_benchmark', 'ctr_benchmark',
        'ctor_benchmark'
        
        """
       
        open_rate_benchmark_list = []
        ctr_benchmark_list = []
        ctor_benchmark_list = []
        
        for vertical, category in zip(self.campaigns_w_vertical_df.vertical, self.campaigns_w_vertical_df.customer_cat):
            for i in range(0, len(aux)):
                if vertical == aux['vertical'].iloc[i] and category == aux['customer_cat'].iloc[i]:
                    open_rate_benchmark_list.append(aux.loc[i,'open_rate'])
                    ctr_benchmark_list.append(aux.loc[i,'ctr'])
                    ctor_benchmark_list.append(aux.loc[i,'ctor'])
        
        
        self.campaigns_w_vertical_df['open_rate_benchmark'] = open_rate_benchmark_list
        self.campaigns_w_vertical_df['ctr_benchmark'] = ctr_benchmark_list
        self.campaigns_w_vertical_df['ctor_benchmark'] = ctor_benchmark_list


class Benchmarker(object):
    """
    This python class reads the tabular data contained in a dataframe and:
    
    - Creates new variables.

    """
    # Initializer
    def __init__(self, campaigns_w_vertical_df):
        
        self.campaigns_w_vertical_df = campaigns_w_vertical_df
        self.vertical_dict = {'architecture': 0,
                              'association': 1,
                              'catering': 2,
                              'ecommerce': 3,
                              'education': 4,
                              'entertainment': 5,
                              'finance': 6,
                              'government': 7,
                              'legal': 8,
                              'leisure': 9,
                              'media': 10,
                              'nonprofit': 11,
                              'technology': 12,
                              'wellness': 13}
        
    # Instance Methods
    
    def campaign_benchmarker(self, variable, benchmark):
        """
        - Creates a new variables.
        """
        result_list = []
        
        for index, row in self.campaigns_w_vertical_df.iterrows():
            
            if row[variable] < row[benchmark]:
                
                result_list.append(True)
            
            else:
                
                result_list.append(False)
        
        self.campaigns_w_vertical_df[str(variable) +'_result'] = result_list
        
        
    def get_top_messages(self, variable, ntop):
        """
        - Receives the name of a variable and the number of messages from the top of the list wanted to be obsserved.
        - Groups the df by vertical.
        - Sorts the dataframe in descending order.
        - Return a df with the n top messages.
        """
        
        top_n_df = self.campaigns_w_vertical_df.groupby('vertical').apply(
            lambda x : x.sort_values(by = variable, ascending = False).head(ntop)).reset_index(drop=True)
        
        return top_n_df
    
    def keywords_extractor(self, message):
        """
        - Uses a trained pipeline to extract the most important words from a message.
        """
        
        # import a spacy trained pipeline as a python package
        nlp = spacy.load("es_core_news_sm")
        
        doc = nlp(message)
        
        return doc.ents

    def keywords_list_creator(self, top_n_df, ntop, vertical, text):
        
        keywords_list = []
        
        for message in range(0,ntop):
            
            keywords = self.keywords_extractor(
                top_n_df[top_n_df['vertical'] == self.vertical_dict[vertical]][text].iloc[message])
            
            keywords_list.append(keywords)
            
        flat_list = [item for sublist in keywords_list for item in sublist]
        
        flat_list = list(map(str, flat_list))
        
        unique = functools.reduce(lambda l, x: l.append(x) or l if x not in l else l, flat_list, [])
        
        return unique   

    def add_keywords_to_file(self, file_path, keywords_list):
        """
        - Adds keywords to file.
        """
        
        with open(file_path,'a') as keywords:
            for sentence in keywords_list:
                keywords.write(sentence + '\n')

                
class CustomerCategoryGetDummies(object):
    """
    This python class reads the tabular data contained in a dataframe and:
    
    - Encodes the main categorical variable of the dataframe 'customer_cat'.
    
    """
    
    # Initializer
    def __init__(self, campaigns_w_vertical_df):
        
        self.campaigns_w_vertical_df = campaigns_w_vertical_df
    
    # Instance Methods    
    def encoder(self):
        """
        - Creates customer_cat dummy variables.
        
        """
        
        self.campaigns_w_vertical_df = pd.get_dummies(self.campaigns_w_vertical_df, columns = ['customer_cat'])


class VerticalGetDummies(object):
    """
    This python class reads the tabular data contained in a dataframe and:
    
    - Encodes the main categorical variable of the dataframe 'vertical'.
    
    """
    
    # Initializer
    def __init__(self, campaigns_w_vertical_df):
        
        self.campaigns_w_vertical_df = campaigns_w_vertical_df
    
    # Instance Methods    
    def encoder(self):
        """
        - Creates customer_cat dummy variables.
        
        """
        
        self.campaigns_w_vertical_df = pd.get_dummies(self.campaigns_w_vertical_df, columns = ['vertical'])

class VerticalEncoder(object):
    """
    This python class receives as input the tabular data contained in a dataframe and:
    
    - Changes the names of the categories under the colum `sector` to abbreviations in English.
    - Displays how many campaigns are classified and how many are not under the column `sector`.
    - Displays the total amount of different categories existing under the column `sector`.
    - Displays the relative frequency of campaigns per category excludind NaN values (unclassified campaigns). 
    - Displays a bar plot with the number of campaigns per category.
    - Encodes the values in the column sector.
    - Prepares a dictionary for future reference.
   
   """
    # Intitalizer
    
    def __init__(self, language_df):
        
        self.language_df = language_df
        self.sectors = self.language_df['sector'].unique()
        self.vertical_abbrev = {
            "Medios de comunicación, marketing y publicidad": "media",
            "Ecommerce": "ecommerce",
            "Sin ánimo de lucro": "nonprofit",
            "Educación y empleo": "education",
            "Salud, bienestar y cuidado personal": "wellness",
            "Negocios, finanzas y banca": "finance",
            "Arquitectura, construcción y sector inmobiliario": "architecture",
            "Órganos de gobierno": "government",
            "Ordenadores, electrónica y tecnología móvil": "technology",
            "Ocio, turismo y experiencias": "leisure",
            "Entretenimiento, eventos y relaciones públicas": "entertainment",
            "Legal y seguros": "legal",
            "Restauración": "catering",
            "Asociación cultural o religiosa": "association"
        }
        
    
    # Instance Methods
    
    def change_names(self):
        """
        - Changes the names of the categories under the colum `sector` to abbreviations in English.
        """
        self.language_df['sector'] = self.language_df['sector'].map(self.vertical_abbrev).fillna(self.language_df['sector'])
    
        
    def check_categories(self, including_unlabeled=False):
        """
        - Displays the total amount of different categories existing under the column `sector`.
        - Displays the relative frequency of campaigns per category excludind NaN values (unclassified campaigns).
        """
        if including_unlabeled == False:
            
            print("Number of categories:", len(self.language_df[self.language_df['sector'].notnull()]['sector'].unique()))
            print(round(self.language_df[self.language_df['sector'].notnull()]['sector'].value_counts() / self.language_df[self.language_df['sector'].notnull()].shape[0] * 100, 2))
            
        else:
            
            print("Number of categories:", len(self.language_df['sector'].unique()))
            print(round(self.language_df['sector'].value_counts()/self.language_df.shape[0] * 100, 2))
    
    def plot_categories(self):
        """
        - Displays a bar plot with the number of campaigns per category.
        """
        fig, ax = plt.subplots()
        fig.suptitle('vertical', fontsize = 12)
        self.language_df['sector'].reset_index().groupby('sector').count().\
        sort_values(by='index').plot(kind='barh', legend=False, ax=ax).grid(axis='x')
        
    def label_encoding(self):
        """
        - Encodes the values in the column sector.
        """
        self.language_df['sector_cat_id'] = self.language_df['sector'].astype('category')
        label_mapping = self.language_df['sector_cat_id'].cat.categories
        self.language_df['sector_cat_id'] = self.language_df['sector_cat_id'].cat.codes

    def label_encoding_df(self):
        """
        - Creates and displays a dataframe with the `sector` and its code for future reference.
        """
        
        sector_id_df = self.language_df[['sector','sector_cat_id']]\
                .drop_duplicates()\
                .sort_values('sector_cat_id')\
                .reset_index(drop=True)
        
        return sector_id_df
                     
    def small_samples_remover(self, nsamples):
        """
        - Receives a desired minimum number of samples.
        - Removes rows from df if the number of samples in 'sector' are less than nsamples.
        """
        
        verticals = self.language_df['sector'].unique()
        
        for vertical in verticals:
            
            if len(self.language_df[self.language_df['sector']==vertical]) <= nsamples:

                self.language_df.drop(
                    self.language_df[self.language_df['sector'] == vertical].index, inplace=True)

class MergePredicted(object):
    """
    This Python class receives `predictions` and 
    
    - Displays the frequency of predicted categories.
    - Merges predicted classes back to original messages.
    
    """
    def __init__ (self, predictions, language_df, test_df):
        
        self.predictions = predictions
        self.language_df = language_df
        self.language_df = self.language_df[self.language_df['sector_cat_id'] != -1]
        self.test_df = test_df
        self.results_df = self.merge_predicted()
    
    def get_predicted_frequency(self):
        
        unique, counts = np.unique(self.predictions, return_counts=True)
        
        print(np.asarray((unique, counts)).T)
        
        return unique, counts
    
    def merge_predicted(self):
        
        self.test_df = self.test_df.reset_index(drop=True)
        
        predictions_df = pd.DataFrame(self.predictions, columns=['predicted_class'])
        results_df = pd.merge(self.test_df, predictions_df, how='outer', left_index=True, right_index=True)
        
        return results_df
    
    def get_vertical_classified_df(self):
        
        vertical_classified_df = pd.concat([self.language_df, self.results_df])
        vertical_classified_df['predicted_class'] = vertical_classified_df['predicted_class'].fillna(0)
        vertical_classified_df['predicted_class'] = vertical_classified_df['predicted_class'].astype(int)
        vertical_classified_df['vertical'] = vertical_classified_df[['predicted_class', 'sector_cat_id']].max(axis=1)
        vertical_classified_df.drop(['predicted_class', 'sector_cat_id', 'sector'], axis=1, inplace=True)
        
        return vertical_classified_df
    
    def save_final_df(self, vertical_classified_df, path, filename):
        """
        -  Saves the file in the specified path.
        """
        vertical_classified_df.to_csv(os.path.join(path,filename), header=None)

class IrrelevantPhrasesRemover(object):
    """
    This python class reads the tabular data contained in a dataframe and:
    
    - Removes irrelevant phrases.
    """
    # Initializer
    
    def __init__(self, campaigns_df, document):
        
        start = time.time()
        self.campaigns_df = campaigns_df
        self.document = document
        print('1. Number of seconds since it started running: ' + str(time.time()-start))
        self.remove_irrelevant_phrases(document)
        print('12. Number of seconds since it started running: ' + str(time.time()-start))
    
    # Instance methods
    
    def word_counter(self):
        """
        - Counts the total number of words in messages.
        """
        
        return  self.campaigns_df['clean_message'].apply(lambda x: len(x.split(' '))).sum()
    
    def template_remover(self, message, ngrmas_list):
        """
        - Receives message as input and returns a clean version.
        """
        
        try:
            
            for sentence in ngrmas_list:
                
                # removing n-grams
                message = message.replace(sentence, ' ')
                
                # tokenization
                message = message.split(' ')
                
                # removing extra empty tokens
                message = ' '.join(message).split()
                
                # back to string
                message = ' '.join(message)
            
            return message
        
        except:
            
            return message
        
    def save_final_df(self, path, filename):
        """
        -  Saves the df in the specified path.
        """
        self.campaigns_df.to_csv(os.path.join(path,filename), header=None)
            
            
    def remove_template(self, irrelevant_list):
        """
        - Applies template_remover method over the dataframe.
        - Counts the total number of words in messages after cleaning, and prints out the result.
        """
        print('7. Applying the remove_template method over the dataframe')
        self.campaigns_df['clean_message'] = self.campaigns_df['clean_message'].apply(self.template_remover, args=(irrelevant_list,))
    
    
    def remove_irrelevant_phrases(self, phrases_data_file):
        
        # Printing the dataframe shape before processing
        print('2. Campaigns shape before removing irrelevant phrases:', self.campaigns_df.shape) 
    
        # Removing empty messages
        print('3. Removing empty messages')     
        self.campaigns_df['clean_message'] = self.campaigns_df['clean_message'].replace(r'', np.NaN)
        self.campaigns_df.dropna(subset=['clean_message'], inplace=True)
    
        # Counting total words before removing irrelevant phrases
        ttl_words_w_templates = self.word_counter()
        print('4. Number of words before removing irrelevant phrases: ', ttl_words_w_templates)
    
        # Removing irrelevant phrases from messages
        print('5. Starting the process of removing irrelevant phrases from messages')
        with open(phrases_data_file, 'r') as irrelevant:
            
            irrelevant_list = []
            
            for sentence in irrelevant:
                
                irrelevant_list.append(sentence)
                
            irrelevant_list = [re.sub('\n','', sentence) for sentence in irrelevant_list]
            print('6. File appended')
        
        self.remove_template(irrelevant_list)
        
        # Counting total words after removing irrelevant phrases
        ttl_words_wo_templates = self.word_counter()
        print('8. Number of words after having removed irrelevant phrases: ', ttl_words_wo_templates)
        
        # Checking how many words have been removed
        print('9. Number of words removed: ', ttl_words_w_templates - ttl_words_wo_templates)
               
        # Removing empty messages
        self.campaigns_df['clean_message'] = self.campaigns_df['clean_message'].replace(r'', np.NaN)
        self.campaigns_df.dropna(subset=['clean_message'], inplace=True)
        print('10. Campaigns shape after having removed empty messages:', self.campaigns_df.shape)
    
        # Counting words after removing empty messages
        final_num_of_words = self.word_counter()
        print('11. Number of words after having removed empty messages: ', final_num_of_words)
    
        return self.campaigns_df
    
    
class ClassifiedCampaignsGetPipeline(object):
    """
    
    This Python class receives a dataframe, features and target as input, and:
    
    - Removes empty rows from 'subject'.
    """
    
    # Initializer
    def __init__(self, campaigns_w_vertical_df, features, target):
        
        self.campaigns_w_vertical_df = campaigns_w_vertical_df
        self.campaigns_w_vertical_df.dropna(subset=['clean_subject'], inplace=True)
        self.features = features
        self.target = target
    
    # Instance Methods
    
    def get_cat_transformer(self):
        
        cat_transformer = Pipeline(steps=[('cat_imputer',
                                 SimpleImputer(strategy='constant')),
                                ('cat_ohe', OneHotEncoder(handle_unknown='ignore'))])
        
        return cat_transformer
    
    def get_text_transformer(self):
        
        text_transformer = Pipeline(steps=[('text_bow', 
                                  CountVectorizer())])
        
        return text_transformer
    
    def get_column_transformer(self, cat_transformer, text_transformer, text_column):
        
        ct = ColumnTransformer(transformers=[
            ('cat', cat_transformer, ['vertical']),
            ('text', text_transformer, text_column)])
        
        return ct
    
    def train_test_splitter(self):
        """
        - Splits the data in test and train data sets.
        """
        
        features_train, features_test, target_train, target_test = train_test_split(self.features, self.target, test_size=0.1)
        
        return features_train, features_test, target_train, target_test
        
    def get_pipeline(self, ct):
        
        pipeline = Pipeline(steps=[('feature_engineer', ct), ('LR', LogisticRegression())])
        
        return pipeline