# Directories Handle
import os


# NLP
import re
import collections

from nltk.util import ngrams
from nltk.tokenize import sent_tokenize

import spacy

# Plots
import matplotlib.pyplot as plt


class CorporaExploratory(object):
    """
    This Python class receives a df containing a text column as the input corpus, and:
    
    - Divides the dataframe by cateories.
    - Creates a text corpus per category.
    - Plots the size of each corpora.
    """
    
    # Initializer
    
    def __init__(self, language_df):
        
        self.language_df = language_df
        
        print('Getting labeled messages')
        self.labeled = self.get_labeled()
        
        print('Tokenizing messages')
        self.labeled['clean_message'] = self.labeled['clean_message'].apply(self.message_tokenizer)
        
        print('Removing empty tokens')
        self.labeled['clean_message'] = self.labeled['clean_message'].apply(self.empty_tokens_remover)
        
        print('Filtering by verticals: media, ecommerce, nonprofit, education, wellness, finance, architecture, government, technology, leisure, entertainment, legal, catering and association')
        self.media = self.labeled[self.labeled['sector'] == "Medios de comunicación, marketing y publicidad"]
        self.ecommerce = self.labeled[self.labeled['sector'] == "Ecommerce"]
        self.nonprofit = self.labeled[self.labeled['sector'] == "Sin ánimo de lucro"]
        self.education = self.labeled[self.labeled['sector'] == "Educación y empleo"]
        self.wellness = self.labeled[self.labeled['sector'] == "Salud, bienestar y cuidado personal"]
        self.finance = self.labeled[self.labeled['sector'] == "Negocios, finanzas y banca"]
        self.architecture = self.labeled[self.labeled['sector'] == "Arquitectura, construcción y sector inmobiliario"]
        self.government = self.labeled[self.labeled['sector'] == "Órganos de gobierno"]
        self.technology = self.labeled[self.labeled['sector'] == "Ordenadores, electrónica y tecnología móvil"]
        self.leisure = self.labeled[self.labeled['sector'] == "Ocio, turismo y experiencias"]
        self.entertainment = self.labeled[self.labeled['sector'] == "Entretenimiento, eventos y relaciones públicas"]
        self.legal = self.labeled[self.labeled['sector'] == "Legal y seguros"]
        self.catering = self.labeled[self.labeled['sector'] == "Restauración"]
        self.association = self.labeled[self.labeled['sector'] == "Asociación cultural o religiosa"]
        
        print('Getting vertical corpora')
        self.media_corpus = self.get_corpus(self.media)
        self.ecommerce_corpus = self.get_corpus(self.ecommerce)
        self.nonprofit_corpus = self.get_corpus(self.nonprofit)
        self.education_corpus = self.get_corpus(self.education)
        self.wellness_corpus = self.get_corpus(self.wellness)
        self.finance_corpus = self.get_corpus(self.finance)
        self.architecture_corpus = self.get_corpus(self.architecture)
        self.government_corpus = self.get_corpus(self.government)
        self.technology_corpus = self.get_corpus(self.technology)
        self.leisure_corpus = self.get_corpus(self.leisure)
        self.entertainment_corpus = self.get_corpus(self.entertainment)
        self.legal_corpus = self.get_corpus(self.legal)
        self.catering_corpus = self.get_corpus(self.catering)
        self.association_corpus = self.get_corpus(self.association)
    

    # Instance Methods
    
    def remove_template(self, message, ngrams_list):
        
        try:
            
            for sentence in ngrams_list:
                
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
        
    def get_labeled(self):
        """
        - Filters the dataframe by labeled messages.
        """
        
        labeled = self.language_df[self.language_df['sector'].notnull()]
        
        return labeled
    
    
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
    
    
    def get_corpus(self, vertical):
        
        corpus = vertical['clean_message']
        
        corpus_list = []
        
        for tokens_list in corpus:
            
            tokens_lists_list = [" ".join(
                tokens_list[token:token+1]) for token in range(0, len(tokens_list), 1)]
        
            corpus_list.append(tokens_lists_list)
            
        all_words = []
        
        for each_list in corpus_list:
            
            for word in each_list:
                
                all_words.append(word)
        
        all_words = set(all_words)
            
        return all_words
    
    def plot_corpora_size(self):
        
        fig, ax = plt.subplots(figsize=(8,12))
        fig.suptitle('corpora size', fontsize = 14)
        
        x_data = ['media', 'ecommerce', 'nonprofit', 'education', 'wellness', 'finance',
                  'architecture', 'government', 'technology', 'leisure', 'entertainment',
                  'legal', 'catering', 'association']

        y_data = [len(self.media_corpus), len(self.ecommerce_corpus), len(self.nonprofit_corpus),
                  len(self.education_corpus), len(self.wellness_corpus), len(self.finance_corpus),
                  len(self.architecture_corpus), len(self.government_corpus),
                  len(self.technology_corpus), len(self.leisure_corpus), len(self.entertainment_corpus),
                  len(self.legal_corpus), len(self.catering_corpus), len(self.association_corpus)]

        plt.barh(x_data, y_data)
        
    
    def keyword_extractor(self, message):
        """
        - Uses a trained pipeline to extract the most important words from a message.
        """
        
        # import a spacy trained pipeline as a python package
        nlp = spacy.load("es_core_news_sm")
        
        doc = nlp(message)
        
        print(doc.ents)
            

class NGramsExploratory(object):
    """
    This Python class receives a df containing a text column as the input corpus, and:
    
    - Divides the dataframe by cateogry.
    - Returns the most common words per cateogory corpus.
    """
   
    # Initializer
    def __init__(self, language_df):
                                                                  
        self.language_df = language_df
        
        print('Getting labeled messages')
        self.labeled = self.language_df[self.language_df['sector'].notnull()]
        
        print("Tokenizing sentences")
        self.labeled['clean_message'] = self.labeled['clean_message'].apply(sent_tokenize)
        
        print('Filtering by verticals: media, ecommerce, nonprofit, education, wellness, finance, architecture, government, technology, leisure, entertainment, legal, catering and association')
        self.media = self.labeled[self.labeled['sector'] == "Medios de comunicación, marketing y publicidad"]
        self.ecommerce = self.labeled[self.labeled['sector'] == "Ecommerce"]
        self.nonprofit = self.labeled[self.labeled['sector'] == "Sin ánimo de lucro"]
        self.education = self.labeled[self.labeled['sector'] == "Educación y empleo"]
        self.wellness = self.labeled[self.labeled['sector'] == "Salud, bienestar y cuidado personal"]
        self.finance = self.labeled[self.labeled['sector'] == "Negocios, finanzas y banca"]
        self.architecture = self.labeled[self.labeled['sector'] == "Arquitectura, construcción y sector inmobiliario"]
        self.government = self.labeled[self.labeled['sector'] == "Órganos de gobierno"]
        self.technology = self.labeled[self.labeled['sector'] == "Ordenadores, electrónica y tecnología móvil"]
        self.leisure = self.labeled[self.labeled['sector'] == "Ocio, turismo y experiencias"]
        self.entertainment = self.labeled[self.labeled['sector'] == "Entretenimiento, eventos y relaciones públicas"]
        self.legal = self.labeled[self.labeled['sector'] == "Legal y seguros"]
        self.catering = self.labeled[self.labeled['sector'] == "Restauración"]
        self.association = self.labeled[self.labeled['sector'] == "Asociación cultural o religiosa"]
        
    # Instance methods
                                                  
    def get_ngrams(self, vertical, my_ngrams, most_common=0):

        sentences = vertical['clean_message']
        
        # creating an aux. corpus
        
        with open('../../../corpora/sentences_file.txt', 'w') as f:
            for elem in sentences:
                for sentence in elem:
                    f.write(sentence + '\n')
        
        with open('../../../corpora/sentences_file.txt', 'r') as f:
            corpus = []
            for line in f:
                corpus.append(line)

        # removing '\n'
        corpus = [re.sub('\n', ' ', sentence) for sentence in corpus]

        # tokenizing
        tokenized = [(sentence.split()) for sentence in corpus]

        flat_list = []

        for sublist in tokenized:
            for word in sublist:
                flat_list.append(word)

        # getting ngrams
        sample_ngrams = ngrams(flat_list, my_ngrams)

        sample_ngrams_freq = collections.Counter(sample_ngrams)

        common_words = sample_ngrams_freq.most_common(most_common)

        sent_list_ngrams = []

        for elem in common_words:
            
            sentence, number = elem
            sentence = ' '.join(sentence)
            sent_list_ngrams.append((sentence, number))

        return sent_list_ngrams
                                             