
import streamlit as st

import sys
sys.path.append('..')

import warnings
warnings.filterwarnings('ignore')

import numpy as np

# Get input
from campaigns.getinputdata import CampaignsFileGetInfo

# EDA
from streamlitapp.eda import ExploratoryDataAnalyser
from plotly.subplots import make_subplots

# Texts Preprocessing
from streamlitapp.nlp import StreamlitAppTextsPreprocessor

# Texts Classification
from streamlitapp.models import ModelApplier, ModelUnwrapper, ExtractKeywordsFromFile

#st.title('Campaigns Performance Dashboard')

# Input data

campaigns_df = CampaignsFileGetInfo(dataroot='../../../datasets/input', datafile='campaigns_w_label_sample_01.csv')

# Performing Exploratory Data Analysis

campaigns_df = ExploratoryDataAnalyser(campaigns_df.campaigns_df)

# Normalize 'total_sent', 'opens' and 'clicks'

campaigns_df.log_normalize()

# Remove Outliers

campaigns_df.remove_total_sent_outliers()
campaigns_df.remove_open_outliers()
campaigns_df.remove_clicks_outliers()

# Data Visualization

menu = ['Performance Analysis', 'Vertical Detector Tool', "KPIs Prediction Tool"]

choice = st.sidebar.selectbox('Please select one option', menu)

if choice == 'Performance Analysis':
    
    st.title('Campaigns Performance Dashboard')
    
    st.subheader('Performance by Customer Category ')
        
        
    radio = st.radio('',('Sent Emails', 'Open Emails', 'Clicks', 'Overview'))
        
    if radio == 'Overview':
        
        st.subheader('All variables: Total Sent, Open Emails & Clicks')
            
        # Displaying boxplots and histograms
            
        campaigns_df.display_overview()
        
    if radio == 'Sent Emails':
            
        st.subheader('Sent Emails Performance')
            
        customer_cat = st.multiselect('Which customer category would you like to observe?', ['all categories', 'category 1', 'category 2','category 3', 'category 4', 'category 5'], ['all categories'])
            
        customer_cat_dict = {'category 1':1, 'category 2':2, 'category 3':3, 'category 4':4, 'category 5':5}
        
        # Displaying boxplots and histograms
            
        if 'all categories' in customer_cat:
                
            campaigns_df.display_bycustomercat_total_sent()

        if 'category 1' in customer_cat:
            
            campaigns_df.display_bycustomercat_total_sent_01()
                
        if 'category 2' in customer_cat:
                
            campaigns_df.display_bycustomercat_total_sent_02()
                
        if 'category 3' in customer_cat:
                
            campaigns_df.display_bycustomercat_total_sent_03()
                
        if 'category 4' in customer_cat:
                
            campaigns_df.display_bycustomercat_total_sent_04()
                
        if 'category 5' in customer_cat:
                
            campaigns_df.display_bycustomercat_total_sent_05()
                              
    if radio == 'Open Emails':
            
        st.subheader('Open Emails Performance')
            
        customer_cat = st.multiselect('Customer Category', ['all categories', 'category 1', 'category 2','category 3', 'category 4', 'category 5'], ['all categories'])
            
        if 'all categories' in customer_cat:
            
            # Displaying boxplot and histogram
                
            campaigns_df.display_bycustomercat_open()
                
        if 'category 1' in customer_cat:
                
            campaigns_df.display_bycustomercat_open_01()
                
        if 'category 2' in customer_cat:
                
            campaigns_df.display_bycustomercat_open_02()
                
        if 'category 3' in customer_cat:

            campaigns_df.display_bycustomercat_open_03()
                
        if 'category 4' in customer_cat:
  
            campaigns_df.display_bycustomercat_open_04()
                
        if 'category 5' in customer_cat:

            campaigns_df.display_bycustomercat_open_05()
               
    if radio == 'Clicks':
            
        st.subheader('Clicks Performance')
            
        customer_cat = st.multiselect('Customer Category', ['all categories', 'category 1', 'category 2','category 3', 'category 4', 'category 5'], ['all categories'])
            
        if 'all categories' in customer_cat:
            
            # Displaying boxplot and histogram
                
            campaigns_df.display_bycustomercat_clicks()
                
        if 'category 1' in customer_cat:
  
            campaigns_df.display_bycustomercat_clicks_01()
                
        if 'category 2' in customer_cat:

            campaigns_df.display_bycustomercat_clicks_02()
                
        if 'category 3' in customer_cat:

            campaigns_df.display_bycustomercat_clicks_03()
                
        if 'category 4' in customer_cat:

            campaigns_df.display_bycustomercat_clicks_04()
                
        if 'category 5' in customer_cat:

            campaigns_df.display_bycustomercat_clicks_05()
                        

        
    
radio_dict = {'Architecture, construction and real state':0,
              'Cultural and religious organizations':1,
              'Catering':2,
              'Ecommerce':3,
              'Education and employment sectors':4,
              'Entertainment, events and public relations':5,
              'Financial services and banking sector':6,
              'Government institution':7,
              'Legal and insurance sectors':8,
              'Leisure and tourism industries':9,
              'Media industry, marketing and adversting':10,
              'Nonprofit organization':11,
              'Technology and electronics sectors, and computer and mobile industries':12,
              'Wellness, health and personal care sectors':13}

inv_radio_dict = {value: key for key, value in radio_dict.items()}


if choice == "Vertical Detector Tool":
    
    st.title('Vertical Detector Tool')
    
    st.subheader('Business Sector Prediction')
    
    second_menu = ['Spanish', 'English', 'French', 'Portuguese']
    
    options = st.selectbox('Please select the language of your email: ', second_menu)
    
    message_input = st.text_input('Type your message:')
    
    classifier_dict = {'Spanish': "../../../datasets/classifiers/vertical_model_multinomial_nb_classifier_spanish",
                       'English': "../../../datasets/classifiers/vertical_model_multinomial_nb_classifier_english",
                       'French': "../../../datasets/classifiers/vertical_model_multinomial_nb_classifier_french",
                       'Portuguese': "../../../datasets/classifiers/vertical_model_multinomial_nb_classifier_portuguese"}
    
    vectorizer_dict = {'Spanish': "../../../datasets/vectorizers/vertical_model_tfidf_vectorizer_spanish",
                       'English': "../../../datasets/vectorizers/vertical_model_tfidf_vectorizer_english",
                       'French': "../../../datasets/vectorizers/vertical_model_tfidf_vectorizer_french",
                       'Portuguese': "../../../datasets/vectorizers/vertical_model_tfidf_vectorizer_portuguese"}
    
    if message_input != '':
        
        # Loading model - MultinomialNB()
        
        model = ModelApplier(classifier_dict[options], vectorizer_dict[options])
        
        # Removing nonwords, tokenizing and stemming
        
        message_cleaner = StreamlitAppTextsPreprocessor(message_input, 'spanish')
        
        message_input = message_cleaner.clean_text(message_input, flag_stemming=True)
        
        print(model)
        
        # Making prediction
        
        prediction = model.classify_message(message_input)
        
        prediction = np.asscalar(prediction)
        
        # Showing prediction
        
        st.write(inv_radio_dict[prediction])


if choice == "KPIs Prediction Tool":
    
    st.title('KPIs Prediction Tool')
    
    second_menu = ['Open Rate Result', 'Ctr Result', 
                   'Ctor Result']
    
    options = st.selectbox('Which KPI would you like to predict?', second_menu)
    
    st.write('Pick your business vertical from the options below: ')
    
    radio = st.radio('',('Architecture, construction and real state',
                         'Cultural and religious organizations',
                         'Catering',
                         'Ecommerce',
                         'Education and employment sectors',
                         'Entertainment, events and public relations',
                         'Financial services and banking sector',
                         'Government institution',
                         'Legal and insurance sectors',
                         'Leisure and tourism industries',
                         'Media industry, marketing and adversting',
                         'Nonprofit organization',
                         'Technology and electronics sectors, and computer and mobile industries',
                         'Wellness, health and personal care sectors'))
    

    if options == 'Open Rate Result':
              
        radio = radio_dict[radio]
        
        # Loading model - logistic regression classifier
        
        model = ModelUnwrapper(radio, '../../../datasets/pipelines/main_model_target_01_pipeline')

        message_input = st.text_input('Type your message:')
        
        if message_input != '':
            
            # Removing nonwords, tokenizing and stemming
        
            message_cleaner = StreamlitAppTextsPreprocessor(message_input, 'spanish')
            
            message_input = message_cleaner.clean_text(message_input, flag_stemming=True)
            
            # Making Prediction
            
            prediction = model.make_predictions_with_subject(message_input)
            
            # showing prediction
            
            if prediction == 0:
                
                st.write("Ok")
                
            else:
                
                st.write("🙁 This email may fall behind its industry's Open Rate benchmark." )
                st.write("🙃 Don't worry! Look at some of the keywords found in our " + "'" + inv_radio_dict[radio]+ "' curated corpus and get some ideas:")
                
                keywords_instance = ExtractKeywordsFromFile(inv_radio_dict[radio], options)
                
                keywords_list = keywords_instance.keywords_retriever()
                
                for keyword in keywords_list:
                    
                    st.write(keyword)
        
    
    if options == 'Ctr Result':
        
        radio = radio_dict[radio]
        
        # Loading model - logistic regression classifier
        
        model = ModelUnwrapper(radio, '../../../datasets/pipelines/main_model_target_02_pipeline')
        
        message_input = st.text_input('Type your message:')
        
        if message_input != '':
            
            # Removing nonwords, tokenizing and stemming
        
            message_cleaner = StreamlitAppTextsPreprocessor(message_input, 'spanish')
            
            message_input = message_cleaner.clean_text(message_input, flag_stemming=True)
            
            # Making Prediction
            
            prediction = model.make_predictions_with_message(message_input)
            
            # showing prediction
            
            if prediction == 0:
                
                st.write("Ok")
                
            else:
                
                st.write("🙁 This email may fall behind its industry's CTR benchmark.")
                st.write("🙃 Don't worry! Look at some of the keywords found in our " + "'" + inv_radio_dict[radio]+ "' curated corpus and get some ideas:")
                
                keywords_instance = ExtractKeywordsFromFile(inv_radio_dict[radio], options)
                
                keywords_list = keywords_instance.keywords_retriever()
                
                for keyword in keywords_list:
                    
                    st.write(keyword)
            
    
    if options == 'Ctor Result':
        
        radio = radio_dict[radio]
        
        # Loading model - logistic regression classifier
        
        model = ModelUnwrapper(radio, '../../../datasets/pipelines/main_model_target_03_pipeline')
        
        message_input = st.text_input('Type your message:')
        
        if message_input != '':
            
            # Removing nonwords, tokenizing and stemming
        
            message_cleaner = StreamlitAppTextsPreprocessor(message_input, 'spanish')
            
            message_input = message_cleaner.clean_text(message_input, flag_stemming=True)
            
            # Making Prediction
            
            prediction = model.make_predictions_with_message(message_input)
            
            # showing prediction
            
            if prediction == 0:
                
                st.write("Ok")
                
            else:
                
                st.write("🙁 This email may fall behind its industry's CTOR benchmark.")
                st.write("🙃 Don't worry! Look at some of the keywords found in our " + "'" + inv_radio_dict[radio]+ "' curated corpus and get some ideas:")
                
                keywords_instance = ExtractKeywordsFromFile(inv_radio_dict[radio], options)
                
                keywords_list = keywords_instance.keywords_retriever()
                
                for keyword in keywords_list:
                    
                    st.write(keyword)
