import sys
sys.path.append('..')


import warnings
warnings.filterwarnings('ignore')

# Parser
import argparse

# Get file
from campaigns.getinputdata import CampaignsFileGetInfo

# Exploratory Analysis
from campaigns.eda import ExploratoryDataAnalyser

# Feature Engineering
from campaigns.modeling.preprocessing import KPIcreator

# Texts Preprocessing ~ Cleaning
from campaigns.modeling.nlp import CampaignsCleaner, TextsCleaner

if __name__ == "__main__":
    
    # Making a parser object
    parser = argparse.ArgumentParser(description='Add datafile')
    
    # Creating variables 'input_root', 'input_file', 'output_root', 'output_file'
    parser.add_argument('--input_root', type=str, help='Input root')
    parser.add_argument('--input_file', type=str, help='Input file')
    parser.add_argument('--output_root', type=str, help= 'Output root')
    parser.add_argument('--output_file', type=str, help='Output file')
    
    # Setting the 'args' variable to the values of the parsed arguments
    args = parser.parse_args()
    print('Dataset: ', args.input_file)
    
    # Getting the df
    # Creating an instance
    main_model = CampaignsFileGetInfo(dataroot=args.input_root, datafile=args.input_file)
    print('Campaigns df shape:', main_model.campaigns_df.shape)
    
    # Log Normalize
    # Creating an instance
    main_model_eda = ExploratoryDataAnalyser(main_model.campaigns_df)
    
    # Normalizing
    main_model_eda.log_normalize()
    
    # Detecting outliers
    total_sent_lower_bound, total_sent_upper_bound = main_model_eda.detect_outliers('total_sent')
    opens_lower_bound,opens_upper_bound = main_model_eda.detect_outliers('opens')
    clicks_lower_bound, clicks_upper_bound = main_model_eda.detect_outliers('clicks')
    
    # Handling outliers
    main_model_eda.handle_outliers('total_sent', total_sent_lower_bound, total_sent_upper_bound)
    main_model_eda.handle_outliers('opens', opens_lower_bound, opens_upper_bound)
    main_model_eda.handle_outliers('clicks', clicks_lower_bound, clicks_upper_bound)
    
    # Detecting remaining outliers
    main_model_eda.detect_outliers('total_sent')
    main_model_eda.detect_outliers('opens')
    main_model_eda.detect_outliers('clicks')
    
    
    # Feature Engineering ~ Creating new variables
    
    # Creating an instance
    main_model_feat = KPIcreator(main_model.campaigns_df)
    
    # Creating new variables 'open_rate', 'ctr' and 'ctor'
    main_model_feat.kpi_creator()
    
    
    # Texts Feature Engineering
    
    # Creating an instance
    vertical_model = CampaignsCleaner(main_model_feat.campaigns_df)
    
    # Removing duplicated users
    vertical_model.duplicated_users_remover()
    
    # Removing empty messages
    vertical_model.empty_messages_remover()
    
    
    # Texts Preprocessing ~ Cleaning
    
    # Creating an instance
    vertical_model = TextsCleaner(vertical_model.campaigns_df)
    
    # Removing non words and counting elements
    vertical_model.clean_texts()
    
    
    # Dropping NaN values in 'subject' and 'message'
    vertical_model.campaigns_df.dropna(subset=['clean_message'], inplace=True)
    vertical_model.campaigns_df.dropna(subset=['clean_subject'], inplace=True)
    
    #Counting words
    vertical_model.word_counter('clean_subject', 'clean_message')

    
    # Saving file
    vertical_model.save_final_df(args.output_root, args.output_file)