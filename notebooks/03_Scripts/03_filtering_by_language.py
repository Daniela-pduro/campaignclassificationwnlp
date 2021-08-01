import sys
sys.path.append('..')


import warnings
warnings.filterwarnings('ignore')

# Parser
import argparse

# Get file
from campaigns.getinputdata import CampaignsAfterEDAFileGetInfo

# Language Filtering
from campaigns.modeling.nlp import LanguageDetector

if __name__ == "__main__":
    
    # Making a parser object
    parser = argparse.ArgumentParser(description='Add datafile')
    
    # Creating variables 'input_root', 'input_file', 'output_root', 'output_file'
    parser.add_argument('--input_root', type=str, help='Input root')
    parser.add_argument('--input_file', type=str, help='Input file')
    parser.add_argument('--output_root', type=str, help= 'Output root')
    
    # Setting the 'args' variable to the values of the parsed arguments
    args = parser.parse_args()
    print(args.input_file)
    
    # Getting the df
    # Creating an instance
    vertical_model = CampaignsAfterEDAFileGetInfo(dataroot=args.input_root, datafile=args.input_file)
    
    # Detecting language
    # Creating an instance
    vertical_model = LanguageDetector(vertical_model.campaigns_df)
    
    # Show Languages
    
    vertical_model.replace_language(original_language_code='')
    vertical_model.replace_language(original_language_code='.')
    vertical_model.replace_language(original_language_code=',')
    
    print(vertical_model.languages_list)
    
    print('Number of campaigns in Spanish: ')
    print(len(vertical_model.campaigns_df[vertical_model.campaigns_df['language'] =='es']))
    
    # Saving language files
    vertical_model.language_filter(args.output_root)