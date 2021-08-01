import sys
sys.path.append('..')


import warnings
warnings.filterwarnings('ignore')

# Parser
import argparse

# Get file
from campaigns.getinputdata import LanguageFileGetInfo

# Texts Preprocessing with NLP
from campaigns.modeling.nlp import StopWordsRemover

if __name__ == "__main__":
    
    # Making a parser object
    parser = argparse.ArgumentParser(description='Add datafile')
    
    # Creating variables 'input_root', 'input_file', 'output_root', 'output_file', 'sw_remover'
    parser.add_argument('--input_root', type=str, help='Input root')
    parser.add_argument('--input_file', type=str, help='Input file')
    parser.add_argument('--output_root', type=str, help= 'Output root')
    parser.add_argument('--output_file', type=str, help='Output file')
    parser.add_argument('--sw_remover', type=str, help='StopWords Remover')
    
    # Setting the 'args' variable to the values of the parsed arguments
    args = parser.parse_args()
    
    # Getting the df
    # Creating an instance
    language_instance = LanguageFileGetInfo(dataroot=args.input_root, datafile=args.input_file)

    language_instance.language_df.dropna(subset=['clean_message'], inplace=True)
    print('Language: ', language_instance.language_df.shape)
    
    # Removing stopwords
    # Creating an instance
    sw_instance = StopWordsRemover(language_instance.language_df, args.sw_remover)
                        
    sw_instance.remove_stopwords()
                        
    # Saving file
    sw_instance.save_final_df(args.output_root, args.output_file)
                        